import torch
import torch.nn as nn
from .layers.attention import AttentionBuilder
from .layers.transformer import TransformerLayer
from .layers.convolution import ConvolutionLayer
from timm.layers import trunc_normal_

class SkipConnection(nn.Module):
    def __init__(self, skip_type='concat', skip_index=None):
        super(SkipConnection, self).__init__()
        self.skip_type = "none" if skip_type is None else str(skip_type).strip().lower()
        self.skip_index = skip_index

    def forward(self, current_feat, current_global_feat, feat_history, global_feat_history):
        if self.skip_type == "none":
            return current_feat, current_global_feat
        if feat_history is None or len(feat_history) == 0:
            return current_feat, current_global_feat

        skip_feat = feat_history[self.skip_index]
        skip_global_feat = global_feat_history[self.skip_index]

        if self.skip_type == 'concat':
            feat = torch.cat((current_feat, skip_feat), dim=1)
            global_feat = torch.cat((current_global_feat, skip_global_feat), dim=1)
        elif self.skip_type == 'add':
            feat = current_feat + skip_feat
            global_feat = current_global_feat + skip_global_feat
        else:
            raise ValueError(f"Unsupported skip method: {self.skip_type}")

        return feat, global_feat

class SPAN_Block(nn.Module):
    def __init__(self, config, layer_index=0, mask_padding=False):
        super(SPAN_Block, self).__init__()
        self.config = config
        self.execution_order = []
        self.modules_dict = nn.ModuleDict()

        if mask_padding:
            self.mask_padding = (layer_index == 0)
        else:
            self.mask_padding = False

        self._build_modules(config, layer_index)

    def _build_modules(self, config, layer_index):

        attn_builder = None

        if isinstance(config, dict):
            config_items = config.items()
        elif isinstance(config, list):

            config_items = config
        else:
            raise ValueError("Config must be either dict or list of tuples")

        for key, value in config_items:
            if key == 'convs':

                conv_layers = []
                for conv_config in value:
                    conv_config = conv_config.copy()
                    if self.mask_padding:
                        conv_config['mask_padding'] = True
                    conv_layers.append(ConvolutionLayer(**conv_config))

                module_name = f'convs_{len(self.execution_order)}'
                self.modules_dict[module_name] = nn.ModuleList(conv_layers)
                self.execution_order.append(('convs', module_name))

            elif key == 'attn_builder':

                attn_config = value.copy()
                if self.mask_padding:
                    attn_config['mask_padding'] = True
                attn_builder = AttentionBuilder(**attn_config)

                self.execution_order.append(('attn_builder', attn_builder))

            elif key == 'trans':

                if attn_builder is None:
                    raise ValueError("'trans' requires 'attn_builder' to be defined before it in config")

                trans_layers = []
                for trans_config in value:
                    trans_layers.append(TransformerLayer(w_RPB=attn_builder.window_size, **trans_config))

                module_name = f'trans_{len(self.execution_order)}'
                self.modules_dict[module_name] = nn.ModuleList(trans_layers)
                self.execution_order.append(('trans', module_name))

            elif key == 'skipc':
                skip_module = SkipConnection(skip_type=value['skip_type'], skip_index=value['skip_index'])
                self.execution_order.append(('skipc', skip_module))

    def forward(
        self,
        ins_pos,
        feat,
        global_feat=None,
        spatial_shape=None,
        pos_dict=None,
        feat_history=None,
        global_feat_history=None,
        return_pre_skip_global_feat=False,
    ):
        pre_skip_feat = None
        pre_skip_global_feat = None

        for step in self.execution_order:
            step_type = step[0]

            if step_type == 'convs':

                module_name = step[1]
                conv_layers = self.modules_dict[module_name]
                for conv_layer in conv_layers:
                    ins_pos, feat, spatial_shape, pos_dict = conv_layer(ins_pos, feat, spatial_shape, pos_dict)
                    global_feat = conv_layer.conv_forward(global_feat)

            elif step_type == 'attn_builder':

                attn_builder = step[1]
                compute_1, compute_2 = attn_builder(ins_pos, global_feat.size(0), mode=attn_builder.mode)

            elif step_type == 'trans':

                module_name = step[1]
                trans_layers = self.modules_dict[module_name]

                for trans_layer in trans_layers:
                    feat, global_feat = trans_layer(feat, global_feat, g1=compute_1, g2=compute_2, ins_pos=attn_builder.positions)

            elif step_type == 'skipc':
                if return_pre_skip_global_feat and pre_skip_global_feat is None:
                    pre_skip_feat = feat
                    pre_skip_global_feat = global_feat
                skip_modules = step[1]
                feat, global_feat = skip_modules(feat, global_feat, feat_history, global_feat_history)

        if return_pre_skip_global_feat:
            if pre_skip_global_feat is None:
                pre_skip_feat = feat
                pre_skip_global_feat = global_feat
            return ins_pos, feat, global_feat, spatial_shape, pos_dict, pre_skip_feat, pre_skip_global_feat
        return ins_pos, feat, global_feat, spatial_shape, pos_dict

def create_block(config, layer_index, mask_padding=False):

    return SPAN_Block(config, layer_index, mask_padding)

class SPAN_Encoder(nn.Module):
    def __init__(self, blocks, embed_dim, token_init_types):
        super(SPAN_Encoder, self).__init__()
        self.blocks = nn.ModuleList(blocks)
        self.token_init_types = token_init_types or []

        token_init_types = token_init_types or []
        self.token_specs = []
        random_tokens = []
        fixed_tokens = []
        learnable_tokens = []

        for t_type in token_init_types:
            if isinstance(t_type, (int, float)):
                token = torch.randn(1, embed_dim)
                trunc_normal_(token, mean=0.0, std=float(t_type))
                self.token_specs.append(("rand", len(random_tokens)))
                random_tokens.append(token)
                continue

            if isinstance(t_type, str):
                name = t_type.strip()
                lower = name.lower()

                if lower in ("max", "mean", "std"):
                    self.token_specs.append((lower, None))
                    continue
                parsed = False
                for prefix in ("fix", "lrn"):
                    if not lower.startswith(prefix):
                        continue
                    rest = lower[len(prefix):].lstrip(":_")
                    if not rest:
                        std = 1e-4
                    else:
                        try:
                            std = float(rest)
                        except ValueError:
                            break
                    token = torch.randn(1, embed_dim)
                    trunc_normal_(token, mean=0.0, std=std)
                    if prefix == "fix":
                        self.token_specs.append(("fix", len(fixed_tokens)))
                        fixed_tokens.append(token)
                    else:
                        self.token_specs.append(("lrn", len(learnable_tokens)))
                        learnable_tokens.append(nn.Parameter(token))
                    parsed = True
                    break

                if parsed:
                    continue

            self.token_specs.append(("none", None))

        if random_tokens:
            self.register_buffer("random_tokens", torch.cat(random_tokens, dim=0))
        else:
            self.random_tokens = None

        if fixed_tokens:
            self.register_buffer("fixed_tokens", torch.cat(fixed_tokens, dim=0))
        else:
            self.fixed_tokens = None

        if learnable_tokens:
            self.learnable_tokens = nn.ParameterList(learnable_tokens)
        else:
            self.learnable_tokens = nn.ParameterList()

    def forward(self, ins_pos, feat):
        tokens = []
        for kind, idx in self.token_specs:
            if kind == "max":
                tokens.append(feat.max(dim=0, keepdim=True)[0])
            elif kind == "mean":
                tokens.append(feat.mean(dim=0, keepdim=True))
            elif kind == "std":
                tokens.append(feat.std(dim=0, keepdim=True))
            elif kind == "rand":
                tokens.append(self.random_tokens[idx:idx + 1])
            elif kind == "fix":
                tokens.append(self.fixed_tokens[idx:idx + 1])
            elif kind == "lrn":
                tokens.append(self.learnable_tokens[idx])

        if len(tokens) > 0:
            global_feat = torch.cat(tokens, dim=0)
        else:
            global_feat = feat.new_empty((0, feat.shape[1]))

        feats = []
        global_feats = []
        coords = []
        spatial_shape = (ins_pos.int().max(dim=0)[0]+1).tolist()
        pos_dict = None
        for idx, block in enumerate(self.blocks):
            ins_pos, feat, global_feat, spatial_shape, pos_dict = block(ins_pos, feat, global_feat=global_feat,
                                                                        spatial_shape=spatial_shape, pos_dict=pos_dict)
            global_feats.append(global_feat)
            feats.append(feat)
            coords.append(ins_pos)
        return coords, feats, global_feats, spatial_shape, pos_dict

class SPAN_Decoder(nn.Module):
    def __init__(self, blocks):
        super(SPAN_Decoder, self).__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, ins_pos, feat, global_feat, spatial_shape, pos_dict, return_pre_output_global_feat=False):
        decoded_feats = []
        coords = []
        if isinstance(feat, list):
            target_device = feat[-1].device
        else:
            target_device = feat.device
        ins_pos = ins_pos.to(target_device)
        target_idx = len(self.blocks) - 2 if len(self.blocks) > 1 else len(self.blocks) - 1

        if not isinstance(feat, list):
            current_global_feat = global_feat
            pre_output_global_feat = current_global_feat
            for idx, block in enumerate(self.blocks):
                current_feat = feat
                if return_pre_output_global_feat and idx == target_idx and idx != len(self.blocks) - 1:
                    (
                        ins_pos,
                        current_feat,
                        current_global_feat,
                        spatial_shape,
                        pos_dict,
                        _,
                        pre_output_global_feat,
                    ) = block(
                        ins_pos,
                        current_feat,
                        global_feat=current_global_feat,
                        spatial_shape=spatial_shape,
                        pos_dict=pos_dict,
                        return_pre_skip_global_feat=True,
                    )
                else:
                    if return_pre_output_global_feat and idx == target_idx:
                        pre_output_global_feat = current_global_feat
                    ins_pos, current_feat, current_global_feat, spatial_shape, pos_dict = block(
                        ins_pos, current_feat, global_feat=current_global_feat,
                        spatial_shape=spatial_shape, pos_dict=pos_dict
                    )
                decoded_feats.append(current_feat)
                coords.append(ins_pos)
            if return_pre_output_global_feat:
                return coords, decoded_feats, pre_output_global_feat
            return coords, decoded_feats
        else:
            current_feat = feat[-1]
            current_global_feat = global_feat[-1]
            pre_output_global_feat = current_global_feat
            for idx, block in enumerate(self.blocks):
                if return_pre_output_global_feat and idx == target_idx and idx != len(self.blocks) - 1:
                    (
                        ins_pos,
                        current_feat,
                        current_global_feat,
                        spatial_shape,
                        pos_dict,
                        _,
                        pre_output_global_feat,
                    ) = block(
                        ins_pos,
                        current_feat,
                        global_feat=current_global_feat,
                        spatial_shape=spatial_shape,
                        pos_dict=pos_dict,
                        feat_history=feat,
                        global_feat_history=global_feat,
                        return_pre_skip_global_feat=True,
                    )
                else:
                    if return_pre_output_global_feat and idx == target_idx:
                        pre_output_global_feat = current_global_feat
                    ins_pos, current_feat, current_global_feat, spatial_shape, pos_dict = block(
                        ins_pos, current_feat, global_feat=current_global_feat,
                        spatial_shape=spatial_shape, pos_dict=pos_dict,
                        feat_history=feat, global_feat_history=global_feat
                    )
                decoded_feats.append(current_feat)
                coords.append(ins_pos)

        if return_pre_output_global_feat:
            return coords, decoded_feats, pre_output_global_feat
        return coords, decoded_feats
