import torch
from torch import nn
from timm.layers import DropPath
from ..functional import create_activation
from .positional_encoding import apply_rope_2d_partial, RelativePositionBias, ALiBiPositionBias

class BaseTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop, proj_drop, drop_path, activation='GELU', w_RPB=None, ff_ratio=2, pos_std=0.02, share_qkv=True, pos_emb_type='rpb', rope_theta=10000.0, rope_partial_factor=1.0):
        super(BaseTransformerLayer, self).__init__()
        input_dim = embed_dim
        embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.pos_emb_type = pos_emb_type.lower()
        self.rope_theta = rope_theta
        self.rope_partial_factor = rope_partial_factor

        if self.pos_emb_type == 'rpb':
            if w_RPB is not None:
                self.table_RPB = RelativePositionBias(num_heads, 2*w_RPB, learned_pos=True, pos_std=pos_std)
            else:
                self.table_RPB = None
        elif self.pos_emb_type == 'alibi':
            self.pos_encoder = ALiBiPositionBias(num_heads)
        elif self.pos_emb_type == 'rope':
            if self.rope_partial_factor < 1.0:
                self.rope_dim = int(self.head_dim * self.rope_partial_factor)
                self.rope_dim = (self.rope_dim // 4) * 4
            else:
                self.rope_dim = None
        elif self.pos_emb_type == 'none':
            pass
        else:
            raise ValueError(f"Unknown pos_emb_type: {pos_emb_type}. Choose from ['rpb', 'alibi', 'rope', 'none']")

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path)

        bias = True
        self.scaling = self.head_dim ** -0.5
        self.q1 = nn.Linear(input_dim, embed_dim, bias=bias)
        self.k1 = nn.Linear(input_dim, embed_dim, bias=bias)
        self.v1 = nn.Linear(input_dim, embed_dim, bias=bias)

        if share_qkv:
            self.q2 = self.q1
            self.k2 = self.k1
            self.v2 = self.v1
        else:
            self.q2 = nn.Linear(input_dim, embed_dim, bias=bias)
            self.k2 = nn.Linear(input_dim, embed_dim, bias=bias)
            self.v2 = nn.Linear(input_dim, embed_dim, bias=bias)

        self.fc_out = nn.Linear(embed_dim, embed_dim, bias=True)

        ff_dim = int(embed_dim * ff_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim, bias=True),
            create_activation(activation),
            nn.Dropout(proj_drop),
            nn.Linear(ff_dim, embed_dim, bias=True),
            nn.Dropout(proj_drop)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def flexible_attention(self, plan, feat, global_feat, ins_pos=None, alter_qkv=False, mask_type='local'):

        all_feats = torch.cat([feat, global_feat], dim=0)

        if alter_qkv:
            q = self.q2(all_feats).view(-1, self.num_heads, self.head_dim)
            k = self.k2(all_feats).view(-1, self.num_heads, self.head_dim)
            v = self.v2(all_feats).view(-1, self.num_heads, self.head_dim)
        else:
            q = self.q1(all_feats).view(-1, self.num_heads, self.head_dim)
            k = self.k1(all_feats).view(-1, self.num_heads, self.head_dim)
            v = self.v1(all_feats).view(-1, self.num_heads, self.head_dim)

        if self.pos_emb_type == 'rope' and ins_pos is not None:
            if global_feat.size(0) > 0:
                global_pos = torch.zeros(global_feat.size(0), 2, device=ins_pos.device, dtype=ins_pos.dtype)
                extended_pos = torch.cat([ins_pos, global_pos], dim=0)
            else:
                extended_pos = ins_pos

            q = apply_rope_2d_partial(q, extended_pos, rotary_dim=self.rope_dim, base=self.rope_theta)
            k = apply_rope_2d_partial(k, extended_pos, rotary_dim=self.rope_dim, base=self.rope_theta)

        from ..ops.attention import attention
        bias_table = slopes = None
        if ins_pos is not None and mask_type == 'local':
            if self.pos_emb_type == 'rpb' and self.table_RPB is not None:
                bias_table = self.table_RPB.adjustment_tables
            elif self.pos_emb_type == 'alibi':
                slopes = self.pos_encoder.slopes
        return attention(
            q, k, v, plan, bias_table=bias_table, alibi_slopes=slopes,
            dropout_p=self.attn_drop.p, training=self.training,
        ).flatten(1)

    def forward(self, *args):
        raise NotImplementedError("This method should be overridden by subclasses")

class SwinTransLayer(BaseTransformerLayer):
    def forward(self, feat, global_feat, plan1, plan2, ins_pos=None):

        shortcut = torch.cat([feat, global_feat], dim=0)
        feat = self.norm1(feat)
        global_feat = self.norm1(global_feat)
        attn_out = 0
        if plan1 is not None:
            attn_out += self.flexible_attention(plan1, feat, global_feat, ins_pos, alter_qkv=False, mask_type='local')
        if plan2 is not None:
            attn_out += self.flexible_attention(plan2, feat, global_feat, ins_pos, alter_qkv=True, mask_type='local')

        attn_out = self.fc_out(attn_out)
        attn_out = self.proj_drop(attn_out)
        feats = shortcut + self.drop_path(attn_out)

        feats = feats + self.drop_path(self.ffn(self.norm2(feats)))

        feat = feats[:feat.size(0)]
        global_feat = feats[feat.size(0):]
        return feat, global_feat

class TradSwinTransLayer(BaseTransformerLayer):
    def __init__(self, embed_dim, num_heads, attn_drop, proj_drop, drop_path, activation='GELU', w_RPB=None, ff_ratio=2, pos_std=0.02, share_qkv=None, **kwargs):
        super(TradSwinTransLayer, self).__init__(embed_dim, num_heads, attn_drop, proj_drop, drop_path, activation, w_RPB, ff_ratio, pos_std, share_qkv=False, **kwargs)
        if w_RPB is not None:
            self.table_RPB  = RelativePositionBias(num_heads, 2*w_RPB, learned_pos=True, pos_std=pos_std)
        self.fc_out_2 = nn.Linear(embed_dim, embed_dim)

        ff_dim = int(embed_dim * ff_ratio)
        input_dim = self.q1.in_features

        self.q2 = nn.Linear(input_dim, embed_dim)
        self.k2 = nn.Linear(input_dim, embed_dim)
        self.v2 = nn.Linear(input_dim, embed_dim)
        self.ffn_2 = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            create_activation(activation),
            nn.Dropout(proj_drop),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(proj_drop)
        )

        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, feat, global_feat, plan1, plan2, ins_pos=None):

        shortcut = torch.cat([feat, global_feat], dim=0)
        feat = self.norm1(feat)
        global_feat = self.norm1(global_feat)
        attn_out = self.flexible_attention(plan1, feat, global_feat, ins_pos, alter_qkv=False, mask_type='local')
        attn_out = self.fc_out(attn_out)
        attn_out = self.proj_drop(attn_out)
        feats = shortcut + self.drop_path(attn_out)
        feats = feats + self.drop_path(self.ffn(self.norm2(feats)))

        feat = feats[:feat.size(0)]
        global_feat = feats[feat.size(0):]

        shortcut = torch.cat([feat, global_feat], dim=0)
        feat = self.norm3(feat)
        global_feat = self.norm3(global_feat)
        attn_out = self.flexible_attention(plan2, feat, global_feat, ins_pos, alter_qkv=True, mask_type='local')
        attn_out = self.fc_out_2(attn_out)
        attn_out = self.proj_drop(attn_out)
        feats = shortcut + self.drop_path(attn_out)
        feats = feats + self.drop_path(self.ffn_2(self.norm4(feats)))

        feat = feats[:feat.size(0)]
        global_feat = feats[feat.size(0):]
        return feat, global_feat

class IdentityTransLayer(nn.Module):
    def __init__(self):
        super(IdentityTransLayer, self).__init__()

    def forward(self, feat, global_feat, *args, **kwargs):
        return feat, global_feat

class LinearTransLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearTransLayer, self).__init__()
        self.dim_reduc = nn.Linear(in_channels, out_channels, bias)

    def forward(self, feat, global_feat, *args, **kwargs):
        feat = self.dim_reduc(feat)
        return feat, global_feat

class LongformerTransLayer(BaseTransformerLayer):
    def forward(self, feat, global_feat, plan1, plan2, ins_pos=None):

        shortcut = torch.cat([feat, global_feat], dim=0)
        feat = self.norm1(feat)
        global_feat = self.norm1(global_feat)

        attn_out = 0
        if plan1 is not None:
            attn_out += self.flexible_attention(plan1, feat, global_feat, ins_pos, alter_qkv=False, mask_type='local')
        if global_feat.size(0) > 0:
            attn_out += self.flexible_attention(plan2, feat, global_feat, ins_pos, alter_qkv=True, mask_type='global')

        attn_out = self.fc_out(attn_out)
        attn_out = self.proj_drop(attn_out)
        feats = shortcut + self.drop_path(attn_out)

        feats = feats + self.drop_path(self.ffn(self.norm2(feats)))

        feat = feats[:feat.size(0)]
        global_feat = feats[feat.size(0):]
        return feat, global_feat

class HybridTransLayer(LongformerTransLayer):
    pass


class TransformerLayer(nn.Module):
    def __init__(self, layer_type: str, embed_dim=None, num_heads=None, **kwargs):
        super().__init__()

        default_args = {'embed_dim': embed_dim, 'num_heads': num_heads, 'attn_drop': 0.0, 'proj_drop': 0.0, 'drop_path': 0.0, 'activation': 'GELU',
                        'w_RPB': None, 'ff_ratio': 2.0, 'pos_std': 0.02, 'share_qkv': True
        }
        default_args.update(kwargs)

        blocks = {
            'hybrid': lambda: HybridTransLayer(**default_args),
            'hybrid_noshift': lambda: HybridTransLayer(**default_args),
            'swin': lambda: SwinTransLayer(**default_args),
            'swin_global': lambda: SwinTransLayer(**default_args),
            'tradswin': lambda: TradSwinTransLayer(**default_args),
            'long': lambda: LongformerTransLayer(**default_args),
            'long_noshift': lambda: LongformerTransLayer(**default_args),
            'linear': lambda: LinearTransLayer(in_channels=kwargs.get('in_channels', embed_dim), out_channels=kwargs.get('out_channels', embed_dim),
                                                bias=kwargs.get('bias', True)
            ),
            'identity': lambda: IdentityTransLayer()
        }

        if layer_type not in blocks:
            raise ValueError(f"Unsupported block type: {layer_type}. Available types: {list(blocks.keys())}")

        self.trans_block = blocks[layer_type]()

    def forward(self, feat, global_feat, plan1=None, plan2=None, ins_pos=None):
        return self.trans_block(feat, global_feat, plan1, plan2, ins_pos)
