import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from ..functional import create_activation

class SentinelState(list):

    def __init__(self, sentinel_mask):
        super().__init__()
        self.sentinel_mask = sentinel_mask.to(dtype=torch.bool)

def _zero_sentinels(feat, sentinel_mask):
    return feat.masked_fill(sentinel_mask.to(feat.device)[:, None], 0)

def _sentinel_state(pos_dict, feat):
    if pos_dict is None:
        pos_dict = SentinelState(torch.zeros(feat.shape[0], dtype=torch.bool, device=feat.device))
    pos_dict.sentinel_mask = pos_dict.sentinel_mask.to(feat.device)
    return pos_dict

def _canonical_norm_position(norm_position):
    value = str(norm_position or 'post').strip().lower()
    if value not in {'pre', 'post'}:
        raise ValueError(f"Unsupported norm_position: {norm_position}. Use 'pre' or 'post'.")
    return value

def _pre_norm(dim, norm_position):
    return nn.LayerNorm(dim) if _canonical_norm_position(norm_position) == 'pre' else nn.Identity()

def _post_norm(dim, norm_position):
    return nn.LayerNorm(dim) if _canonical_norm_position(norm_position) == 'post' else nn.Identity()

def _pack_sparse_mask(coord, spatial_shape):
    return {"coord": coord, "spatial_shape": [int(spatial_shape[0]), int(spatial_shape[1])]}

def _unpack_sparse_mask(entry):
    if isinstance(entry, dict) and "coord" in entry:
        return entry["coord"], entry["spatial_shape"]
    if torch.is_tensor(entry):
        return torch.nonzero(entry), [*entry.shape[:2]]
    raise TypeError(f"Unsupported mask type: {type(entry)}")

def _append_sparse_mask(pos_dict, coord, spatial_shape):
    entry = _pack_sparse_mask(coord, spatial_shape)
    if pos_dict is None:
        return [entry]
    if isinstance(pos_dict, SentinelState):
        entry['sentinel_mask'] = pos_dict.sentinel_mask.clone()
    pos_dict.append(entry)
    return pos_dict

def _global_convolution(conv, global_feat, pool):
    from ..ops.global_ops import global_convolution
    return global_convolution(global_feat, conv, pool)

def _init_pixel_shuffle_weight(weight: torch.Tensor, upscale_factor: int) -> None:
    if upscale_factor <= 1:
        nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
        return

    out_features, in_features = weight.shape
    scale = upscale_factor ** 2
    if out_features % scale != 0:
        nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
        return
    with torch.no_grad():
        sub_weight = weight.new_empty(out_features // scale, in_features)
        nn.init.kaiming_normal_(sub_weight, mode='fan_out', nonlinearity='relu')
        weight.copy_(sub_weight.repeat_interleave(scale, dim=0))

class PatchConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups, bias,
                 edge_mode='none', edge_eps=1e-6, norm_position='post', backend='auto'):
        super().__init__()
        self.backend = backend
        self.input_norm = _pre_norm(in_channels, norm_position)
        self.output_norm = _post_norm(out_channels, norm_position)
        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                dilation=dilation, padding=0, bias=bias, groups=groups)
        self.edge_mode, self.edge_eps = edge_mode, edge_eps

    def forward(self, coord, feat, spatial_shape, pos_dict=None):
        from ..ops.convolution import sparse_conv2d
        pos_dict = _sentinel_state(pos_dict, feat)
        feat = _zero_sentinels(feat, pos_dict.sentinel_mask)
        feat = _zero_sentinels(self.input_norm(feat), pos_dict.sentinel_mask)
        pos_dict = _append_sparse_mask(pos_dict, coord, spatial_shape)
        out_coord, out_feat, out_shape, sentinel, plan = sparse_conv2d(
            coord, feat, spatial_shape, self.dwconv, self.edge_mode, self.edge_eps,
            pos_dict.sentinel_mask, backend=self.backend,
        )
        pos_dict[-1]['plan'] = plan
        pos_dict.sentinel_mask = sentinel
        out_feat = _zero_sentinels(self.output_norm(out_feat), sentinel)
        return out_coord, out_feat, out_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        out = _global_convolution(self.dwconv, self.input_norm(global_feat), pool)
        return self.output_norm(out)


class PatchMerge(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups, bias,
                 edge_mode='none', edge_eps=1e-6, norm_position='post', backend='auto'):
        super().__init__()
        self.backend = backend
        self.input_norm = _pre_norm(in_channels * 4, norm_position)
        self.output_norm = _post_norm(out_channels, norm_position)
        self.pwconv = nn.Linear(in_channels * 4, out_channels, bias=bias)
        self.edge_mode, self.edge_eps = edge_mode, edge_eps

    def forward(self, coord, feat, spatial_shape, pos_dict=None):
        from ..ops.resampling import sparse_patch_merge
        pos_dict = _sentinel_state(pos_dict, feat)
        pos_dict = _append_sparse_mask(pos_dict, coord, spatial_shape)
        out_coord, out_feat, out_shape, sentinel, plan = sparse_patch_merge(
            coord, feat, spatial_shape, self.pwconv, self.input_norm,
            self.edge_mode, self.edge_eps, pos_dict.sentinel_mask, backend=self.backend,
        )
        pos_dict[-1]['plan'] = plan
        pos_dict.sentinel_mask = sentinel
        out_feat = _zero_sentinels(self.output_norm(out_feat), sentinel)
        return out_coord, out_feat, out_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        bsz, dim = global_feat.shape
        merged = global_feat.unsqueeze(-1).expand(bsz, dim, 4).reshape(bsz, 4 * dim)
        return self.output_norm(self.pwconv(self.input_norm(merged)))


class PatchInverseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups, bias,
                 backend='auto'):
        super().__init__()
        self.backend = backend
        self.norm = nn.LayerNorm(out_channels)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=stride, dilation=dilation, bias=bias, groups=groups)

    def forward(self, coord, feat, spatial_shape, pos_dict):
        from ..ops.convolution import sparse_conv_transpose2d
        if pos_dict is None or not len(pos_dict):
            raise ValueError('pos_dict is required for sparse transpose convolution')
        pos_dict = _sentinel_state(pos_dict, feat)
        feat = _zero_sentinels(feat, pos_dict.sentinel_mask)
        entry = pos_dict.pop()
        out_coord, out_shape = _unpack_sparse_mask(entry)
        out_feat = sparse_conv_transpose2d(
            coord, feat, spatial_shape, out_coord, self.upconv,
            plan=entry.get('plan'), backend=self.backend,
        )
        pos_dict.sentinel_mask = entry['sentinel_mask'].to(feat.device)
        out_feat = _zero_sentinels(self.norm(out_feat), pos_dict.sentinel_mask)
        return out_coord, out_feat, out_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        from ..ops.global_ops import global_transpose
        out = global_transpose(global_feat, self.upconv, pool)
        return self.norm(out)


class PixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias, norm_position='post', backend='auto'):
        super().__init__()
        self.backend = backend
        self.upscale_factor, self.out_channels = stride, out_channels
        self.input_norm = _pre_norm(in_channels, norm_position)
        self.output_norm = _post_norm(out_channels, norm_position)
        self.conv = nn.Linear(in_channels, out_channels * stride ** 2, bias=bias)
        _init_pixel_shuffle_weight(self.conv.weight, stride)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, coord, feat, spatial_shape, pos_dict=None):
        from ..ops.resampling import pixel_shuffle
        pos_dict = _sentinel_state(pos_dict, feat)
        feat = _zero_sentinels(feat, pos_dict.sentinel_mask)
        feat = _zero_sentinels(self.input_norm(feat), pos_dict.sentinel_mask)
        entry = pos_dict.pop() if len(pos_dict) else None
        target, shape = _unpack_sparse_mask(entry) if entry is not None else (None, None)
        out_coord, out_feat, out_shape, sentinel = pixel_shuffle(
            coord, feat, spatial_shape, self.conv, self.upscale_factor, self.out_channels,
            target_coord=target, sentinel_mask=pos_dict.sentinel_mask,
            target_sentinel_mask=entry['sentinel_mask'] if entry is not None else None,
            target_shape=shape, backend=self.backend,
        )
        pos_dict.sentinel_mask = sentinel
        out_feat = _zero_sentinels(self.output_norm(out_feat), sentinel)
        return out_coord, out_feat, out_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        x = self.conv(self.input_norm(global_feat))
        x = x.view(*x.shape[:-1], self.upscale_factor ** 2, self.out_channels).mean(-2)
        return self.output_norm(x)


class ConvNextLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias, ff_ratio,
                 drop_path, activation, layer_scale):
        super(ConvNextLayer, self).__init__()
        self.kernel_size = kernel_size
        stride = 1

        self.norm = nn.LayerNorm(in_channels, eps=1e-6)

        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                stride=stride, dilation=dilation,
                                padding=(kernel_size//2)*dilation,
                                bias=bias, groups=in_channels)
        self.pwconv1 = nn.Conv2d(in_channels, int(in_channels * ff_ratio), kernel_size=1, bias=bias)
        self.act = create_activation(activation)
        self.pwconv2 = nn.Conv2d(int(in_channels * ff_ratio), in_channels, kernel_size=1, bias=bias)

        self.gamma = nn.Parameter(layer_scale * torch.ones(in_channels, 1, 1), requires_grad=True) if layer_scale > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, coord, feat, spatial_shape, pos_dict=None):
        from ..ops.convolution import sparse_submanifold_conv2d
        pos_dict = _sentinel_state(pos_dict, feat)
        shortcut = _zero_sentinels(feat, pos_dict.sentinel_mask)
        normalized = _zero_sentinels(self.norm(shortcut), pos_dict.sentinel_mask)
        out = sparse_submanifold_conv2d(coord, normalized, spatial_shape, self.dwconv,
                                        backend=getattr(self, 'backend', 'auto'))
        out = F.linear(out, self.pwconv1.weight[:, :, 0, 0], self.pwconv1.bias)
        out = self.act(out)
        out = F.linear(out, self.pwconv2.weight[:, :, 0, 0], self.pwconv2.bias)
        if self.gamma is not None:
            out = out * self.gamma[:, 0, 0]
        order = torch.argsort(coord[:, 0].long() * int(spatial_shape[1]) + coord[:, 1])
        out = shortcut[order] + self.drop_path(out[order])
        pos_dict.sentinel_mask = pos_dict.sentinel_mask[order]
        return coord[order], _zero_sentinels(out, pos_dict.sentinel_mask), spatial_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        shortcut = global_feat
        global_feat = _global_convolution(self.dwconv, global_feat, pool)
        global_feat = _global_convolution(self.pwconv1, global_feat, pool)
        global_feat = self.act(global_feat)
        global_feat = _global_convolution(self.pwconv2, global_feat, pool)
        if self.gamma is not None:
            gamma = self.gamma.squeeze()
            global_feat = gamma * global_feat
        global_feat = shortcut + self.drop_path(global_feat)
        return global_feat

class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias, activation, dropout, mlp_ratio, prenorm, drop_first):
        super(LinearLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if activation is not None:
            self.linear = nn.Sequential(nn.Linear(in_channels, int(in_channels*mlp_ratio), bias=bias),
                                        create_activation(activation),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_channels*mlp_ratio), out_channels, bias=bias),

                                        )
        else:
            self.linear = nn.Sequential(nn.Linear(in_channels, out_channels, bias),
                                        nn.Dropout(dropout))
            if drop_first:
                self.linear = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(in_channels, out_channels, bias))

        self.prenorm = nn.LayerNorm(in_channels) if prenorm else nn.Identity()

    def forward(self, coord, feat, spatial_shape=None, pos_dict=None):
        pos_dict = _sentinel_state(pos_dict, feat)
        feat = _zero_sentinels(feat, pos_dict.sentinel_mask)
        feat = _zero_sentinels(self.prenorm(feat), pos_dict.sentinel_mask)
        feat = self.linear(feat)
        feat = _zero_sentinels(feat, pos_dict.sentinel_mask)

        coord = coord.to(feat.device)
        spatial_shape = (coord.int().max(dim=0)[0]+1).tolist() if len(coord) else [0, 0]
        return coord, feat, spatial_shape, pos_dict

    def conv_forward(self, global_feat, pool=None):
        global_feat = self.prenorm(global_feat)
        global_feat = self.linear(global_feat)

        return global_feat

class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(ProjectionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projection = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, coord, feat, spatial_shape=None, pos_dict=None):
        pos_dict = _sentinel_state(pos_dict, feat)
        feat = _zero_sentinels(feat, pos_dict.sentinel_mask)
        feat = self.projection(feat)
        feat = _zero_sentinels(feat, pos_dict.sentinel_mask)
        coord = coord.to(feat.device)
        if spatial_shape is None:
            spatial_shape = (coord.int().max(dim=0)[0]+1).tolist() if len(coord) else [0, 0]
        return coord, feat, spatial_shape, pos_dict

    def conv_forward(self, global_feat, pool=None):
        global_feat = self.projection(global_feat)
        return global_feat

class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, *inputs):
        return inputs

    def conv_forward(self, global_feat, pool=None):
        return global_feat

class ConvolutionLayer(nn.Module):
    def __init__(self, layer_type: str, in_channels=None, out_channels=None, **kwargs):
        super().__init__()
        self.pool = kwargs.get('pool', 'mean')
        self.layer_type = layer_type
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.global_strategy = kwargs.get('global_strategy', 'proj')
        out_channels = self.out_channels
        self.norm_position = _canonical_norm_position(kwargs.get('norm_position', 'post'))
        self.global_input_norm = _pre_norm(in_channels, self.norm_position)
        backend = kwargs.get('sparse_backend', 'auto')
        if backend not in ('auto', 'torch', 'triton'):
            raise ValueError('sparse_backend must be auto/torch/triton')
        kind = layer_type
        spatial = dict(kernel_size=kwargs.get('kernel_size', 2), stride=kwargs.get('stride', 2),
                       dilation=kwargs.get('dilation', 1), groups=kwargs.get('groups', 1),
                       bias=kwargs.get('bias', False), backend=backend)
        if kind in ('patchconv', 'dwpatchconv', 'patchinvs', 'dwpatchinvs'):
            groups = in_channels if kind.startswith('dw') else spatial['groups']
            if groups <= 0 or in_channels % groups or out_channels % groups:
                raise ValueError(
                    f"{kind}: in_channels={in_channels} and out_channels={out_channels} "
                    f"must both be divisible by groups={groups}. "
                    "Depthwise layers use groups=in_channels; channel reduction requires "
                    "a separate projection or a non-depthwise layer. For decoder concat "
                    "skips, use patchinvs/pixelshuffle or adjust the skip and channel settings."
                )
        norm = dict(norm_position=self.norm_position)
        edge = dict(edge_mode=kwargs.get('edge_mode', 'none'), edge_eps=kwargs.get('edge_eps', 1e-6))
        if kind in ('patchconv', 'dwpatchconv'):
            if kind.startswith('dw'):
                spatial['groups'] = in_channels
            self.conv_block = PatchConvBlock(in_channels, out_channels, **spatial, **norm, **edge)
        elif kind in ('patchinvs', 'dwpatchinvs'):
            if kind.startswith('dw'):
                spatial['groups'] = in_channels
            self.conv_block = PatchInverseBlock(in_channels, out_channels, **spatial)
        elif kind == 'patchmerge':
            self.conv_block = PatchMerge(in_channels, out_channels, **spatial, **norm, **edge)
        elif kind == 'pixelshuffle':
            self.conv_block = PixelShuffle(in_channels, out_channels, spatial['stride'], spatial['bias'],
                                           backend=backend, **norm)
        elif kind == 'linear':
            self.conv_block = LinearLayer(in_channels, out_channels, kwargs.get('bias', False),
                kwargs.get('activation', 'ReLU'), kwargs.get('dropout', 0.), kwargs.get('mlp_ratio', .5),
                kwargs.get('prenorm', False), kwargs.get('drop_first', False))
        elif kind == 'convnext':
            self.conv_block = ConvNextLayer(in_channels, spatial['kernel_size'], spatial['dilation'],
                spatial['bias'], kwargs.get('ff_ratio', 2.), kwargs.get('drop_path', 0.),
                kwargs.get('activation', 'GELU'), kwargs.get('layer_scale', 1e-6))
            self.conv_block.backend = backend
        elif kind == 'projection':
            self.conv_block = ProjectionLayer(in_channels, out_channels, kwargs.get('bias', False))
        elif kind == 'identity':
            self.conv_block = IdentityLayer()
        else:
            raise ValueError(f'Unsupported block type: {layer_type}')
        self._init_global_modules()

    def _init_global_modules(self):
        if self.global_strategy == 'proj':
            self.g_proj = nn.Linear(self.in_channels, self.out_channels, bias=True)
        elif self.global_strategy == 'identity':
            self.identity_proj = (nn.Linear(self.in_channels, self.out_channels)
                                  if self.in_channels != self.out_channels else nn.Identity())

    def forward(self, coord, feat, spatial_shape=None, pos_dict=None):
        if spatial_shape is None:
            spatial_shape = ((coord.int().max(0).values + 1).tolist() if len(coord) else [0, 0])
        pos_dict = _sentinel_state(pos_dict, feat)
        coord = coord.to(feat.device)
        feat = _zero_sentinels(feat, pos_dict.sentinel_mask)
        out_coord, out_feat, spatial_shape, pos_dict = self.conv_block(coord, feat, spatial_shape, pos_dict)
        return out_coord, _zero_sentinels(out_feat, pos_dict.sentinel_mask), spatial_shape, pos_dict

    def conv_forward(self, global_feat):
        if self.global_strategy == 'kernel':
            return self.conv_block.conv_forward(global_feat, self.pool)
        if self.global_strategy == 'identity':
            return self.identity_proj(self.global_input_norm(global_feat))
        if self.global_strategy == 'proj':
            return self.g_proj(self.global_input_norm(global_feat))
        return global_feat
