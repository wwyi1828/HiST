import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from ..functional import create_activation

def _canonical_norm_position(norm_position):
    value = str(norm_position or 'post').strip().lower()
    if value not in {'pre', 'post'}:
        raise ValueError(f"Unsupported norm_position: {norm_position}. Use 'pre' or 'post'.")
    return value

def _pre_norm(dim, norm_position):
    return nn.LayerNorm(dim) if _canonical_norm_position(norm_position) == 'pre' else nn.Identity()

def _post_norm(dim, norm_position):
    return nn.LayerNorm(dim) if _canonical_norm_position(norm_position) == 'post' else nn.Identity()

def _build_dense(coord, feat, spatial_shape):
    dense = feat.new_zeros(spatial_shape + [feat.shape[-1]])
    dense[coord[:, 0], coord[:, 1]] = feat
    return dense

def _build_mask(coord, spatial_shape, dtype=torch.float32):
    mask = torch.zeros(spatial_shape, dtype=dtype, device=coord.device)
    mask[coord[:, 0], coord[:, 1]] = 1
    return mask

def _gather_by_mask(dense_feat, mask):
    out_coord = torch.nonzero(mask)
    out_feat = dense_feat[out_coord[:, 0], out_coord[:, 1]]
    spatial_shape = [*mask.shape[:2]]
    return out_coord, out_feat, spatial_shape

def dense_ConvwithMM(conv, global_feat, pool):
    kernel_size = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
    dilation = conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation
    padding = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding

    expanded_height = kernel_size + (kernel_size - 1) * (dilation - 1)
    expanded_width = kernel_size + (kernel_size - 1) * (dilation - 1)

    global_feat_expanded = global_feat.unsqueeze(-1).unsqueeze(-1)
    global_feat_expanded = global_feat_expanded.expand(-1, -1, expanded_height, expanded_width)

    conv_out = conv(global_feat_expanded)
    if pool == 'sum':
        global_feat = conv_out.sum([-1, -2])
    elif pool == 'mean':
        global_feat = conv_out.mean([-1, -2])
    elif pool == 'max':
        global_feat = conv_out.amax(dim=[-1, -2])
    return global_feat

def _to_pair(val):
    if isinstance(val, (list, tuple)):
        return int(val[0]), int(val[1])
    return int(val), int(val)

def _conv_out_size(length, kernel, stride, dilation):
    return (length - dilation * (kernel - 1) - 1) // stride + 1

def _kernel_offsets(k_h, k_w, d_h, d_w, device):
    ys = torch.arange(k_h, device=device) * d_h
    xs = torch.arange(k_w, device=device) * d_w
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)

def _build_index_map(coord, spatial_shape):
    coord = coord.long()
    index_map = torch.zeros(spatial_shape, dtype=torch.int32, device=coord.device)
    if coord.numel() == 0:
        return index_map
    flat = torch.arange(coord.shape[0], device=coord.device, dtype=torch.int32) + 1
    index_map[coord[:, 0], coord[:, 1]] = flat
    return index_map

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
    pos_dict.append(entry)
    return pos_dict

def _apply_conv_weight(neigh, conv):
    weight = conv.weight
    bias = conv.bias
    n_out, k_len, c_in = neigh.shape
    groups = conv.groups
    out_channels = conv.out_channels

    if groups == 1:
        w = weight.view(out_channels, -1)
        neigh_flat = neigh.permute(0, 2, 1).reshape(n_out, -1)
        out = neigh_flat @ w.t()
    else:
        c_g = c_in // groups
        o_g = out_channels // groups
        neigh_g = neigh.view(n_out, k_len, groups, c_g).permute(0, 2, 1, 3)
        k_h, k_w = weight.shape[2], weight.shape[3]
        w_g = weight.view(groups, o_g, c_g, k_h * k_w).permute(0, 1, 3, 2)
        out = torch.einsum('ngkc,gokc->ngo', neigh_g, w_g)
        out = out.reshape(n_out, out_channels)

    if bias is not None:
        out = out + bias
    return out

def _apply_conv_transpose_weight(neigh, conv):
    weight = conv.weight
    bias = conv.bias
    n_out, k_len, c_in = neigh.shape
    groups = conv.groups
    out_channels = conv.out_channels
    k_h, k_w = weight.shape[2], weight.shape[3]

    if groups == 1:
        w = weight.view(c_in, out_channels, k_h * k_w).permute(0, 2, 1)
        out = torch.einsum('nkc,cko->no', neigh, w)
    else:
        c_g = c_in // groups
        o_g = out_channels // groups
        neigh_g = neigh.view(n_out, k_len, groups, c_g).permute(0, 2, 1, 3)
        w_g = weight.view(groups, c_g, o_g, k_h * k_w)
        out = torch.einsum('ngkc,gcok->ngo', neigh_g, w_g)
        out = out.reshape(n_out, out_channels)

    if bias is not None:
        out = out + bias
    return out

def _sparse_conv2d(coord, feat, spatial_shape, conv, edge_mode='none', edge_eps=1e-6):
    coord = coord.long()
    h_in, w_in = int(spatial_shape[0]), int(spatial_shape[1])
    if coord.numel() == 0:
        out_shape = [_conv_out_size(h_in, _to_pair(conv.kernel_size)[0], _to_pair(conv.stride)[0], _to_pair(conv.dilation)[0]),
                     _conv_out_size(w_in, _to_pair(conv.kernel_size)[1], _to_pair(conv.stride)[1], _to_pair(conv.dilation)[1])]
        return coord.new_zeros((0, 2)), feat.new_zeros((0, conv.out_channels)), out_shape

    k_h, k_w = _to_pair(conv.kernel_size)
    s_h, s_w = _to_pair(conv.stride)
    d_h, d_w = _to_pair(conv.dilation)
    h_out = _conv_out_size(h_in, k_h, s_h, d_h)
    w_out = _conv_out_size(w_in, k_w, s_w, d_w)

    if h_out <= 0 or w_out <= 0:
        return coord.new_zeros((0, 2)), feat.new_zeros((0, conv.out_channels)), [h_out, w_out]

    offsets = _kernel_offsets(k_h, k_w, d_h, d_w, coord.device)
    delta = coord[:, None, :] - offsets[None, :, :]
    valid = (delta[..., 0] >= 0) & (delta[..., 1] >= 0)
    if s_h != 1:
        valid &= (delta[..., 0] % s_h == 0)
    if s_w != 1:
        valid &= (delta[..., 1] % s_w == 0)
    out_y = delta[..., 0] // s_h
    out_x = delta[..., 1] // s_w
    valid &= (out_y < h_out) & (out_x < w_out)
    flat = (out_y * w_out + out_x)[valid]

    if flat.numel() == 0:
        return coord.new_zeros((0, 2)), feat.new_zeros((0, conv.out_channels)), [h_out, w_out]

    flat = torch.unique(flat, sorted=True)
    out_coord = torch.stack([flat // w_out, flat % w_out], dim=1)

    index_map = _build_index_map(coord, spatial_shape)
    in_y = out_coord[:, 0, None] * s_h + offsets[:, 0][None, :]
    in_x = out_coord[:, 1, None] * s_w + offsets[:, 1][None, :]
    idx = index_map[in_y, in_x]

    feat_pad = torch.cat([feat.new_zeros(1, feat.shape[1]), feat], dim=0)
    neigh = feat_pad[idx.long()]
    out_feat = _apply_conv_weight(neigh, conv)

    if edge_mode and edge_mode != 'none':
        mask = (idx > 0).to(out_feat.dtype)
        if edge_mode == 'count':
            denom = mask.sum(dim=1, keepdim=True).clamp(min=edge_eps)
            out_feat = out_feat / denom
        elif edge_mode == 'partial':
            w_abs = conv.weight.abs().sum(dim=1).reshape(conv.out_channels, -1)
            denom = mask @ w_abs.t()
            denom = denom.clamp(min=edge_eps)
            out_feat = out_feat / denom

    return out_coord, out_feat, [h_out, w_out]

def _sparse_conv_transpose2d(coord, feat, spatial_shape, out_coord, conv):
    coord = coord.long()
    out_coord = out_coord.long()
    h_in, w_in = int(spatial_shape[0]), int(spatial_shape[1])
    if coord.numel() == 0 or out_coord.numel() == 0:
        return feat.new_zeros((out_coord.shape[0], conv.out_channels))

    k_h, k_w = _to_pair(conv.kernel_size)
    s_h, s_w = _to_pair(conv.stride)
    d_h, d_w = _to_pair(conv.dilation)
    offsets = _kernel_offsets(k_h, k_w, d_h, d_w, out_coord.device)

    delta = out_coord[:, None, :] - offsets[None, :, :]
    valid = (delta[..., 0] >= 0) & (delta[..., 1] >= 0)
    if s_h != 1:
        valid &= (delta[..., 0] % s_h == 0)
    if s_w != 1:
        valid &= (delta[..., 1] % s_w == 0)
    in_y = delta[..., 0] // s_h
    in_x = delta[..., 1] // s_w
    valid &= (in_y < h_in) & (in_x < w_in)

    in_y_safe = torch.where(valid, in_y, torch.zeros_like(in_y))
    in_x_safe = torch.where(valid, in_x, torch.zeros_like(in_x))

    index_map = _build_index_map(coord, spatial_shape)
    idx = index_map[in_y_safe, in_x_safe]
    idx = torch.where(valid, idx, torch.zeros_like(idx))

    feat_pad = torch.cat([feat.new_zeros(1, feat.shape[1]), feat], dim=0)
    neigh = feat_pad[idx.long()]
    out_feat = _apply_conv_transpose_weight(neigh, conv)

    return out_feat

def _pixelshuffle_expand(coord, feat, upscale_factor, out_channels, linear):
    coord = coord.long()
    if coord.numel() == 0:
        return coord.new_zeros((0, 2)), feat.new_zeros((0, out_channels))
    expanded = linear(feat)
    expanded = expanded.view(coord.shape[0], out_channels, upscale_factor, upscale_factor)
    expanded = expanded.permute(0, 2, 3, 1).reshape(coord.shape[0], upscale_factor * upscale_factor, out_channels)
    offsets = _kernel_offsets(upscale_factor, upscale_factor, 1, 1, coord.device)
    out_coord = coord[:, None, :] * upscale_factor + offsets[None, :, :]
    out_coord = out_coord.reshape(-1, 2)
    out_feat = expanded.reshape(-1, out_channels)
    return out_coord, out_feat


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

class s_PatchConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups, bias,
                 edge_mode='none', edge_eps=1e-6, norm_position='post'):
        super(s_PatchConvBlock, self).__init__()
        self.input_norm = _pre_norm(in_channels, norm_position)
        self.output_norm = _post_norm(out_channels, norm_position)
        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, dilation=dilation, padding=0,
                                bias=bias, groups=groups)
        self.edge_mode = edge_mode
        self.edge_eps = edge_eps

    def forward(self, coord, feat, spatial_shape, pos_dict=None):
        coord = coord.long()
        feat = self.input_norm(feat)
        pos_dict = _append_sparse_mask(pos_dict, coord, spatial_shape)
        out_coord, out_feat, out_shape = _sparse_conv2d(
            coord, feat, spatial_shape, self.dwconv, self.edge_mode, self.edge_eps
        )
        out_feat = self.output_norm(out_feat)
        return out_coord, out_feat, out_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        global_feat = self.input_norm(global_feat)
        out = dense_ConvwithMM(self.dwconv, global_feat, pool)
        out = self.output_norm(out)
        return out

class s_PatchMerge(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups, bias,
                 edge_mode='none', edge_eps=1e-6, norm_position='post'):
        super(s_PatchMerge, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.input_norm = _pre_norm(in_channels * 4, norm_position)
        self.output_norm = _post_norm(out_channels, norm_position)
        self.pwconv = nn.Linear(in_channels * 4, out_channels, bias=bias)
        self.edge_mode = edge_mode
        self.edge_eps = edge_eps

    def forward(self, coord, feat, spatial_shape, pos_dict=None):
        coord = coord.long()
        pos_dict = _append_sparse_mask(pos_dict, coord, spatial_shape)
        h_out = int(spatial_shape[0]) // 2
        w_out = int(spatial_shape[1]) // 2

        if coord.numel() == 0:
            out_coord = coord.new_zeros((0, 2))
            out_feat = feat.new_zeros((0, self.pwconv.out_features))
            return out_coord, out_feat, [h_out, w_out], pos_dict

        out_coord_base = coord // 2
        valid = (out_coord_base[:, 0] >= 0) & (out_coord_base[:, 1] >= 0)
        valid &= (out_coord_base[:, 0] < h_out) & (out_coord_base[:, 1] < w_out)
        out_coord_base = out_coord_base[valid]
        feat = feat[valid]
        offset = (coord[valid, 0] % 2) * 2 + (coord[valid, 1] % 2)

        flat = out_coord_base[:, 0] * w_out + out_coord_base[:, 1]
        unique_flat, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        out_coord = torch.stack([unique_flat // w_out, unique_flat % w_out], dim=1)

        c_in = feat.shape[1]
        out_feat_full = feat.new_zeros((out_coord.shape[0], 4 * c_in))
        cols = offset[:, None] * c_in + torch.arange(c_in, device=feat.device)[None, :]
        out_feat_full[inverse[:, None], cols] = feat

        if self.edge_mode and self.edge_mode != 'none':
            counts = feat.new_zeros(out_coord.shape[0])
            counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=counts.dtype))
            counts = counts.clamp(min=1.0)
            scale = (1.0 / counts.clamp(min=self.edge_eps)).unsqueeze(1)
            out_feat_full = out_feat_full * scale

        out_feat_full = self.input_norm(out_feat_full)
        out_feat = self.pwconv(out_feat_full)
        out_feat = self.output_norm(out_feat)
        return out_coord, out_feat, [h_out, w_out], pos_dict

    def conv_forward(self, global_feat, pool):
        bsz, dim = global_feat.shape
        merged = global_feat.unsqueeze(-1).expand(bsz, dim, 4).reshape(bsz, 4 * dim)
        merged = self.input_norm(merged)
        out = self.pwconv(merged)
        out = self.output_norm(out)
        return out

class s_PatchInverseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups, bias):
        super(s_PatchInverseBlock, self).__init__()
        self.norm = nn.LayerNorm(out_channels)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=stride, dilation=dilation, bias=bias, groups=groups)
        self.pwconv = nn.Identity()

    def forward(self, coord, feat, spatial_shape, pos_dict):
        if pos_dict is None or len(pos_dict) == 0:
            raise ValueError("pos_dict is required for s_PatchInverseBlock")
        entry = pos_dict.pop()
        out_coord, out_shape = _unpack_sparse_mask(entry)
        out_feat = _sparse_conv_transpose2d(coord, feat, spatial_shape, out_coord, self.upconv)
        out_feat = self.pwconv(out_feat)
        out_feat = self.norm(out_feat)
        return out_coord, out_feat, out_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        global_feat_expanded = global_feat.unsqueeze(-1).unsqueeze(-1)
        global_feat_out = self.upconv(global_feat_expanded)
        if pool == 'mean':
            global_feat_out = torch.nn.functional.adaptive_avg_pool2d(global_feat_out, (1, 1))
        elif pool == 'max':
            global_feat_out = torch.nn.functional.adaptive_max_pool2d(global_feat_out, (1, 1))

        global_feat_out = global_feat_out.squeeze(-1).squeeze(-1)
        global_feat_out = self.pwconv(global_feat_out)
        global_feat_out = self.norm(global_feat_out)
        return global_feat_out

class s_PixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias, norm_position='post'):
        super(s_PixelShuffle, self).__init__()
        self.upscale_factor = stride
        self.out_channels = out_channels
        self.input_norm = _pre_norm(in_channels, norm_position)
        self.output_norm = _post_norm(out_channels, norm_position)
        self.conv = nn.Linear(in_channels, out_channels * self.upscale_factor ** 2, bias=bias)
        _init_pixel_shuffle_weight(self.conv.weight, self.upscale_factor)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, coord, feat, spatial_shape, pos_dict):
        feat = self.input_norm(feat)
        if pos_dict is None:
            out_coord, out_feat = _pixelshuffle_expand(coord, feat, self.upscale_factor, self.out_channels, self.conv)
            spatial_shape = [int(spatial_shape[0]) * self.upscale_factor, int(spatial_shape[1]) * self.upscale_factor]
            out_feat = self.output_norm(out_feat)
            return out_coord, out_feat, spatial_shape, pos_dict

        entry = pos_dict.pop()
        target_coord, target_shape = _unpack_sparse_mask(entry)
        all_coord, all_feat = _pixelshuffle_expand(coord, feat, self.upscale_factor, self.out_channels, self.conv)
        index_map = _build_index_map(all_coord, target_shape)
        idx = index_map[target_coord[:, 0], target_coord[:, 1]]
        valid = idx > 0
        out_coord = target_coord[valid]
        out_feat = all_feat[(idx[valid] - 1).long()]
        out_feat = self.output_norm(out_feat)
        return out_coord, out_feat, target_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        global_feat = self.input_norm(global_feat)
        x = self.conv(global_feat)
        x = x.view(*x.shape[:-1], self.upscale_factor ** 2, self.conv.out_features // (self.upscale_factor ** 2))
        global_feat_out = torch.mean(x, dim=-2)
        global_feat_out = self.output_norm(global_feat_out)
        return global_feat_out

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
        shortcut = feat
        feat = self.norm(feat)
        dense_feat = _build_dense(coord, feat, spatial_shape)
        mask = _build_mask(coord, spatial_shape)
        dense_out_feat = self.dwconv(dense_feat.unsqueeze(0).permute(0, 3, 1, 2))
        dense_out_feat = self.pwconv1(dense_out_feat)
        dense_out_feat = self.act(dense_out_feat)
        dense_out_feat = self.pwconv2(dense_out_feat)

        if self.gamma is not None:
            dense_out_feat = self.gamma * dense_out_feat

        dense_out_feat = dense_out_feat.squeeze(0).permute(1, 2, 0)
        out_coord, out_feat, _ = _gather_by_mask(dense_out_feat, mask)

        out_feat = self.drop_path(out_feat)
        out_feat = shortcut + out_feat

        return out_coord, out_feat, spatial_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        shortcut = global_feat
        global_feat = dense_ConvwithMM(self.dwconv, global_feat, pool)
        global_feat = dense_ConvwithMM(self.pwconv1, global_feat, pool)
        global_feat = self.act(global_feat)
        global_feat = dense_ConvwithMM(self.pwconv2, global_feat, pool)
        if self.gamma is not None:
            gamma = self.gamma.squeeze()
            global_feat = gamma * global_feat
        global_feat = shortcut + self.drop_path(global_feat)
        return global_feat

class PatchMerge(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups, bias,
                 norm_position='post'):
        super(PatchMerge, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        groups = groups

        self.input_norm = _pre_norm(in_channels * 4, norm_position)
        self.output_norm = _post_norm(out_channels, norm_position)
        self.pwconv = nn.Linear(in_channels*4, out_channels, bias=bias)

    def _patch_merging(self, x):
        B, H, W, C = x.shape
        if self.dilation == 2:
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 0::2, 1::2, :]
            x2 = x[:, 1::2, 0::2, :]
            x3 = x[:, 1::2, 1::2, :]
            merged = torch.cat([x0, x1, x2, x3], -1)
        else:
            x = x.view(B, H//2, 2, W//2, 2, C)
            x = x.permute(0, 1, 3, 2, 4, 5)
            merged = x.reshape(B, H//2, W//2, 4*C)
        return merged

    def forward(self, coord, feat, spatial_shape, pos_dict=None):
        dense_feat = _build_dense(coord, feat, spatial_shape)
        mask = _build_mask(coord, spatial_shape)

        if pos_dict is None:
            pos_dict = [mask]
        else:
            pos_dict.append(mask)

        merged_feat = self._patch_merging(dense_feat.unsqueeze(0))
        merged_feat = self.input_norm(merged_feat)
        dense_out_feat = self.pwconv(merged_feat).squeeze(0)

        merged_mask = self._patch_merging(mask.unsqueeze(0).unsqueeze(-1))
        out_mask = (merged_mask.sum(dim=-1) > 0).float().squeeze()

        out_coord, out_feat, _ = _gather_by_mask(dense_out_feat, out_mask)
        out_feat = self.output_norm(out_feat)

        spatial_shape = [*out_mask.shape[:2]]

        return out_coord, out_feat, spatial_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        B, D = global_feat.shape
        merged = global_feat.unsqueeze(-1).expand(B, D, 4).reshape(B, 4 * D)
        merged = self.input_norm(merged)
        out = self.pwconv(merged)
        out = self.output_norm(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups, bias,
                 norm_position='post'):
        super(BasicBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        groups = groups
        self.input_norm = _pre_norm(in_channels, norm_position)
        self.output_norm = _post_norm(out_channels, norm_position)

        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation, padding=0,
                              bias=bias, groups=groups)

    def forward(self, coord, feat, spatial_shape, pos_dict=None):
        feat = self.input_norm(feat)
        dense_feat = _build_dense(coord, feat, spatial_shape)
        mask = _build_mask(coord, spatial_shape)

        if pos_dict is None:
            pos_dict = [mask]
        else:
            pos_dict.append(mask)

        dense_out_feat = self.dwconv(dense_feat.unsqueeze(0).permute(0, 3, 1, 2))
        dense_out_feat = dense_out_feat.squeeze(0).permute(1, 2, 0)

        with torch.no_grad():

            unfold = nn.Unfold(
                kernel_size=(self.kernel_size, self.kernel_size),
                dilation=self.dilation,
                stride=self.stride, padding=0,
            )
            unfolded_mask = unfold(mask.unsqueeze(0).unsqueeze(0))

            out_mask = (unfolded_mask.sum(dim=1) > 0).float()
            out_mask = out_mask.view(dense_out_feat.shape[0], dense_out_feat.shape[1])

        out_coord, out_feat, _ = _gather_by_mask(dense_out_feat, out_mask)
        out_feat = self.output_norm(out_feat)

        spatial_shape = [*out_mask.shape[:2]]

        return out_coord, out_feat, spatial_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        global_feat = self.input_norm(global_feat)
        out = dense_ConvwithMM(self.dwconv, global_feat, pool)
        out = self.output_norm(out)
        return out

class InverseBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups, bias):
        super(InverseBasicBlock, self).__init__()
        self.kernel_size = kernel_size
        groups = groups

        self.norm = nn.LayerNorm(out_channels)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation, bias=bias, groups=groups)

        self.pwconv = nn.Identity()

    def forward(self, coord, feat, spatial_shape, pos_dict):
        dense_feat = _build_dense(coord, feat, spatial_shape)

        dense_out_feat = self.upconv(dense_feat.unsqueeze(0).permute(0, 3, 1, 2)).squeeze(0).permute(1, 2, 0)
        out_mask = pos_dict.pop()
        if isinstance(out_mask, dict) and "coord" in out_mask:
            out_mask = _build_mask(out_mask["coord"], out_mask["spatial_shape"])

        out_coord, out_feat, _ = _gather_by_mask(dense_out_feat, out_mask.squeeze())

        spatial_shape = [*out_mask.shape[:2]]
        out_feat = self.pwconv(out_feat)
        out_feat = self.norm(out_feat)

        return out_coord, out_feat, spatial_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        global_feat_expanded = global_feat.unsqueeze(-1).unsqueeze(-1)

        global_feat_out = self.upconv(global_feat_expanded)
        if pool == 'mean':
            global_feat_out = torch.nn.functional.adaptive_avg_pool2d(global_feat_out, (1, 1))
        elif pool == 'max':
            global_feat_out = torch.nn.functional.adaptive_max_pool2d(global_feat_out, (1, 1))

        global_feat_out = global_feat_out.squeeze(-1).squeeze(-1)
        global_feat_out = self.pwconv(global_feat_out)
        global_feat_out = self.norm(global_feat_out)

        return global_feat_out

class PixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias, norm_position='post'):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = stride

        self.input_norm = _pre_norm(in_channels, norm_position)
        self.output_norm = _post_norm(out_channels, norm_position)

        self.conv = nn.Linear(in_channels, out_channels * self.upscale_factor**2, bias=bias)
        _init_pixel_shuffle_weight(self.conv.weight, self.upscale_factor)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, coord, feat, spatial_shape, pos_dict):
        feat = self.input_norm(feat)
        dense_feat = _build_dense(coord, feat, spatial_shape)
        x = self.conv(dense_feat)
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        x = nn.functional.pixel_shuffle(x, self.upscale_factor)
        dense_out_feat = x.squeeze(0).permute(1, 2, 0)

        if pos_dict is None:
            mask_matrix = _build_mask(coord, spatial_shape)

            out_mask = F.interpolate(
                mask_matrix.unsqueeze(0).unsqueeze(0),
                scale_factor=self.upscale_factor,
                mode='nearest'
            ).squeeze(0).squeeze(0)

        else:
            out_mask = pos_dict.pop()
            if isinstance(out_mask, dict) and "coord" in out_mask:
                out_mask = _build_mask(out_mask["coord"], out_mask["spatial_shape"])

        out_coord, out_feat, _ = _gather_by_mask(dense_out_feat, out_mask.squeeze())
        out_feat = self.output_norm(out_feat)
        spatial_shape = [*out_mask.shape[:2]]
        return out_coord, out_feat, spatial_shape, pos_dict

    def conv_forward(self, global_feat, pool):
        global_feat = self.input_norm(global_feat)
        x = self.conv(global_feat)
        x = x.view(*x.shape[:-1], self.upscale_factor**2, self.conv.out_features//(self.upscale_factor**2))
        global_feat_out = torch.mean(x, dim=-2)
        global_feat_out = self.output_norm(global_feat_out)
        return global_feat_out

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
        feat = self.prenorm(feat)
        feat = self.linear(feat)

        coord = coord.to(feat.device)
        spatial_shape = (coord.int().max(dim=0)[0]+1).tolist()
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
        feat = self.projection(feat)
        coord = coord.to(feat.device)
        if spatial_shape is None:
            spatial_shape = (coord.int().max(dim=0)[0]+1).tolist()
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
    def __init__(self, layer_type: str, in_channels=None, out_channels=None, mask_padding=False, **kwargs):
        super().__init__()
        self.pool = kwargs.get('pool', 'mean')
        self.layer_type = layer_type
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.global_strategy = kwargs.get('global_strategy', 'proj')

        default_args = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': 2, 'stride': 2, 'dilation': 1, 'groups': 1,
                        'bias': False, 'norm_position': 'post'
        }
        extra_defaults = {
            'linear': { 'activation': 'ReLU', 'dropout': 0.0, 'mlp_ratio': 0.5, 'prenorm': False, 'drop_first': False},
            'convnext': {'activation': 'GELU', 'ff_ratio': 2.0, 'drop_path': 0.0,'layer_scale': 1e-6}
        }
        if layer_type in extra_defaults:
            default_args.update(extra_defaults[layer_type])
        default_args.update(kwargs)
        self.norm_position = _canonical_norm_position(default_args.get('norm_position', 'post'))
        self.global_input_norm = _pre_norm(in_channels, self.norm_position)

        blocks = {
            'patchconv': lambda: s_PatchConvBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'], stride=default_args['stride'],
                                                dilation=default_args['dilation'], groups=default_args['groups'], bias=default_args['bias'],
                                                edge_mode=default_args.get('edge_mode', 'none'),
                                                edge_eps=default_args.get('edge_eps', 1e-6),
                                                norm_position=self.norm_position),
            'dwpatchconv': lambda: s_PatchConvBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'], stride=default_args['stride'],
                                                dilation=default_args['dilation'], groups=in_channels, bias=default_args['bias'],
                                                edge_mode=default_args.get('edge_mode', 'none'),
                                                edge_eps=default_args.get('edge_eps', 1e-6),
                                                norm_position=self.norm_position),
            'd_patchconv': lambda: BasicBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'], stride=default_args['stride'],
                                                dilation=default_args['dilation'], groups=default_args['groups'], bias=default_args['bias'],
                                                norm_position=self.norm_position),
            'd_dwpatchconv': lambda: BasicBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'], stride=default_args['stride'],
                                                dilation=default_args['dilation'], groups=in_channels, bias=default_args['bias'],
                                                norm_position=self.norm_position),
            's_patchconv': lambda: s_PatchConvBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'], stride=default_args['stride'],
                                                dilation=default_args['dilation'], groups=default_args['groups'], bias=default_args['bias'],
                                                edge_mode=default_args.get('edge_mode', 'none'),
                                                edge_eps=default_args.get('edge_eps', 1e-6),
                                                norm_position=self.norm_position),
            's_dwpatchconv': lambda: s_PatchConvBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'], stride=default_args['stride'],
                                                dilation=default_args['dilation'], groups=in_channels, bias=default_args['bias'],
                                                edge_mode=default_args.get('edge_mode', 'none'),
                                                edge_eps=default_args.get('edge_eps', 1e-6),
                                                norm_position=self.norm_position),
            'patchinvs': lambda: s_PatchInverseBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'],
                                                          stride=default_args['stride'], dilation=default_args['dilation'], groups=default_args['groups'],
                                                          bias=default_args['bias']),
            'dwpatchinvs': lambda: s_PatchInverseBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'],
                                                          stride=default_args['stride'], dilation=default_args['dilation'], groups=in_channels,
                                                          bias=default_args['bias']),
            'd_patchinvs': lambda: InverseBasicBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'],
                                                          stride=default_args['stride'], dilation=default_args['dilation'], groups=default_args['groups'],
                                                          bias=default_args['bias']),
            'd_dwpatchinvs': lambda: InverseBasicBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'],
                                                          stride=default_args['stride'], dilation=default_args['dilation'], groups=in_channels,
                                                          bias=default_args['bias']),
            's_patchinvs': lambda: s_PatchInverseBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'],
                                                          stride=default_args['stride'], dilation=default_args['dilation'], groups=default_args['groups'],
                                                          bias=default_args['bias']),
            's_dwpatchinvs': lambda: s_PatchInverseBlock(in_channels, out_channels, kernel_size=default_args['kernel_size'],
                                                          stride=default_args['stride'], dilation=default_args['dilation'], groups=in_channels,
                                                          bias=default_args['bias']),
            'pixelshuffle': lambda: s_PixelShuffle(in_channels, out_channels, stride=default_args['stride'], bias=default_args['bias'],
                                                   norm_position=self.norm_position),
            'd_pixelshuffle': lambda: PixelShuffle(in_channels, out_channels, stride=default_args['stride'], bias=default_args['bias'],
                                                   norm_position=self.norm_position),
            's_pixelshuffle': lambda: s_PixelShuffle(in_channels, out_channels, stride=default_args['stride'], bias=default_args['bias'],
                                                     norm_position=self.norm_position),
            'linear': lambda: LinearLayer(in_channels, out_channels, bias=default_args['bias'], activation=default_args['activation'],
                                          dropout=default_args['dropout'],
                                          mlp_ratio=default_args['mlp_ratio'], prenorm=default_args['prenorm'],
                                          drop_first=default_args['drop_first']),
            'convnext': lambda: ConvNextLayer(in_channels, kernel_size=default_args['kernel_size'], dilation=default_args['dilation'],
                                              bias=default_args['bias'], ff_ratio=default_args['ff_ratio'],
                                              drop_path=default_args['drop_path'], activation=default_args['activation'],
                                              layer_scale=default_args['layer_scale']),
            'patchmerge': lambda: s_PatchMerge(in_channels, out_channels, kernel_size=default_args['kernel_size'],
                                             stride=default_args['stride'], dilation=default_args['dilation'],
                                             groups=default_args['groups'], bias=default_args['bias'],
                                             edge_mode=default_args.get('edge_mode', 'none'),
                                             edge_eps=default_args.get('edge_eps', 1e-6),
                                             norm_position=self.norm_position),
            'd_patchmerge': lambda: PatchMerge(in_channels, out_channels, kernel_size=default_args['kernel_size'],
                                             stride=default_args['stride'], dilation=default_args['dilation'],
                                             groups=default_args['groups'], bias=default_args['bias'],
                                             norm_position=self.norm_position),
            's_patchmerge': lambda: s_PatchMerge(in_channels, out_channels, kernel_size=default_args['kernel_size'],
                                             stride=default_args['stride'], dilation=default_args['dilation'],
                                             groups=default_args['groups'], bias=default_args['bias'],
                                             edge_mode=default_args.get('edge_mode', 'none'),
                                             edge_eps=default_args.get('edge_eps', 1e-6),
                                             norm_position=self.norm_position),
            'projection': lambda: ProjectionLayer(in_channels, out_channels, bias=default_args['bias']),
            'identity': lambda: IdentityLayer()
        }
        if layer_type not in blocks:
            raise ValueError(f"Unsupported block type: {layer_type}. Available types: {list(blocks.keys())}")
        self.conv_block = blocks[layer_type]()
        self.mask_padding = mask_padding
        self._init_global_modules()

    def _init_global_modules(self):
        if self.global_strategy == 'proj':
            C_in = self.in_channels
            C_out = self.out_channels
            self.g_proj = nn.Linear(C_in, C_out, bias=True)
        if self.global_strategy == 'identity':
            self.identity_proj = nn.Linear(self.in_channels, self.out_channels) if self.in_channels != self.out_channels else nn.Identity()

    def forward(self, coord, feat, spatial_shape=None, pos_dict=None):
        if spatial_shape is None:
            spatial_shape = (coord.int().max(dim=0)[0] + 1).tolist()
        out_coord, out_feat, spatial_shape, pos_dict = self.conv_block(coord, feat, spatial_shape, pos_dict)
        if self.mask_padding:
            out_feat[[0,-1]] = 0
        return out_coord, out_feat, spatial_shape, pos_dict

    def conv_forward(self, global_feat):
        if self.global_strategy == 'kernel':
            return self.conv_block.conv_forward(global_feat, self.pool)
        if self.global_strategy == 'identity':
            return self.identity_proj(self.global_input_norm(global_feat))
        if self.global_strategy == 'proj':
            return self.g_proj(self.global_input_norm(global_feat))
        return global_feat
