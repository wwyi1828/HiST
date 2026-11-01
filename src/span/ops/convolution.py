from dataclasses import dataclass

import importlib.util

import torch


_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None


def _pair(value):
    return tuple(map(int, value)) if isinstance(value, (tuple, list)) else (int(value), int(value))


@dataclass(frozen=True)
class ConvPlan:
    input_coord: torch.Tensor
    out_coord: torch.Tensor
    input_shape: tuple
    out_shape: tuple
    indices: torch.Tensor  # [outputs, kernel cells], -1 for absent/non-real inputs
    reverse: torch.Tensor  # [inputs, kernel cells], -1 for no output at this offset
    sentinel_mask: torch.Tensor
    kernel: tuple
    stride: tuple
    dilation: tuple
    padding: tuple
    transposed: bool = False


def _lookup(coord, spatial_shape, queries, valid):
    if not len(coord):
        return torch.full(queries.shape[:-1], -1, dtype=torch.int32, device=coord.device)
    width = max(int(spatial_shape[1]), 1)
    keys, order = torch.sort(coord[:, 0].long() * width + coord[:, 1].long())
    query_keys = queries[..., 0] * width + queries[..., 1]
    locations = torch.searchsorted(keys, query_keys.contiguous()).clamp(max=len(keys) - 1)
    found = valid & (keys[locations] == query_keys)
    return torch.where(found, order[locations], -1).to(torch.int32)


def _same_tensor(a, b):
    return a.shape == b.shape and a.device == b.device and a.data_ptr() == b.data_ptr()


def convolution_plan(coord, spatial_shape, conv, target_coord=None, sentinel_mask=None,
                     transposed=False, inverse_of=None):
    coord = coord.long().contiguous()
    shape = tuple(map(int, spatial_shape))
    if isinstance(conv.padding, str):
        raise ValueError('Sparse convolution requires explicit integer padding')
    kernel, stride = _pair(conv.kernel_size), _pair(conv.stride)
    dilation, padding = _pair(conv.dilation), _pair(conv.padding)
    if sentinel_mask is not None:
        sentinel_mask = sentinel_mask.to(device=coord.device, dtype=torch.bool)
    if transposed and isinstance(inverse_of, ConvPlan) and not inverse_of.transposed and (
        (kernel, stride, dilation, padding) ==
        (inverse_of.kernel, inverse_of.stride, inverse_of.dilation, inverse_of.padding)
        and _same_tensor(coord, inverse_of.out_coord)
        and target_coord is not None and _same_tensor(target_coord, inverse_of.input_coord)
        and sentinel_mask is None
    ):
        indices, reverse = inverse_of.reverse, inverse_of.indices
        return ConvPlan(coord, target_coord, shape, inverse_of.input_shape, indices, reverse,
                        (indices < 0).all(1), kernel, stride, dilation, padding, True)
    device = coord.device
    if (not transposed and target_coord is None and kernel == (2, 2)
            and stride == (2, 2) and dilation == (1, 1) and padding == (0, 0)):
        from .spatial import downsample_plan
        down = downsample_plan(coord, shape, sentinel_mask,
                               backend='triton' if coord.is_cuda and _TRITON_AVAILABLE else 'torch')
        reverse = torch.empty((len(coord), 4), dtype=torch.int32, device=device)
        if len(coord):
            if coord.is_cuda and _TRITON_AVAILABLE:
                from ._convolution_kernels import expand_reverse
                expand_reverse(down.input_parent, down.input_slot, reverse)
            else:
                slots = torch.arange(4, device=device)[None, :]
                reverse.copy_(torch.where(slots == down.input_slot[:, None],
                                          down.input_parent[:, None], -1))
        return ConvPlan(coord, down.out_coord, shape, down.out_shape, down.children,
                        reverse, down.sentinel_mask, kernel, stride, dilation, padding)
    if transposed:
        output_padding = _pair(getattr(conv, 'output_padding', 0))
        native_out_shape = tuple((shape[d] - 1) * stride[d] - 2 * padding[d]
                                 + dilation[d] * (kernel[d] - 1) + output_padding[d] + 1 for d in (0, 1))
    else:
        native_out_shape = tuple((shape[d] + 2 * padding[d] - dilation[d] * (kernel[d] - 1) - 1)
                                 // stride[d] + 1 for d in (0, 1))
    if (coord.is_cuda and _TRITON_AVAILABLE and 0 < shape[0] * shape[1] <= 1048576
            and (target_coord is not None or 0 < native_out_shape[0] * native_out_shape[1] <= 1048576)
            and min(native_out_shape) > 0):
        from ._convolution_kernels import _native_plan
        return _native_plan(coord, shape, target_coord, sentinel_mask, kernel, stride,
                            dilation, padding, native_out_shape, transposed)
    ys = torch.arange(kernel[0], device=device) * dilation[0]
    xs = torch.arange(kernel[1], device=device) * dilation[1]
    y, x = torch.meshgrid(ys, xs, indexing='ij')
    offsets = torch.stack((y.flatten(), x.flatten()), 1)
    s = coord.new_tensor(stride)
    pad = coord.new_tensor(padding)
    if transposed:
        output_padding = _pair(getattr(conv, 'output_padding', 0))
        out_shape = tuple((shape[d] - 1) * stride[d] - 2 * padding[d]
                          + dilation[d] * (kernel[d] - 1) + output_padding[d] + 1 for d in (0, 1))
        if target_coord is None:
            raise ValueError('Transpose sparse convolution requires target_coord')
        out_coord = target_coord.long().contiguous()
        delta = out_coord[:, None, :] + pad - offsets[None, :, :]
        valid = (delta >= 0).all(-1) & (delta % s == 0).all(-1)
        queries = torch.div(delta, s, rounding_mode='floor')
    else:
        out_shape = tuple((shape[d] + 2 * padding[d] - dilation[d] * (kernel[d] - 1) - 1)
                          // stride[d] + 1 for d in (0, 1))
        if target_coord is None:
            if min(out_shape) <= 0 or not len(coord):
                out_coord = coord.new_empty((0, 2))
            else:
                delta = coord[:, None, :] + pad - offsets[None, :, :]
                valid = (delta >= 0).all(-1) & (delta % s == 0).all(-1)
                candidates = torch.div(delta, s, rounding_mode='floor')
                valid &= (candidates[..., 0] < out_shape[0]) & (candidates[..., 1] < out_shape[1])
                flat = candidates[..., 0] * out_shape[1] + candidates[..., 1]
                flat = torch.unique(flat[valid], sorted=True)
                out_coord = torch.stack((flat // out_shape[1], flat % out_shape[1]), 1)
        else:
            out_coord = target_coord.long().contiguous()
        queries = out_coord[:, None, :] * s - pad + offsets[None, :, :]
        valid = (queries >= 0).all(-1)
    valid &= (queries[..., 0] < shape[0]) & (queries[..., 1] < shape[1])
    indices = _lookup(coord, shape, queries, valid)
    if sentinel_mask is not None and len(coord):
        real = (indices >= 0) & ~sentinel_mask[indices.long().clamp(min=0)]
        indices = torch.where(real, indices, -1)
    reverse = torch.full((len(coord), len(offsets)), -1, dtype=torch.int32, device=device)
    rows = torch.arange(len(out_coord), device=device)[:, None].expand_as(indices)
    slots = torch.arange(len(offsets), device=device)[None, :].expand_as(indices)
    present = indices >= 0
    reverse[indices[present].long(), slots[present]] = rows[present].to(torch.int32)
    return ConvPlan(coord, out_coord, shape, out_shape, indices.contiguous(), reverse,
                    ~present.any(1), kernel, stride, dilation, padding, transposed)


def _torch_indexed_matmul(feat, weight, plan, groups, out_channels):
    icg, ocg = feat.shape[1] // groups, out_channels // groups
    out = feat.new_zeros((len(plan.out_coord), out_channels))
    # Empty tensors remain connected to trainable inputs with zero gradients.
    out = out + feat.sum() * 0 + weight.sum() * 0
    if not len(feat) or not len(plan.out_coord):
        return out
    k = plan.indices.shape[1]
    for slot in range(k):
        index = plan.indices[:, slot].long()
        x = feat[index.clamp(min=0)] * (index >= 0)[:, None].to(feat.dtype)
        for group in range(groups):
            if plan.transposed:
                w = weight.reshape(feat.shape[1], ocg, k)[group * icg:(group + 1) * icg, :, slot]
            else:
                w = weight.reshape(out_channels, icg, k)[group * ocg:(group + 1) * ocg, :, slot].t()
            out[:, group * ocg:(group + 1) * ocg] = (
                out[:, group * ocg:(group + 1) * ocg] + x[:, group * icg:(group + 1) * icg] @ w)
    return out


def indexed_matmul(feat, weight, plan, groups, out_channels, backend='auto'):
    if backend not in ('auto', 'triton', 'torch'):
        raise ValueError('backend must be auto/triton/torch')
    if backend == 'triton' and (not feat.is_cuda or not _TRITON_AVAILABLE):
        raise ValueError('Triton backend requires CUDA and Triton')
    if feat.ndim != 2 or len(feat) != len(plan.input_coord):
        raise ValueError('feat must have shape [len(plan.input_coord), C]')
    if feat.device != weight.device or feat.device != plan.indices.device:
        raise ValueError('features, weights and plan must share a device')
    if not feat.is_floating_point() or not weight.is_floating_point():
        raise ValueError('features and weights must be floating point')
    compute_dtype = torch.promote_types(feat.dtype, weight.dtype)
    if torch.is_autocast_enabled() and compute_dtype != torch.float64:
        compute_dtype = torch.get_autocast_gpu_dtype()
    feat = feat.to(compute_dtype).contiguous()
    weight = weight.to(compute_dtype).contiguous()
    if backend == 'torch' or not feat.is_cuda or not _TRITON_AVAILABLE or compute_dtype == torch.float64:
        return _torch_indexed_matmul(feat, weight, plan, groups, out_channels)
    # Turing has no BF16 arithmetic. Torch casts round-trip in software, allowing
    # the same FP32 kernels to execute the BF16 operator with FP32 accumulation.
    emulate_bfloat = compute_dtype == torch.bfloat16 and torch.cuda.get_device_capability(feat.device)[0] < 8
    if emulate_bfloat:
        feat, weight = feat.float(), weight.float()
    from ._convolution_kernels import _IndexedMatmul, _VendorIndexedMatmul
    implementation = _VendorIndexedMatmul if feat.dtype == torch.float32 and groups != feat.shape[1] else _IndexedMatmul
    with torch.cuda.device(feat.device):
        out = implementation.apply(feat, weight, plan.indices, plan.reverse, groups,
                                   plan.transposed, out_channels)
    return out.to(compute_dtype) if emulate_bfloat else out


def sparse_conv2d(coord, feat, spatial_shape, conv, edge_mode='none', edge_eps=1e-6,
                  sentinel_mask=None, plan=None, backend='auto'):
    if edge_mode not in ('none', 'count', 'partial', None):
        raise ValueError('edge_mode must be none/count/partial')
    if plan is None:
        plan = convolution_plan(coord, spatial_shape, conv, sentinel_mask=sentinel_mask)
    out = indexed_matmul(feat, conv.weight, plan, conv.groups, conv.out_channels, backend)
    if conv.bias is not None:
        out = out + conv.bias
    present = plan.indices >= 0
    real = present.any(1, keepdim=True)
    if edge_mode == 'count':
        denom = present.to(out.dtype).sum(1, keepdim=True)
        denom = torch.where(real, denom.clamp(min=edge_eps), torch.ones_like(denom))
        out = out / denom
    elif edge_mode == 'partial':
        weights = conv.weight.abs().sum(1).reshape(conv.out_channels, -1)
        denom = present.to(out.dtype) @ weights.t()
        denom = torch.where(real, denom.clamp(min=edge_eps), torch.ones_like(denom))
        out = out / denom
    out = out.masked_fill(~real, 0)
    return plan.out_coord, out, list(plan.out_shape), plan.sentinel_mask, plan


def sparse_conv_transpose2d(coord, feat, spatial_shape, out_coord, conv, plan=None, backend='auto'):
    if not len(coord) or not len(out_coord):
        zero = feat.sum() * 0 + conv.weight.sum() * 0
        if conv.bias is not None:
            zero = zero + conv.bias.sum() * 0
        return feat.new_zeros((len(out_coord), conv.out_channels)) + zero
    topology = convolution_plan(coord, spatial_shape, conv, target_coord=out_coord,
                                 transposed=True, inverse_of=plan)
    out = indexed_matmul(feat, conv.weight, topology, conv.groups, conv.out_channels, backend)
    return out + conv.bias if conv.bias is not None else out


def sparse_submanifold_conv2d(coord, feat, spatial_shape, conv, sentinel_mask=None, backend='auto'):
    topology = convolution_plan(coord, spatial_shape, conv, target_coord=coord,
                                 sentinel_mask=sentinel_mask)
    out = indexed_matmul(feat, conv.weight, topology, conv.groups, conv.out_channels, backend)
    return out + conv.bias if conv.bias is not None else out
