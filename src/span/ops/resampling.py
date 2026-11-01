from dataclasses import dataclass

import torch
import torch.nn.functional as F
import importlib.util

from .spatial import downsample_plan


def _backend(backend, tensor):
    if backend not in ('auto', 'torch', 'triton'):
        raise ValueError("backend must be auto, torch or triton")
    if backend == 'triton' and not tensor.is_cuda:
        raise ValueError('Triton resampling requires CUDA tensors')
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        if backend == 'triton':
            raise ValueError('Triton resampling supports FP32, FP16 and BF16')
        return 'torch'
    if tensor.is_cuda and backend != 'torch':
        if importlib.util.find_spec('triton') is not None:
            return 'triton'
        if backend == 'triton':
            raise RuntimeError('Triton backend requested but Triton is not installed')
    return 'torch'


def _linear(feat, linear):
    # Turing has no native BF16 GEMM. Cast through FP32 while retaining the
    # original parameter/activation dtypes and autograd cast edges.
    dtype = torch.get_autocast_gpu_dtype() if feat.is_cuda and torch.is_autocast_enabled() else feat.dtype
    if dtype == torch.bfloat16 and (not feat.is_cuda or torch.cuda.get_device_capability(feat.device)[0] < 8):
        bias = linear.bias.to(dtype).float() if linear.bias is not None else None
        with torch.autocast(device_type=feat.device.type, enabled=False):
            return F.linear(feat.to(dtype).float(), linear.weight.to(dtype).float(), bias).to(dtype)
    return linear(feat)


def sparse_patch_merge(coord, feat, spatial_shape, linear, input_norm,
                       edge_mode='none', edge_eps=1e-6, sentinel_mask=None, backend='auto'):
    backend = _backend(backend, feat)
    if feat.ndim != 2 or linear.in_features != 4 * feat.shape[1]:
        raise ValueError('patch merge requires [N,C] input and Linear(4*C, out_channels)')
    if edge_mode not in (None, 'none', 'count', 'partial') or edge_eps <= 0:
        raise ValueError('edge_mode must be none/count/partial and edge_eps must be positive')
    if sentinel_mask is None:
        sentinel_mask = torch.zeros(len(coord), dtype=torch.bool, device=feat.device)
    sentinel_mask = sentinel_mask.to(device=feat.device, dtype=torch.bool).contiguous()
    # Keep sentinel indices in the topology so missing and sentinel slots remain
    # distinguishable after LayerNorm. No large feature tensor is scattered.
    plan = downsample_plan(coord, spatial_shape, backend=backend)
    safe = plan.children.clamp(min=0).long()
    slot_sentinel = sentinel_mask[safe] & (plan.children >= 0) if len(feat) else torch.zeros_like(plan.children, dtype=torch.bool)
    real = (plan.children >= 0) & ~slot_sentinel
    out_sentinel = ~real.any(1)
    norm = isinstance(input_norm, torch.nn.LayerNorm)
    if norm and tuple(input_norm.normalized_shape) != (feat.shape[1] * 4,):
        raise ValueError('input LayerNorm must normalize 4*C channels')
    if not norm and not isinstance(input_norm, torch.nn.Identity):
        raise ValueError('input_norm must be LayerNorm or Identity')
    if backend == 'triton':
        from ._resampling_kernels import merge_gather
        gamma = input_norm.weight if norm else None
        beta = input_norm.bias if norm else None
        if norm and (gamma is None or beta is None):
            gamma = feat.new_ones(4 * feat.shape[1]) if gamma is None else gamma
            beta = feat.new_zeros(4 * feat.shape[1]) if beta is None else beta
        soft_bf16 = feat.dtype == torch.bfloat16 and torch.cuda.get_device_capability(feat.device)[0] < 8
        kernel_feat = feat.float() if soft_bf16 else feat
        if soft_bf16:
            gamma = gamma.float() if gamma is not None else None
            beta = beta.float() if beta is not None else None
        merged = merge_gather(kernel_feat.contiguous(), plan.children, sentinel_mask,
                                   gamma, beta, float(input_norm.eps) if norm else 0.,
                                   bool(edge_mode and edge_mode != 'none'), float(edge_eps), soft_bf16)
        if soft_bf16:
            merged = merged.to(feat.dtype)
    else:
        if len(feat):
            merged = feat[safe].masked_fill(~real[:, :, None], 0).reshape(len(plan.children), -1)
        else:
            merged = feat.new_zeros((len(plan.children), 4 * feat.shape[1])) + feat.sum() * 0
        if edge_mode and edge_mode != 'none':
            count = real.to(feat.dtype).sum(1).clamp(min=1.).clamp(min=edge_eps)
            merged = merged * count.reciprocal()[:, None]
        merged = input_norm(merged)
        merged = merged.masked_fill(slot_sentinel.repeat_interleave(feat.shape[1], 1), 0)
    out = _linear(merged, linear)
    out = out.masked_fill(out_sentinel[:, None], 0)
    return plan.out_coord, out, list(plan.out_shape), out_sentinel, plan


@dataclass(frozen=True)
class _PixelPlan:
    coord: torch.Tensor
    parent: torch.Tensor
    slot: torch.Tensor
    output_mask: torch.Tensor
    sentinel_mask: torch.Tensor
    order: torch.Tensor
    starts: torch.Tensor
    counts: torch.Tensor
    reverse: torch.Tensor
    shape: list


def _pixel_plan(coord, feat, spatial_shape, factor, target_coord, sentinel,
                target_sentinel, target_shape, backend):
    n, s = len(coord), factor * factor
    device = feat.device
    if target_coord is None:
        shape = [int(spatial_shape[0]) * factor, int(spatial_shape[1]) * factor]
        if backend == 'triton':
            from ._resampling_kernels import expand_pixel_coordinates
            out_coord, parent, slot, out_sentinel = expand_pixel_coordinates(coord, sentinel, factor)
        else:
            parent = torch.arange(n, device=device).repeat_interleave(s)
            slot = torch.arange(s, device=device).repeat(n)
            offset = torch.stack((slot // factor, slot % factor), 1)
            out_coord = coord[parent] * factor + offset
            out_sentinel = sentinel[parent]
        output_mask = out_sentinel
    else:
        out_coord = target_coord.long().to(device)
        shape = list(target_shape) if target_shape is not None else [int(spatial_shape[0]) * factor, int(spatial_shape[1]) * factor]
        target_parent = out_coord // factor
        slot = (out_coord[:, 0] % factor * factor + out_coord[:, 1] % factor).long()
        if n:
            width = max(int(spatial_shape[1]), 1)
            keys, order = torch.sort(coord[:, 0].long() * width + coord[:, 1].long())
            target_keys = target_parent[:, 0] * width + target_parent[:, 1]
            rank = torch.searchsorted(keys, target_keys).clamp(max=n - 1)
            parent = order[rank]
            present = (coord[parent] == target_parent).all(1)
            parent = torch.where(present, parent, -1)
        else:
            parent = torch.full((len(out_coord),), -1, dtype=torch.int64, device=device)
        if target_sentinel is None:
            target_sentinel = torch.zeros(len(out_coord), dtype=torch.bool, device=device)
        else:
            target_sentinel = target_sentinel.to(device=device, dtype=torch.bool)
        keep = parent >= 0
        out_coord, parent, slot, target_sentinel = (t[keep] for t in (out_coord, parent, slot, target_sentinel))
        out_sentinel = target_sentinel.contiguous()
        parent_sentinel = sentinel[parent.clamp(min=0)] & (parent >= 0) if n else torch.zeros_like(parent, dtype=torch.bool)
        output_mask = out_sentinel | parent_sentinel
    m = len(parent)
    if target_coord is None:
        # Every projected child is an output: one vendor GEMM and a fused
        # layout/mask kernel are preferable to grouping rows by child slot.
        order = torch.empty(0, dtype=torch.int64, device=device)
        counts = starts = torch.empty(0, dtype=torch.int32, device=device)
        reverse = torch.empty((0, 0), dtype=torch.int32, device=device)
    else:
        order = torch.argsort(slot).to(torch.int64)
        counts = torch.bincount(slot, minlength=s).to(torch.int32)
        starts = (counts.cumsum(0) - counts).to(torch.int32)
        reverse = torch.full((n, s), -1, dtype=torch.int32, device=device)
    if m and target_coord is not None:
        if backend == 'triton':
            from ._resampling_kernels import fill_pixel_reverse
            fill_pixel_reverse(parent, slot, reverse, s)
        else:
            valid = parent >= 0
            reverse[parent[valid], slot[valid]] = torch.arange(m, device=device)[valid].to(torch.int32)
    return _PixelPlan(out_coord, parent.contiguous(), slot.contiguous(), output_mask.contiguous(),
                      out_sentinel, order, starts, counts, reverse, shape)


def pixel_shuffle(coord, feat, spatial_shape, linear, factor, out_channels,
                  target_coord=None, sentinel_mask=None, target_sentinel_mask=None,
                  target_shape=None, backend='auto'):
    backend = _backend(backend, feat)
    factor, out_channels = int(factor), int(out_channels)
    if factor <= 0 or out_channels <= 0 or linear.in_features != feat.shape[1] or linear.out_features != factor ** 2 * out_channels:
        raise ValueError('pixel shuffle requires positive factor/channels and a matching Linear')
    if sentinel_mask is None:
        sentinel_mask = torch.zeros(len(coord), dtype=torch.bool, device=feat.device)
    sentinel_mask = sentinel_mask.to(device=feat.device, dtype=torch.bool).contiguous()
    plan = _pixel_plan(coord, feat, spatial_shape, factor, target_coord, sentinel_mask,
                       target_sentinel_mask, target_shape, backend)
    s, o = factor ** 2, out_channels
    if backend == 'triton' and target_coord is None:
        from ._resampling_kernels import shuffle_layout
        out = shuffle_layout(_linear(feat, linear), plan.parent, plan.slot,
                             plan.output_mask, o, s)
    elif backend == 'triton':
        from ._resampling_kernels import pixel_linear
        compute = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else feat.dtype
        soft_bf16 = compute == torch.bfloat16 and torch.cuda.get_device_capability(feat.device)[0] < 8
        kernel_feat = feat.to(compute)
        kernel_weight = linear.weight.to(compute)
        kernel_bias = linear.bias.to(compute) if linear.bias is not None else None
        if soft_bf16:
            kernel_feat, kernel_weight = kernel_feat.float(), kernel_weight.float()
            kernel_bias = kernel_bias.float() if kernel_bias is not None else None
        out = pixel_linear(kernel_feat.contiguous(), kernel_weight.contiguous(),
                                 kernel_bias.contiguous() if kernel_bias is not None else None,
                                 plan.parent, plan.order, plan.starts, plan.counts,
                                 plan.reverse, plan.output_mask, s, o,
                                 torch.float32 if soft_bf16 else compute)
        if soft_bf16:
            out = out.to(compute)
    else:
        if len(feat):
            gathered = feat[plan.parent.clamp(min=0)].masked_fill((plan.parent < 0)[:, None], 0)
        else:
            gathered = feat.new_zeros((len(plan.coord), feat.shape[1])) + feat.sum() * 0
        weights = linear.weight.view(o, s, feat.shape[1]).permute(1, 0, 2)[plan.slot]
        # FP32 accumulation supplies the software BF16 path on pre-Ampere GPUs.
        dtype = torch.get_autocast_gpu_dtype() if feat.is_cuda and torch.is_autocast_enabled() else feat.dtype
        x = gathered.to(dtype)
        weights = weights.to(dtype)
        with torch.autocast(device_type=feat.device.type, enabled=False):
            if dtype in (torch.float16, torch.bfloat16):
                out = torch.einsum('mc,moc->mo', x.float(), weights.float())
            else:
                out = torch.einsum('mc,moc->mo', x, weights)
        if linear.bias is not None:
            out = out + linear.bias.view(o, s).t()[plan.slot].to(dtype).to(out.dtype)
        out = out.to(dtype)
        out = out.masked_fill(plan.output_mask[:, None], 0)
    return plan.coord, out, plan.shape, plan.sentinel_mask
