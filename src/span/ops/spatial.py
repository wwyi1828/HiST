from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DownsamplePlan:
    out_coord: torch.Tensor
    out_shape: tuple
    children: torch.Tensor  # [parents, 4], real input indices, -1 for missing/sentinel
    input_parent: torch.Tensor  # [inputs], -1 for excluded inputs
    input_slot: torch.Tensor
    sentinel_mask: torch.Tensor


def downsample_plan(coord, spatial_shape, sentinel_mask=None, backend='torch'):
    if coord.ndim != 2 or coord.shape[1] != 2:
        raise ValueError('coord must have shape [N, 2]')
    if coord.dtype not in (torch.int32, torch.int64):
        raise ValueError('coord must contain integer coordinates')
    if sentinel_mask is None:
        sentinel_mask = torch.zeros(len(coord), dtype=torch.bool, device=coord.device)
    if sentinel_mask.shape != (len(coord),) or sentinel_mask.device != coord.device:
        raise ValueError('sentinel_mask must match coord')
    if backend not in ('torch', 'triton'):
        raise ValueError("backend must be 'torch' or 'triton'")
    h, w = int(spatial_shape[0]) // 2, int(spatial_shape[1]) // 2
    if h < 0 or w < 0:
        raise ValueError('spatial_shape must be nonnegative')
    if backend == 'triton' and coord.is_cuda and 0 < h * w <= 262144:
        from ._topology import downsample_tables
        return downsample_tables(coord, spatial_shape, sentinel_mask)
    parent_coord = coord.long() // 2
    valid = ((parent_coord >= 0).all(1)
             & (parent_coord[:, 0] < h) & (parent_coord[:, 1] < w))
    inputs = torch.arange(len(coord), device=coord.device)[valid]
    flat = parent_coord[valid, 0] * w + parent_coord[valid, 1]
    flat, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    out_coord = torch.stack((flat // max(w, 1), flat % max(w, 1)), 1)
    slot = (coord[:, 0] % 2 * 2 + coord[:, 1] % 2).to(torch.int32)
    children = torch.full((len(flat), 4), -1, dtype=torch.int32, device=coord.device)
    real = ~sentinel_mask[valid].bool()
    children[inverse[real], slot[inputs[real]].long()] = inputs[real].to(torch.int32)
    input_parent = torch.full((len(coord),), -1, dtype=torch.int32, device=coord.device)
    input_parent[inputs[real]] = inverse[real].to(torch.int32)
    return DownsamplePlan(out_coord, (h, w), children, input_parent, slot,
                          (children < 0).all(1))
