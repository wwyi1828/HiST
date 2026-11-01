from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WindowPlan:
    coord: torch.Tensor
    window: torch.Tensor
    shifted_window: torch.Tensor
    members: torch.Tensor
    shifted_members: torch.Tensor
    side: int
    union: bool
    shift: int = 0


def _group_windows(coord, inputs, side, shift, spatial_shape):
    windows = (coord[inputs].long() + shift) // side
    width = (int(spatial_shape[1]) + shift + side - 1) // side
    keys = windows[:, 0] * max(width, 1) + windows[:, 1]
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    local = (coord[inputs] + shift) % side
    slot = local[:, 0] * side + local[:, 1]
    members = torch.full((len(unique), side * side), -1, dtype=torch.int32, device=coord.device)
    members[inverse, slot] = inputs.to(torch.int32)
    window = torch.full((len(coord),), -1, dtype=torch.int32, device=coord.device)
    window[inputs] = inverse.to(torch.int32)
    return window, members


def window_plan(coord, spatial_shape, window_size=3, sentinel_mask=None, union=True,
                shifted=False, backend='torch'):
    if coord.ndim != 2 or coord.shape[1] != 2 or coord.dtype not in (torch.int32, torch.int64):
        raise ValueError('coord must have integer shape [N, 2]')
    if window_size < 1 or int(window_size) != window_size:
        raise ValueError('window_size must be a positive integer')
    if union and shifted:
        raise ValueError('shifted=True requires union=False')
    if backend not in ('torch', 'triton'):
        raise ValueError("backend must be 'torch' or 'triton'")
    side = int(window_size) * 2
    if sentinel_mask is None:
        sentinel_mask = torch.zeros(len(coord), dtype=torch.bool, device=coord.device)
    if sentinel_mask.shape != (len(coord),) or sentinel_mask.device != coord.device:
        raise ValueError('sentinel_mask must match coord')
    # AttentionBuilder uses full attention if both spatial dimensions are small.
    small = int(spatial_shape[0]) < side and int(spatial_shape[1]) < side
    shift = side // 2 if shifted and not small else 0
    if backend == 'triton' and coord.is_cuda and int(spatial_shape[0]) * int(spatial_shape[1]) <= 1048576:
        from ._topology import window_tables
        return window_tables(coord, spatial_shape, sentinel_mask, side, union and not small, shift)
    inputs = torch.arange(len(coord), device=coord.device)[~sentinel_mask.bool()]
    w0, m0 = _group_windows(coord, inputs, side, shift, spatial_shape)
    if union and not small:
        w1, m1 = _group_windows(coord, inputs, side, side // 2, spatial_shape)
    else:
        w1, m1 = w0, m0
        union = False
    return WindowPlan(coord.contiguous(), w0, w1, m0, m1, side, bool(union), shift)


def local_attention(q, k, v, plan: WindowPlan, relative_bias=None,
                    alibi_slopes=None, dropout_p=0., training=False, seed=None):
    from .attention import AttentionPlan, attention
    real = (plan.window >= 0).contiguous()
    topology = AttentionPlan(plan.coord, real, 0, plan, False, 'auto')
    return attention(q, k, v, topology, relative_bias, alibi_slopes,
                     dropout_p, training, seed)
