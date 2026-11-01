from dataclasses import dataclass
import importlib.util

import torch

from .window_attention import WindowPlan, window_plan


@dataclass(frozen=True)
class AttentionPlan:
    positions: torch.Tensor
    real: torch.Tensor
    num_globals: int
    windows: object = None
    global_edges: bool = False
    backend: str = 'auto'

    @property
    def num_local(self):
        return len(self.positions)

    @property
    def num_nodes(self):
        return self.num_local + self.num_globals

    @property
    def local_capacity(self):
        if self.windows is None:
            return 0
        side = self.windows.side
        return side * side + (3 * (side // 2) ** 2 if self.windows.union else 0)


def build_attention_plans(positions, valid_globals, mode='long', window_size=3,
                          sentinel_mask=None, spatial_shape=None, backend='auto'):
    mode = {'hybrid': 'long', 'hybrid_noshift': 'long_noshift'}.get(mode, mode)
    if mode not in ('long', 'long_noshift', 'swin', 'tradswin', 'swin_global'):
        raise ValueError(f'Unsupported attention mode: {mode}')
    if backend not in ('auto', 'triton', 'torch'):
        raise ValueError("attention backend must be 'auto', 'triton' or 'torch'")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError('positions must have shape [N, 2]')
    if valid_globals < 0:
        raise ValueError('valid_globals must be nonnegative')
    positions = positions.long().contiguous()
    if sentinel_mask is None:
        sentinel_mask = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)
    if sentinel_mask.shape != (len(positions),) or sentinel_mask.device != positions.device:
        raise ValueError('sentinel mask must match positions')
    real = (~sentinel_mask.bool()).contiguous()
    # Shifted windows use the occupied extent. A larger convolution shape hint
    # would change that topology on a small canvas. Long's union already
    # contains the regular window, so it can keep the hint.
    if spatial_shape is None or mode in ('swin', 'tradswin', 'swin_global'):
        spatial_shape = (positions.max(0).values + 1).tolist() if len(positions) else (0, 0)
    topology_backend = ('triton' if positions.is_cuda and backend != 'torch'
                        and importlib.util.find_spec('triton') else 'torch')
    def windows(union=False, shifted=False):
        return window_plan(positions, spatial_shape, window_size, ~real,
                           union=union, shifted=shifted, backend=topology_backend)
    def plan(w=None, global_edges=False):
        return AttentionPlan(positions, real, int(valid_globals), w, global_edges, backend)
    if mode == 'long':
        return plan(windows(union=True)), plan(global_edges=True)
    if mode == 'long_noshift':
        return plan(windows()), plan(global_edges=True)
    return plan(windows(shifted=True)), plan(windows(), mode == 'swin_global')


def _dropout_scale(seed, query, head, slot, heads, p):
    if p == 0:
        return 1.
    mask = 0xffffffff
    x = (((query * heads + head) * 0x9e3779b9) ^ slot ^ seed.long()) & mask
    x = ((x ^ (x >> 16)) * 0x7feb352d) & mask
    x = ((x ^ (x >> 15)) * 0x846ca68b) & mask
    x = x ^ (x >> 16)
    return ((x >> 8).float() * (1. / 16777216) >= p).float() / (1. - p)


def cached_local_edges(plan):
    edges = getattr(plan, '_edges', None)
    if edges is None:
        edges = _compact_local_edges(plan)
        object.__setattr__(plan, '_edges', edges)
    return edges


def _compact_local_edges(plan):
    device = plan.positions.device
    n, nl, ng = plan.num_nodes, plan.num_local, plan.num_globals
    empty = torch.zeros(n + 1, dtype=torch.int32, device=device)
    none = torch.empty(0, dtype=torch.int32, device=device)
    if plan.windows is None or n == 0:
        return empty, none, none, empty, none, none
    counts = torch.zeros(n, dtype=torch.int32, device=device)
    idx_parts, slot_parts = [], []
    if nl:
        queries = torch.arange(nl, device=device)
        ids, slots = _local_ids(plan, queries)
        valid = ids >= 0
        local_counts = valid.sum(1).to(torch.int32)
        counts[:nl] = local_counts
        idx_parts.append(ids[valid].to(torch.int32))
        slot_parts.append(slots.to(ids.dtype).view(1, -1).expand_as(ids)[valid].to(torch.int32))
    if plan.global_edges and ng:
        real = torch.cat((plan.real, torch.ones(ng, dtype=torch.bool, device=device)))
        keys = torch.arange(n, device=device)[real].to(torch.int32)
        counts[nl:] = keys.numel()
        idx_parts.append(keys.repeat(ng))
        slot_parts.append(keys.repeat(ng))
    q_idx = torch.cat(idx_parts) if idx_parts else none
    q_slot = torch.cat(slot_parts) if slot_parts else none
    q_ptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
    q_ptr[1:] = counts.cumsum(0)
    qid = torch.repeat_interleave(torch.arange(n, device=device), counts.long())
    order = torch.argsort(q_idx.long(), stable=True) if q_idx.numel() else q_idx.long()
    k_idx = qid[order].to(torch.int32)
    k_slot = q_slot[order]
    k_ptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
    if q_idx.numel():
        k_ptr[1:] = torch.bincount(q_idx.long(), minlength=n).to(torch.int32).cumsum(0)
    return q_ptr, q_idx, q_slot, k_ptr, k_idx, k_slot


def _local_ids(plan, queries):
    w = plan.windows
    cap = plan.local_capacity
    slots = torch.arange(cap + (plan.num_globals if plan.global_edges else 0), device=queries.device)
    ids = torch.full((len(queries), len(slots)), -1, dtype=torch.long, device=queries.device)
    if w is not None and cap:
        wid = w.window[queries].long()
        first = w.side ** 2
        if len(w.members):
            ids[:, :first] = w.members[wid.clamp(min=0)].long()
        if w.union:
            half = w.side // 2
            area = half * half
            local = (plan.positions[queries] + half) % w.side
            shared = local[:, 0] // half * 2 + local[:, 1] // half
            t = torch.arange(3 * area, device=queries.device)[None, :]
            quad = t // area
            quad = quad + (quad >= shared[:, None])
            within = t % area
            slot = (quad // 2 * half + within // half) * w.side + quad % 2 * half + within % half
            swid = w.shifted_window[queries].long()
            if len(w.shifted_members):
                ids[:, first:cap] = w.shifted_members[swid.clamp(min=0)[:, None], slot].long()
        ids[wid < 0] = -1
    if plan.global_edges and plan.num_globals:
        ids[:, cap:] = plan.num_local + slots[cap:] - cap
    ids[~plan.real[queries]] = -1
    return ids, slots


def _torch_attention(q, k, v, plan, bias, slopes, p, seed):
    work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, vf = (x.to(work_dtype) for x in (q, k, v))
    n, h, d = q.shape
    parts = []
    # Empty slices preserve zero gradients for parameters with no active edges.
    zero = qf.sum() * 0 + kf.sum() * 0 + vf.sum() * 0
    if bias is not None:
        zero = zero + bias.sum() * 0
    if slopes is not None:
        zero = zero + slopes.sum() * 0
    for begin in range(0, plan.num_local, 128):
        queries = torch.arange(begin, min(begin + 128, plan.num_local), device=q.device)
        ids, slots = _local_ids(plan, queries)
        if ids.shape[1] == 0:
            parts.append(qf[queries] * 0 + zero)
            continue
        valid = ids >= 0
        safe = ids.clamp(min=0)
        scores = (qf[queries, :, None, :] * kf[safe].permute(0, 2, 1, 3)).sum(-1) * d ** -.5
        if bias is not None or slopes is not None:
            local = valid & (ids < plan.num_local)
            delta = plan.positions[safe.clamp(max=max(plan.num_local - 1, 0))] - plan.positions[queries, None]
            if bias is not None:
                side = plan.windows.side if plan.windows is not None else (bias.shape[-1] + 1) // 2
                yi = (delta[..., 0] + side - 1).clamp(0, bias.shape[-2] - 1)
                xi = (delta[..., 1] + side - 1).clamp(0, bias.shape[-1] - 1)
                pos = bias[:, yi, xi].permute(1, 0, 2).to(work_dtype)
                scores = scores + pos * local[:, None]
            if slopes is not None:
                dist = delta.to(work_dtype).square().sum(-1).sqrt()
                scores = scores - slopes.reshape(1, h, 1).to(work_dtype) * (dist * local)[:, None]
        scores = scores.masked_fill(~valid[:, None], -float('inf'))
        scores = torch.where(valid.any(-1)[:, None, None], scores, torch.zeros_like(scores))
        prob = scores.softmax(-1).masked_fill(~valid[:, None], 0)
        scale = _dropout_scale(seed, queries[:, None, None],
                               torch.arange(h, device=q.device)[None, :, None], slots[None, None, :], h, p)
        parts.append(((prob * scale)[..., None] * vf[safe].permute(0, 2, 1, 3)).sum(-2))
    if plan.num_globals:
        queries = torch.arange(plan.num_local, n, device=q.device)
        if plan.global_edges:
            ids = torch.arange(n, device=q.device)
            valid = torch.cat((plan.real, torch.ones(plan.num_globals, dtype=torch.bool, device=q.device)))
            scores = torch.einsum('qhd,khd->qhk', qf[queries], kf) * d ** -.5
            scores = scores.masked_fill(~valid[None, None], -float('inf'))
            prob = scores.softmax(-1)
            scale = _dropout_scale(seed, queries[:, None, None],
                                   torch.arange(h, device=q.device)[None, :, None], ids[None, None, :], h, p)
            parts.append(torch.einsum('qhk,khd->qhd', prob * scale, vf))
        else:
            parts.append(qf[queries] * 0 + zero)
    return ((torch.cat(parts) if parts else qf * 0) + zero).to(q.dtype)


def attention(q, k, v, plan, bias_table=None, alibi_slopes=None,
              dropout_p=0., training=False, seed=None):
    if not isinstance(plan, AttentionPlan):
        raise TypeError('plan must be an AttentionPlan')
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != plan.num_nodes:
        raise ValueError('q/k/v must have matching shape [plan.num_nodes, heads, head_dim]')
    if q.shape[1] < 1 or q.shape[2] < 1:
        raise ValueError('heads and head_dim must be positive')
    if any(x.device != q.device or x.dtype != q.dtype for x in (k, v)):
        raise ValueError('q/k/v must have the same device and dtype')
    if plan.positions.device != q.device:
        raise ValueError('plan and features must be on the same device')
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError('q/k/v must be floating point tensors')
    if not 0 <= dropout_p < 1:
        raise ValueError('dropout_p must be in [0, 1)')
    if bias_table is not None:
        side = plan.windows.side if plan.windows is not None else (bias_table.shape[-1] + 1) // 2
        if bias_table.shape != (q.shape[1], 2 * side - 1, 2 * side - 1):
            raise ValueError('bias table must have shape [heads, 2*side-1, 2*side-1]')
    if alibi_slopes is not None and alibi_slopes.numel() != q.shape[1]:
        raise ValueError('ALiBi slopes must have one value per head')
    for x in (bias_table, alibi_slopes):
        if x is not None and (x.device != q.device or not x.is_floating_point()):
            raise ValueError('position parameters must be floating point on the feature device')
    p = float(dropout_p) if training else 0.
    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,), device=q.device, dtype=torch.int64) if p else torch.empty(0, device=q.device, dtype=torch.int64)
    else:
        seed = torch.as_tensor(seed, device=q.device, dtype=torch.int64).reshape(1)
    use_cuda = q.is_cuda and q.dtype != torch.float64 and plan.backend != 'torch'
    if use_cuda and not importlib.util.find_spec('triton'):
        if plan.backend == 'triton':
            raise RuntimeError('Triton backend requested but Triton is unavailable')
        use_cuda = False
    if use_cuda:
        from ._attention_kernels import fused_attention
        return fused_attention(q, k, v, plan, bias_table, alibi_slopes, p, seed)
    return _torch_attention(q, k, v, plan, bias_table, alibi_slopes, p, seed)
