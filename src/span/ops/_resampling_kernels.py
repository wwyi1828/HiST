import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

@triton.jit
def _bf16_round(value):
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 32767 + ((bits >> 16) & 1)) & 4294901760
    return rounded.to(tl.float32, bitcast=True)



@triton.jit
def _merge_row(X, Children, Sentinel, p, cols, C: tl.constexpr,
               EDGE: tl.constexpr, EDGE_EPS: tl.constexpr, SOFT_BF16: tl.constexpr):
    slot, channel = cols // C, cols % C
    child = tl.load(Children + p * 4 + slot, cols < C * 4, other=-1)
    sentinel = tl.load(Sentinel + child, (cols < C * 4) & (child >= 0), other=0).to(tl.int1)
    real = (child >= 0) & ~sentinel
    value = tl.load(X + child * C + channel, (cols < C * 4) & real, other=0)
    scale = tl.full((), 1., tl.float32)
    if EDGE:
        count = tl.full((), 0., tl.float32)
        for k in tl.static_range(4):
            ci = tl.load(Children + p * 4 + k)
            si = tl.load(Sentinel + ci, ci >= 0, other=1).to(tl.int1)
            count += ((ci >= 0) & ~si).to(tl.float32)
        denom = tl.maximum(tl.maximum(count, 1.), EDGE_EPS)
        scale = (1. / denom).to(value.dtype).to(tl.float32)
        if SOFT_BF16:
            scale = _bf16_round(scale)
        value = (value.to(tl.float32) * scale).to(value.dtype)
        if SOFT_BF16:
            value = _bf16_round(value)
    return value.to(tl.float32), child, sentinel, real, scale



@triton.jit
def _merge_forward(X, Children, Sentinel, Gamma, Beta, Out, Mean, Rstd,
                   C: tl.constexpr, NORM: tl.constexpr, NORM_EPS: tl.constexpr,
                   EDGE: tl.constexpr, EDGE_EPS: tl.constexpr, SOFT_BF16: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x, _, sentinel, _, _ = _merge_row(X, Children, Sentinel, p, cols, C, EDGE, EDGE_EPS, SOFT_BF16)
    if NORM:
        mean = tl.sum(x, 0) / (4 * C)
        centered = tl.where(cols < 4 * C, x - mean, 0.)
        var = tl.sum(centered * centered, 0) / (4 * C)
        rstd = 1. / tl.sqrt(var + NORM_EPS)
        y = (x - mean) * rstd
        gamma = tl.load(Gamma + cols, cols < 4 * C, other=0).to(tl.float32)
        beta = tl.load(Beta + cols, cols < 4 * C, other=0).to(tl.float32)
        y = y * gamma + beta
        tl.store(Mean + p, mean)
        tl.store(Rstd + p, rstd)
    else:
        y = x
    tl.store(Out + p * (4 * C) + cols, tl.where(sentinel, 0., y), cols < 4 * C)



@triton.jit
def _merge_backward_x(X, Children, Sentinel, Gamma, DY, Mean, Rstd, DX,
                      C: tl.constexpr, NORM: tl.constexpr, EDGE: tl.constexpr,
                      EDGE_EPS: tl.constexpr, SOFT_BF16: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x, child, sentinel, real, scale = _merge_row(X, Children, Sentinel, p, cols, C, EDGE, EDGE_EPS, SOFT_BF16)
    dy = tl.load(DY + p * 4 * C + cols, (cols < 4 * C) & ~sentinel, other=0).to(tl.float32)
    if NORM:
        mean, rstd = tl.load(Mean + p), tl.load(Rstd + p)
        xhat = (x - mean) * rstd
        gamma = tl.load(Gamma + cols, cols < 4 * C, other=0).to(tl.float32)
        z = dy * gamma
        s1 = tl.sum(z, 0) / (4 * C)
        s2 = tl.sum(tl.where(cols < 4 * C, z * xhat, 0.), 0) / (4 * C)
        dx = (z - s1 - xhat * s2) * rstd
    else:
        dx = dy
    tl.store(DX + child * C + cols % C, dx * scale, (cols < 4 * C) & real)



@triton.jit
def _merge_backward_params(X, Children, Sentinel, DY, Mean, Rstd, DG, DB,
                           P, C: tl.constexpr, EDGE: tl.constexpr,
                           EDGE_EPS: tl.constexpr, SOFT_BF16: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    slot, channel = cols // C, cols % C
    child = tl.load(Children + rows[:, None] * 4 + slot[None, :],
                    (rows[:, None] < P) & (cols[None, :] < 4 * C), other=-1)
    sentinel = tl.load(Sentinel + child, child >= 0, other=0).to(tl.int1)
    real = (child >= 0) & ~sentinel
    x = tl.load(X + child * C + channel[None, :], real, other=0)
    if EDGE:
        count = tl.full((ROWS,), 0., tl.float32)
        for k in tl.static_range(4):
            ci = tl.load(Children + rows * 4 + k, rows < P, other=-1)
            si = tl.load(Sentinel + ci, ci >= 0, other=1).to(tl.int1)
            count += ((ci >= 0) & ~si).to(tl.float32)
        scale = (1. / tl.maximum(tl.maximum(count, 1.), EDGE_EPS)).to(x.dtype).to(tl.float32)
        if SOFT_BF16:
            scale = _bf16_round(scale)
        x = (x.to(tl.float32) * scale[:, None]).to(x.dtype)
        if SOFT_BF16:
            x = _bf16_round(x)
    x = x.to(tl.float32)
    mean = tl.load(Mean + rows, rows < P, other=0)
    rstd = tl.load(Rstd + rows, rows < P, other=0)
    xhat = (x - mean[:, None]) * rstd[:, None]
    dy = tl.load(DY + rows[:, None] * (4 * C) + cols[None, :],
                 (rows[:, None] < P) & (cols[None, :] < 4 * C) & ~sentinel, other=0).to(tl.float32)
    dg, db = tl.sum(dy * xhat, 0), tl.sum(dy, 0)
    tl.store(DG + tl.program_id(0) * 4 * C + cols, dg, cols < 4 * C)
    tl.store(DB + tl.program_id(0) * 4 * C + cols, db, cols < 4 * C)



class _MergeGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, children, sentinel, gamma, beta, norm_eps, edge, edge_eps, soft_bf16):
        p, c = len(children), feat.shape[1]
        out = feat.new_empty((p, 4 * c))
        mean = torch.empty(p, device=feat.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        if p:
            with torch.cuda.device(feat.device):
                _merge_forward[(p,)](feat, children, sentinel, gamma, beta, out, mean, rstd,
                                     c, gamma is not None, norm_eps, edge, edge_eps, soft_bf16,
                                     triton.next_power_of_2(4 * c), num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(feat, children, sentinel, gamma, beta, mean, rstd)
        ctx.edge, ctx.edge_eps, ctx.soft_bf16 = edge, edge_eps, soft_bf16
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        feat, children, sentinel, gamma, beta, mean, rstd = ctx.saved_tensors
        p, c = len(children), feat.shape[1]
        grad = grad.contiguous()
        dx = torch.zeros_like(feat) if ctx.needs_input_grad[0] else None
        dg = db = None
        with torch.cuda.device(feat.device):
            if p and dx is not None:
                _merge_backward_x[(p,)](feat, children, sentinel, gamma, grad, mean, rstd, dx,
                                        c, gamma is not None, ctx.edge, ctx.edge_eps, ctx.soft_bf16,
                                        triton.next_power_of_2(4 * c), num_warps=4, enable_fp_fusion=False)
            if gamma is not None and (ctx.needs_input_grad[3] or ctx.needs_input_grad[4]):
                parts = triton.cdiv(p, 128)
                pg = torch.empty((parts, 4 * c), dtype=torch.float32, device=feat.device)
                pb = torch.empty_like(pg)
                if p:
                    _merge_backward_params[(parts, triton.cdiv(4 * c, 32))](
                        feat, children, sentinel, grad, mean, rstd, pg, pb, p, c,
                        ctx.edge, ctx.edge_eps, ctx.soft_bf16, 128, 32, num_warps=4, enable_fp_fusion=False)
                dg = pg.sum(0).to(gamma.dtype) if ctx.needs_input_grad[3] else None
                db = pb.sum(0).to(beta.dtype) if ctx.needs_input_grad[4] else None
        return dx, None, None, dg, db, None, None, None, None



@triton.jit
def _expand_pixel_coord(Coord, Sentinel, OutCoord, Parent, Slot, Mask,
                        N, S: tl.constexpr, FACTOR: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n, slot = row // S, row % S
    y = tl.load(Coord + n * 2, n < N, other=0)
    x = tl.load(Coord + n * 2 + 1, n < N, other=0)
    sentinel = tl.load(Sentinel + n, n < N, other=0)
    tl.store(OutCoord + row * 2, y * FACTOR + slot // FACTOR, n < N)
    tl.store(OutCoord + row * 2 + 1, x * FACTOR + slot % FACTOR, n < N)
    tl.store(Parent + row, n, n < N)
    tl.store(Slot + row, slot, n < N)
    tl.store(Mask + row, sentinel, n < N)



@triton.jit
def _pixel_reverse(Parent, Slot, Reverse, M, S: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parent = tl.load(Parent + row, row < M, other=-1)
    slot = tl.load(Slot + row, row < M, other=0)
    tl.store(Reverse + parent * S + slot, row, (row < M) & (parent >= 0))



@triton.jit
def _dot(a, b, HALF: tl.constexpr, BF16: tl.constexpr):
    if HALF:
        return tl.dot(a.to(tl.float16), b.to(tl.float16), allow_tf32=False)
    elif BF16:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), allow_tf32=False)
    return tl.dot(a.to(tl.float32), b.to(tl.float32), allow_tf32=False)



@triton.jit
def _pixel_forward(X, W, Bias, Parent, Order, Starts, Counts, Mask, Y,
                   C: tl.constexpr, O: tl.constexpr, S: tl.constexpr,
                   BIAS: tl.constexpr, HALF: tl.constexpr, BF16: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    slot = tl.program_id(2)
    start, count = tl.load(Starts + slot), tl.load(Counts + slot)
    if tl.program_id(0) * BM < count:
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        ids = tl.load(Order + start + rows, rows < count, other=0)
        parent = tl.load(Parent + ids, rows < count, other=-1)
        masked = tl.load(Mask + ids, rows < count, other=1).to(tl.int1)
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range((C + BK - 1) // BK):
            rk = k * BK + ks
            a = tl.load(X + parent[:, None] * C + rk[None, :],
                        (rows[:, None] < count) & (parent[:, None] >= 0) & ~masked[:, None] & (rk[None, :] < C), other=0)
            b = tl.load(W + (cols[None, :] * S + slot) * C + rk[:, None],
                        (cols[None, :] < O) & (rk[:, None] < C), other=0)
            acc += _dot(a, b, HALF, BF16)
        if BIAS:
            acc += tl.load(Bias + cols * S + slot, cols < O, other=0)[None, :].to(tl.float32)
        acc = tl.where(masked[:, None], 0., acc)
        tl.store(Y + ids[:, None] * O + cols[None, :], acc,
                 (rows[:, None] < count) & (cols[None, :] < O))



@triton.jit
def _pixel_backward_x(W, DY, Reverse, Mask, DX, N, C: tl.constexpr,
                      O: tl.constexpr, S: tl.constexpr, HALF: tl.constexpr, BF16: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for slot in range(S):
        ids = tl.load(Reverse + rows * S + slot, rows < N, other=-1)
        masked = tl.load(Mask + ids, ids >= 0, other=1).to(tl.int1)
        for k in range((O + BK - 1) // BK):
            rk = k * BK + ks
            a = tl.load(DY + ids[:, None] * O + rk[None, :],
                        (rows[:, None] < N) & (ids[:, None] >= 0) & ~masked[:, None] & (rk[None, :] < O), other=0)
            b = tl.load(W + (rk[:, None] * S + slot) * C + cols[None, :],
                        (rk[:, None] < O) & (cols[None, :] < C), other=0)
            acc += _dot(a, b, HALF, BF16)
    tl.store(DX + rows[:, None] * C + cols[None, :], acc,
             (rows[:, None] < N) & (cols[None, :] < C))



@triton.jit
def _pixel_backward_w(X, DY, Parent, Order, Starts, Counts, Mask, DW,
                      C: tl.constexpr, O: tl.constexpr, S: tl.constexpr,
                      HALF: tl.constexpr, BF16: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    slot = tl.program_id(2)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    start, count = tl.load(Starts + slot), tl.load(Counts + slot)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range((count + BK - 1) // BK):
        rk = k * BK + ks
        ids = tl.load(Order + start + rk, rk < count, other=0)
        parent = tl.load(Parent + ids, rk < count, other=-1)
        masked = tl.load(Mask + ids, rk < count, other=1).to(tl.int1)
        a = tl.load(X + parent[None, :] * C + rows[:, None],
                    (rows[:, None] < C) & (rk[None, :] < count) & (parent[None, :] >= 0) & ~masked[None, :], other=0)
        b = tl.load(DY + ids[:, None] * O + cols[None, :],
                    (rk[:, None] < count) & ~masked[:, None] & (cols[None, :] < O), other=0)
        acc += _dot(a, b, HALF, BF16)
    tl.store(DW + (cols[None, :] * S + slot) * C + rows[:, None], acc,
             (rows[:, None] < C) & (cols[None, :] < O))


@triton.jit
def _pixel_backward_bias(DY, Order, Starts, Counts, Mask, DB,
                         O: tl.constexpr, S: tl.constexpr, ROWS: tl.constexpr):
    slot, channel = tl.program_id(1), tl.program_id(0)
    rows = tl.arange(0, ROWS)
    start, count = tl.load(Starts + slot), tl.load(Counts + slot)
    result = tl.full((), 0., tl.float32)
    for block in range((count + ROWS - 1) // ROWS):
        rk = block * ROWS + rows
        ids = tl.load(Order + start + rk, rk < count, other=0)
        masked = tl.load(Mask + ids, rk < count, other=1).to(tl.int1)
        dy = tl.load(DY + ids * O + channel, (rk < count) & ~masked, other=0).to(tl.float32)
        result += tl.sum(dy, 0)
    tl.store(DB + channel * S + slot, result)



class _PixelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, weight, bias, parent, order, starts, counts, reverse, mask, s, o, compute):
        m, c = len(parent), feat.shape[1]
        out = torch.empty((m, o), dtype=compute, device=feat.device)
        half = compute == torch.float16
        bf16 = compute == torch.bfloat16 and torch.cuda.get_device_capability(feat.device)[0] >= 8
        with torch.cuda.device(feat.device):
            if m:
                _pixel_forward[(triton.cdiv(m, 16), triton.cdiv(o, 32), s)](
                    feat, weight, bias, parent, order, starts, counts, mask, out,
                    c, o, s, bias is not None, half, bf16, 16, 32, 32, num_warps=4)
        ctx.save_for_backward(feat, weight, bias, parent, order, starts, counts, reverse, mask)
        ctx.s, ctx.o, ctx.half, ctx.bf16 = s, o, half, bf16
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        feat, weight, bias, parent, order, starts, counts, reverse, mask = ctx.saved_tensors
        c, o, s = feat.shape[1], ctx.o, ctx.s
        dx = dw = db = None
        grad = grad.contiguous()
        with torch.cuda.device(feat.device):
            if ctx.needs_input_grad[0]:
                dx = torch.empty_like(feat)
                if len(feat):
                    _pixel_backward_x[(triton.cdiv(len(feat), 16), triton.cdiv(c, 32))](
                        weight, grad, reverse, mask, dx, len(feat), c, o, s,
                        ctx.half, ctx.bf16, 16, 32, 32, num_warps=4)
            if ctx.needs_input_grad[1]:
                dw = torch.empty_like(weight)
                _pixel_backward_w[(triton.cdiv(c, 16), triton.cdiv(o, 32), s)](
                    feat, grad, parent, order, starts, counts, mask, dw, c, o, s,
                    ctx.half, ctx.bf16, 16, 32, 32, num_warps=4)
            if bias is not None and ctx.needs_input_grad[2]:
                db = torch.empty_like(bias)
                _pixel_backward_bias[(o, s)](
                    grad, order, starts, counts, mask, db, o, s, 128, num_warps=4)
        return dx, dw, db, None, None, None, None, None, None, None, None, None


def merge_gather(feat, children, sentinel, gamma, beta, norm_eps, edge, edge_eps, soft_bf16):
    return _MergeGather.apply(feat, children, sentinel, gamma, beta, norm_eps, edge, edge_eps, soft_bf16)


def pixel_linear(feat, weight, bias, parent, order, starts, counts, reverse, mask, s, o, compute):
    return _PixelLinear.apply(feat, weight, bias, parent, order, starts, counts, reverse, mask, s, o, compute)


def expand_pixel_coordinates(coord, sentinel, factor):
    n, s = len(coord), factor * factor
    m = n * s
    out_coord = torch.empty((m, 2), dtype=torch.int64, device=coord.device)
    parent = torch.empty(m, dtype=torch.int64, device=coord.device)
    slot = torch.empty(m, dtype=torch.int64, device=coord.device)
    mask = torch.empty(m, dtype=torch.bool, device=coord.device)
    if m:
        with torch.cuda.device(coord.device):
            _expand_pixel_coord[(triton.cdiv(m, 256),)](
                coord.contiguous(), sentinel, out_coord, parent, slot, mask, n, s, factor, 256)
    return out_coord, parent, slot, mask


def fill_pixel_reverse(parent, slot, reverse, s):
    if len(parent):
        with torch.cuda.device(parent.device):
            _pixel_reverse[(triton.cdiv(len(parent), 256),)](parent, slot, reverse, len(parent), s, 256)


@triton.jit
def _shuffle_layout(Projected, Parent, Slot, Mask, Out,
                    M, O: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, channel = i // O, i % O
    parent = tl.load(Parent + row, row < M, other=0)
    slot = tl.load(Slot + row, row < M, other=0)
    masked = tl.load(Mask + row, row < M, other=1).to(tl.int1)
    value = tl.load(Projected + parent * O * S + channel * S + slot,
                    (row < M) & ~masked, other=0)
    tl.store(Out + i, value, row < M)


@triton.jit
def _shuffle_layout_backward(DY, Parent, Slot, Mask, DX,
                             M, O: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, channel = i // O, i % O
    parent = tl.load(Parent + row, row < M, other=0)
    slot = tl.load(Slot + row, row < M, other=0)
    masked = tl.load(Mask + row, row < M, other=1).to(tl.int1)
    value = tl.load(DY + i, (row < M) & ~masked, other=0)
    tl.store(DX + parent * O * S + channel * S + slot, value, row < M)


class _ShuffleLayout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, parent, slot, mask, o, s):
        out = projected.new_empty((len(parent), o))
        if out.numel():
            with torch.cuda.device(projected.device):
                _shuffle_layout[(triton.cdiv(out.numel(), 256),)](
                    projected, parent, slot, mask, out, len(parent), o, s, 256)
        ctx.save_for_backward(parent, slot, mask)
        ctx.shape, ctx.o, ctx.s = projected.shape, o, s
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        parent, slot, mask = ctx.saved_tensors
        grad = grad.contiguous()
        dx = grad.new_empty(ctx.shape)
        if grad.numel():
            with torch.cuda.device(grad.device):
                _shuffle_layout_backward[(triton.cdiv(grad.numel(), 256),)](
                    grad, parent, slot, mask, dx, len(parent), ctx.o, ctx.s, 256)
        return dx, None, None, None, None, None


def shuffle_layout(projected, parent, slot, mask, o, s):
    # BF16 loads/stores are unsupported by pre-Ampere Triton. Keeping only the
    # layout conversion in FP32 supplies identical BF16 values and cast gradients.
    soft = projected.dtype == torch.bfloat16 and torch.cuda.get_device_capability(projected.device)[0] < 8
    result = _ShuffleLayout.apply(projected.float() if soft else projected,
                                   parent, slot, mask, o, s)
    return result.to(projected.dtype) if soft else result
