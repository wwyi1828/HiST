import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


# Spatial counts and offsets stay runtime values: differently sized slides
# reuse compiled code, including values 1 and multiples of the tile size.
@triton.jit(do_not_specialize=['N', 'H', 'W', 'OH', 'OW'])
def _populate_lookup(Coord, Sentinel, Lookup, Occupied, N,
                     H, W, OH, OW,
                     KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                     DH: tl.constexpr, DW: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
                     DISCOVER: tl.constexpr, HAS_SENTINEL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(Coord + i * 2, i < N, other=-1)
    x = tl.load(Coord + i * 2 + 1, i < N, other=-1)
    valid = (i < N) & (y >= 0) & (x >= 0) & (y < H) & (x < W)
    real = valid
    if HAS_SENTINEL:
        real &= ~tl.load(Sentinel + i, i < N, other=1).to(tl.int1)
    tl.store(Lookup + y * W + x, tl.where(real, i, -1), valid)
    if DISCOVER:
        for k in tl.static_range(KH * KW):
            dy, dx = y + PH - k // KW * DH, x + PW - k % KW * DW
            oy, ox = dy // SH, dx // SW
            found = valid & (dy >= 0) & (dx >= 0) & (dy % SH == 0) & (dx % SW == 0)
            found &= (oy < OH) & (ox < OW)
            tl.atomic_or(Occupied + oy * OW + ox, 1, found, sem='relaxed')


@triton.jit(do_not_specialize=['M', 'H', 'W', 'OW'])
def _write_neighborhoods(Flat, Coord, Lookup, Indices, Reverse, M,
                         H, W, OW,
                         KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                         DH: tl.constexpr, DW: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
                         TRANSPOSE: tl.constexpr, FROM_FLAT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m, k = i // (KH * KW), i % (KH * KW)
    if FROM_FLAT:
        flat = tl.load(Flat + m, m < M, other=0)
        y, x = flat // OW, flat % OW
        tl.store(Coord + m * 2, y, (m < M) & (k == 0))
        tl.store(Coord + m * 2 + 1, x, (m < M) & (k == 0))
    else:
        y = tl.load(Coord + m * 2, m < M, other=0)
        x = tl.load(Coord + m * 2 + 1, m < M, other=0)
    if TRANSPOSE:
        dy, dx = y + PH - k // KW * DH, x + PW - k % KW * DW
        iy, ix = dy // SH, dx // SW
        valid = (dy >= 0) & (dx >= 0) & (dy % SH == 0) & (dx % SW == 0)
    else:
        iy, ix = y * SH - PH + k // KW * DH, x * SW - PW + k % KW * DW
        valid = (iy >= 0) & (ix >= 0)
    valid &= (m < M) & (iy < H) & (ix < W)
    src = tl.load(Lookup + iy * W + ix, valid, other=-1)
    tl.store(Indices + i, src, m < M)
    tl.store(Reverse + src * KH * KW + k, m, (m < M) & (src >= 0))


def _native_plan(coord, shape, target_coord, sentinel_mask, kernel, stride,
                 dilation, padding, out_shape, transposed):
    from .convolution import ConvPlan
    device = coord.device
    n, k = len(coord), kernel[0] * kernel[1]
    discover = target_coord is None
    with torch.cuda.device(device):
        lookup = torch.full((shape[0] * shape[1],), -1, dtype=torch.int32, device=device)
        occupied = torch.zeros(out_shape[0] * out_shape[1], dtype=torch.int32, device=device) if discover else None
        if n:
            _populate_lookup[(triton.cdiv(n, 256),)](
                coord, sentinel_mask, lookup, occupied, n, *shape, *out_shape,
                *kernel, *stride, *dilation, *padding, discover, sentinel_mask is not None, 256)
        if discover:
            flat = occupied.nonzero().flatten()
            out_coord = torch.empty((len(flat), 2), dtype=torch.long, device=device)
        else:
            flat, out_coord = None, target_coord.long().contiguous()
        m = len(out_coord)
        indices = torch.empty((m, k), dtype=torch.int32, device=device)
        reverse = torch.full((n, k), -1, dtype=torch.int32, device=device)
        if m:
            _write_neighborhoods[(triton.cdiv(m * k, 256),)](
                flat, out_coord, lookup, indices, reverse, m, *shape, out_shape[1],
                *kernel, *stride, *dilation, *padding, transposed, discover, 256)
    return ConvPlan(coord, out_coord, shape, out_shape, indices, reverse,
                    (indices < 0).all(1), kernel, stride, dilation, padding, transposed)


@triton.jit(do_not_specialize=['N'])
def _expand_reverse(Parent, Slot, Reverse, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n, k = i // 4, i % 4
    parent = tl.load(Parent + n, n < N, other=-1)
    slot = tl.load(Slot + n, n < N, other=0)
    tl.store(Reverse + i, tl.where(slot == k, parent, -1), n < N)


@triton.jit
def _weight_offsets(ic, oc, k, IC: tl.constexpr, OC: tl.constexpr,
                    K: tl.constexpr, GROUPS: tl.constexpr, TRANSPOSE: tl.constexpr):
    if TRANSPOSE:
        return ic * (OC // GROUPS) * K + (oc % (OC // GROUPS)) * K + k
    return oc * (IC // GROUPS) * K + (ic % (IC // GROUPS)) * K + k


@triton.jit(do_not_specialize=['M'])
def _matmul_forward(X, W, Indices, Y, M, IC: tl.constexpr,
                    OC: tl.constexpr, K: tl.constexpr, GROUPS: tl.constexpr,
                    TRANSPOSE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    BR: tl.constexpr):
    group = tl.program_id(2)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    o = tl.program_id(1) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv((IC // GROUPS) * K, BR)):
        reduction = block * BR + r
        local_ic, k = reduction // K, reduction % K
        ic, oc = group * (IC // GROUPS) + local_ic, group * (OC // GROUPS) + o
        src = tl.load(Indices + m[:, None] * K + k[None, :],
                      (m[:, None] < M) & (reduction[None, :] < (IC // GROUPS) * K), other=-1)
        x = tl.load(X + src * IC + ic[None, :],
                    (src >= 0) & (local_ic[None, :] < IC // GROUPS), other=0)
        w_offset = _weight_offsets(ic[:, None], oc[None, :], k[:, None], IC, OC, K, GROUPS, TRANSPOSE)
        w = tl.load(W + w_offset, (local_ic[:, None] < IC // GROUPS) & (o[None, :] < OC // GROUPS), other=0)
        acc += tl.dot(x, w, allow_tf32=False)
    tl.store(Y + m[:, None] * OC + group * (OC // GROUPS) + o[None, :], acc,
             (m[:, None] < M) & (o[None, :] < OC // GROUPS))


@triton.jit(do_not_specialize=['N'])
def _matmul_backward_x(DY, W, Reverse, DX, N, IC: tl.constexpr,
                       OC: tl.constexpr, K: tl.constexpr, GROUPS: tl.constexpr,
                       TRANSPOSE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       BR: tl.constexpr):
    group = tl.program_id(2)
    n = tl.program_id(0) * BM + tl.arange(0, BM)
    i = tl.program_id(1) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv((OC // GROUPS) * K, BR)):
        reduction = block * BR + r
        local_oc, k = reduction // K, reduction % K
        ic, oc = group * (IC // GROUPS) + i, group * (OC // GROUPS) + local_oc
        row = tl.load(Reverse + n[:, None] * K + k[None, :],
                      (n[:, None] < N) & (reduction[None, :] < (OC // GROUPS) * K), other=-1)
        dy = tl.load(DY + row * OC + oc[None, :],
                     (row >= 0) & (local_oc[None, :] < OC // GROUPS), other=0)
        w_offset = _weight_offsets(ic[None, :], oc[:, None], k[:, None], IC, OC, K, GROUPS, TRANSPOSE)
        w = tl.load(W + w_offset, (local_oc[:, None] < OC // GROUPS) & (i[None, :] < IC // GROUPS), other=0)
        acc += tl.dot(dy, w, allow_tf32=False)
    tl.store(DX + n[:, None] * IC + group * (IC // GROUPS) + i[None, :], acc,
             (n[:, None] < N) & (i[None, :] < IC // GROUPS))


@triton.jit(do_not_specialize=['M'])
def _matmul_backward_w(X, DY, Indices, DW, M, IC: tl.constexpr,
                       OC: tl.constexpr, K: tl.constexpr, GROUPS: tl.constexpr,
                       TRANSPOSE: tl.constexpr, BI: tl.constexpr, BO: tl.constexpr,
                       BR: tl.constexpr):
    group_kernel = tl.program_id(2)
    group, k = group_kernel // K, group_kernel % K
    i = tl.program_id(0) * BI + tl.arange(0, BI)
    o = tl.program_id(1) * BO + tl.arange(0, BO)
    r = tl.arange(0, BR)
    ic, oc = group * (IC // GROUPS) + i, group * (OC // GROUPS) + o
    acc = tl.zeros((BI, BO), tl.float32)
    for block in range(tl.cdiv(M, BR)):
        m = block * BR + r
        src = tl.load(Indices + m * K + k, m < M, other=-1)
        x = tl.load(X + src[None, :] * IC + ic[:, None],
                    (src[None, :] >= 0) & (i[:, None] < IC // GROUPS), other=0)
        dy = tl.load(DY + m[:, None] * OC + oc[None, :],
                     (m[:, None] < M) & (o[None, :] < OC // GROUPS), other=0)
        acc += tl.dot(x, dy, allow_tf32=False)
    offset = _weight_offsets(ic[:, None], oc[None, :], k, IC, OC, K, GROUPS, TRANSPOSE)
    tl.store(DW + offset, acc, (i[:, None] < IC // GROUPS) & (o[None, :] < OC // GROUPS))


@triton.jit(do_not_specialize=['M'])
def _depthwise_forward(X, W, Indices, Y, M, IC: tl.constexpr,
                       OC: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m, oc = a // OC, a % OC
    ic = oc // (OC // IC)
    acc = tl.full((BLOCK,), 0., tl.float32)
    for k in tl.static_range(K):
        src = tl.load(Indices + m * K + k, m < M, other=-1)
        x = tl.load(X + src * IC + ic, (m < M) & (src >= 0), other=0).to(tl.float32)
        w = tl.load(W + oc * K + k, m < M, other=0).to(tl.float32)
        acc += x * w
    tl.store(Y + a, acc, m < M)


@triton.jit(do_not_specialize=['N'])
def _depthwise_backward_x(DY, W, Reverse, DX, N, IC: tl.constexpr,
                          OC: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n, ic = a // IC, a % IC
    acc = tl.full((BLOCK,), 0., tl.float32)
    for k in tl.static_range(K):
        row = tl.load(Reverse + n * K + k, n < N, other=-1)
        for mult in tl.static_range(OC // IC):
            oc = ic * (OC // IC) + mult
            dy = tl.load(DY + row * OC + oc, (n < N) & (row >= 0), other=0).to(tl.float32)
            w = tl.load(W + oc * K + k, n < N, other=0).to(tl.float32)
            acc += dy * w
    tl.store(DX + a, acc, n < N)


@triton.jit(do_not_specialize=['M'])
def _depthwise_backward_w(X, DY, Indices, Partials, M, IC: tl.constexpr,
                          OC: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr,
                          COLS: tl.constexpr):
    m = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    columns = tl.program_id(1) * COLS + tl.arange(0, COLS)
    oc = columns[None, :]
    ic = oc // (OC // IC)
    dy = tl.load(DY + m * OC + oc, (m < M) & (oc < OC), other=0).to(tl.float32)
    for k in tl.static_range(K):
        src = tl.load(Indices + m * K + k, m < M, other=-1)
        x = tl.load(X + src * IC + ic, (m < M) & (oc < OC) & (src >= 0), other=0).to(tl.float32)
        dw = tl.sum(x * dy, 0)
        tl.store(Partials + tl.program_id(0) * OC * K + columns * K + k, dw, columns < OC)


class _IndexedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, weight, indices, reverse, groups, transposed, out_channels):
        m, k = indices.shape
        ic, oc = feat.shape[1], out_channels
        out = feat.new_empty((m, oc))
        depthwise = groups == ic
        if m:
            if depthwise:
                _depthwise_forward[(triton.cdiv(m * oc, 256),)](
                    feat, weight, indices, out, m, ic, oc, k, 256, enable_fp_fusion=False)
            else:
                _matmul_forward[(triton.cdiv(m, 16), triton.cdiv(oc // groups, 32), groups)](
                    feat, weight, indices, out, m, ic, oc, k, groups, transposed,
                    16, 32, 32, num_warps=4)
        ctx.save_for_backward(feat, weight, indices, reverse)
        ctx.groups, ctx.transposed, ctx.depthwise, ctx.oc = groups, transposed, depthwise, oc
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        feat, weight, indices, reverse = ctx.saved_tensors
        grad = grad.contiguous()
        n, ic, oc = len(feat), feat.shape[1], ctx.oc
        m, k = indices.shape
        dx = dw = None
        with torch.cuda.device(feat.device):
            if ctx.needs_input_grad[0]:
                dx = torch.empty_like(feat)
                if n:
                    if ctx.depthwise:
                        _depthwise_backward_x[(triton.cdiv(n * ic, 256),)](
                            grad, weight, reverse, dx, n, ic, oc, k, 256, enable_fp_fusion=False)
                    else:
                        _matmul_backward_x[(triton.cdiv(n, 16), triton.cdiv(ic // ctx.groups, 32), ctx.groups)](
                            grad, weight, reverse, dx, n, ic, oc, k, ctx.groups, ctx.transposed,
                            16, 32, 32, num_warps=4)
            if ctx.needs_input_grad[1]:
                if ctx.depthwise:
                    parts = triton.cdiv(m, 128)
                    partials = torch.empty((parts, oc, k), device=feat.device, dtype=torch.float32)
                    if m:
                        _depthwise_backward_w[(parts, triton.cdiv(oc, 32))](
                            feat, grad, indices, partials, m, ic, oc, k, 128, 32,
                            enable_fp_fusion=False)
                    dw = partials.sum(0).reshape_as(weight).to(weight.dtype)
                else:
                    dw = torch.empty_like(weight)
                    _matmul_backward_w[(triton.cdiv(ic // ctx.groups, 16),
                                        triton.cdiv(oc // ctx.groups, 32), ctx.groups * k)](
                        feat, grad, indices, dw, m, ic, oc, k, ctx.groups, ctx.transposed,
                        16, 32, 32, num_warps=4)
        return dx, dw, None, None, None, None, None


def expand_reverse(parent, slot, reverse):
    with torch.cuda.device(parent.device):
        _expand_reverse[(triton.cdiv(len(parent) * 4, 256),)](
            parent, slot, reverse, len(parent), 256)



@triton.jit(do_not_specialize=['START', 'ROWS'])
def _pack_neighborhood(X, Indices, Packed, START, ROWS,
                        CHANNELS: tl.constexpr, GROUPS: tl.constexpr,
                        K: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(1)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, reduction = i // ((CHANNELS // GROUPS) * K), i % ((CHANNELS // GROUPS) * K)
    channel, slot = reduction // K, reduction % K
    src = tl.load(Indices + (row + START) * K + slot, row < ROWS, other=-1)
    value = tl.load(X + src * CHANNELS + group * (CHANNELS // GROUPS) + channel,
                    (row < ROWS) & (src >= 0), other=0)
    tl.store(Packed + group * ROWS * (CHANNELS // GROUPS) * K + i, value, row < ROWS)


def _pack(feat, indices, start, rows, groups):
    channels, k = feat.shape[1], indices.shape[1]
    packed = feat.new_empty((groups, rows, channels // groups * k))
    _pack_neighborhood[(triton.cdiv(rows * (channels // groups) * k, 256), groups)](
        feat, indices, packed, start, rows, channels, groups, k, 256)
    return packed


def _chunk_rows(channels, kernel, dtype):
    # One packed operand is live at a time, including during backward. This
    # budget bounds spatial activation storage independently of slide size.
    itemsize = torch.empty((), dtype=dtype).element_size()
    return max(1, (32 * 1024 * 1024) // max(1, channels * kernel * itemsize))


def _canonical_weight(weight, ic, oc, groups, k, transposed):
    if transposed:
        return weight.view(groups, ic // groups, oc // groups, k).permute(0, 2, 1, 3).reshape(groups, oc // groups, ic // groups * k)
    return weight.view(groups, oc // groups, ic // groups * k)


class _VendorIndexedMatmul(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, feat, weight, indices, reverse, groups, transposed, out_channels):
        m, k = indices.shape
        ic, oc = feat.shape[1], out_channels
        canonical = _canonical_weight(weight, ic, oc, groups, k, transposed)
        out = feat.new_empty((m, oc))
        step = _chunk_rows(ic, k, feat.dtype)
        for start in range(0, m, step):
            rows = min(step, m - start)
            packed = _pack(feat, indices, start, rows, groups)
            if groups == 1:
                torch.mm(packed[0], canonical[0].t(), out=out[start:start + rows])
            else:
                product = torch.bmm(packed, canonical.transpose(1, 2))
                out[start:start + rows].view(rows, groups, oc // groups).copy_(product.permute(1, 0, 2))
                del product
            del packed
        ctx.save_for_backward(feat, weight, indices, reverse)
        ctx.groups, ctx.transposed, ctx.oc = groups, transposed, oc
        return out

    @staticmethod
    @once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        feat, weight, indices, reverse = ctx.saved_tensors
        grad = grad.contiguous()
        groups, ic, oc = ctx.groups, feat.shape[1], ctx.oc
        m, k = indices.shape
        n = len(feat)
        dx = dw = None
        with torch.cuda.device(feat.device):
            canonical = _canonical_weight(weight, ic, oc, groups, k, ctx.transposed)
            if ctx.needs_input_grad[0]:
                dx = torch.empty_like(feat)
                dx_weight = canonical.view(groups, oc // groups, ic // groups, k).permute(0, 1, 3, 2).reshape(groups, oc // groups * k, ic // groups)
                step = _chunk_rows(oc, k, grad.dtype)
                for start in range(0, n, step):
                    rows = min(step, n - start)
                    packed = _pack(grad, reverse, start, rows, groups)
                    if groups == 1:
                        torch.mm(packed[0], dx_weight[0], out=dx[start:start + rows])
                    else:
                        product = torch.bmm(packed, dx_weight)
                        dx[start:start + rows].view(rows, groups, ic // groups).copy_(product.permute(1, 0, 2))
                        del product
                    del packed
            if ctx.needs_input_grad[1]:
                grad_weight = feat.new_zeros((groups, oc // groups, ic // groups * k))
                step = _chunk_rows(ic, k, feat.dtype)
                for start in range(0, m, step):
                    rows = min(step, m - start)
                    packed = _pack(feat, indices, start, rows, groups)
                    dy = grad[start:start + rows].view(rows, groups, oc // groups).permute(1, 2, 0)
                    torch.baddbmm(grad_weight, dy, packed, beta=1, out=grad_weight)
                    del packed
                if ctx.transposed:
                    dw = grad_weight.view(groups, oc // groups, ic // groups, k).permute(0, 2, 1, 3).reshape_as(weight)
                else:
                    dw = grad_weight.reshape_as(weight)
        return dx, dw, None, None, None, None, None
