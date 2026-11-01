import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _keys(Real, Coord, Window, ShiftWindow, Members, ShiftMembers, query, r,
          NL, NG: tl.constexpr, SIDE: tl.constexpr,
          LOCAL: tl.constexpr, UNION: tl.constexpr, GLOBAL: tl.constexpr,
          CAP: tl.constexpr):
    if query < NL:
        real_query = tl.load(Real + query).to(tl.int1)
        if LOCAL:
            first = SIDE * SIDE
            wid = tl.load(Window + query)
            ids = tl.load(Members + wid * first + r,
                          real_query & (wid >= 0) & (r < first), other=-1)
            if UNION:
                half = SIDE // 2
                area = half * half
                y = tl.load(Coord + query * 2)
                x = tl.load(Coord + query * 2 + 1)
                shared = ((y + half) % SIDE // half) * 2 + (x + half) % SIDE // half
                t = r - first
                quad = t // area
                quad += (quad >= shared).to(tl.int32)
                within = t % area
                slot = (quad // 2 * half + within // half) * SIDE + quad % 2 * half + within % half
                swid = tl.load(ShiftWindow + query)
                other = tl.load(ShiftMembers + swid * first + slot,
                                real_query & (swid >= 0) & (t >= 0) & (t < 3 * area), other=-1)
                ids = tl.where(r >= first, other, ids)
        else:
            ids = tl.full(r.shape, -1, tl.int32)
        if GLOBAL:
            ids = tl.where((r >= CAP) & (r < CAP + NG), NL + r - CAP, ids)
        valid = real_query & (ids >= 0)
    else:
        ids = r
        local_real = tl.load(Real + r, r < NL, other=0).to(tl.int1)
        valid = (r < NL + NG) & ((r >= NL) | local_real)
        if not GLOBAL:
            valid = tl.full(r.shape, False, tl.int1)
    return ids, valid


@triton.jit
def _drop(Seed, query, head, slots, H: tl.constexpr, P: tl.constexpr):
    if P == 0.:
        return tl.full(slots.shape, 1., tl.float32)
    seed = tl.load(Seed).to(tl.uint32)
    x = ((query * H + head).to(tl.uint32) * 0x9e3779b9) ^ slots.to(tl.uint32) ^ seed
    x = (x ^ (x >> 16)) * 0x7feb352d
    x = (x ^ (x >> 15)) * 0x846ca68b
    x = x ^ (x >> 16)
    uniform = (x >> 8).to(tl.float32) * (1. / 16777216.)
    return (uniform >= P).to(tl.float32) / (1. - P)


@triton.jit
def _scores(K, Bias, Slopes, Coord, qvec, query, head, ids, valid, d,
            NL, H: tl.constexpr, D: tl.constexpr,
            SIDE: tl.constexpr, RPB: tl.constexpr, ALIBI: tl.constexpr,
            REVERSE: tl.constexpr = False):
    kv = tl.load(K + (ids[:, None] * H + head) * D + d[None, :],
                 valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
    scores = tl.sum(kv * qvec[None, :], 1) * (D ** -.5)
    ll = valid & (ids < NL) & (query < NL)
    bi = tl.full(ids.shape, 0, tl.int64)
    distance = tl.full(ids.shape, 0., tl.float32)
    if RPB or ALIBI:
        qy = tl.load(Coord + query * 2, query < NL, other=0)
        qx = tl.load(Coord + query * 2 + 1, query < NL, other=0)
        ky = tl.load(Coord + ids * 2, ll, other=0)
        kx = tl.load(Coord + ids * 2 + 1, ll, other=0)
        dy, dx = ky - qy, kx - qx
        if REVERSE:
            dy, dx = -dy, -dx
        if RPB:
            width = 2 * SIDE - 1
            bi = head * width * width + (dy + SIDE - 1) * width + dx + SIDE - 1
            scores += tl.load(Bias + bi, ll, other=0).to(tl.float32)
        if ALIBI:
            distance = tl.sqrt(dy.to(tl.float32) * dy.to(tl.float32) + dx.to(tl.float32) * dx.to(tl.float32))
            slope = tl.load(Slopes + head).to(tl.float32)
            scores -= tl.where(ll, slope * distance, 0.)
    return tl.where(valid, scores, -float('inf')), kv, bi, distance, ll


@triton.jit(do_not_specialize=['NL'])
def _forward(Q, K, V, Bias, Slopes, Seed, Real, Coord, Window, ShiftWindow,
             Members, ShiftMembers, Out, Saved, LSE,
             NL, NG: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
             SIDE: tl.constexpr, LOCAL: tl.constexpr, UNION: tl.constexpr,
             GLOBAL: tl.constexpr, CAP: tl.constexpr, RPB: tl.constexpr,
             ALIBI: tl.constexpr, P: tl.constexpr, SHIFT: tl.constexpr, KEEP: tl.constexpr,
             BK: tl.constexpr, BD: tl.constexpr):
    query, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    qvec = tl.load(Q + (query * H + head) * D + d, d < D, other=0).to(tl.float32)
    length = tl.where(query < NL, CAP + (NG if GLOBAL else 0), NL + NG if GLOBAL else 0)
    maximum = tl.full((), -float('inf'), tl.float32)
    denominator = tl.full((), 0., tl.float32)
    numerator = tl.full((BD,), 0., tl.float32)
    for begin in range(0, length, BK):
        r = begin + tl.arange(0, BK)
        ids, valid = _keys(Real, Coord, Window, ShiftWindow, Members, ShiftMembers,
                           query, r, NL, NG, SIDE, LOCAL, UNION, GLOBAL, CAP)
        scores, _, _, _, _ = _scores(K, Bias, Slopes, Coord, qvec, query, head,
                                     ids, valid, d, NL, H, D, SIDE, RPB, ALIBI)
        block_max = tl.max(scores, 0)
        updated = tl.maximum(maximum, block_max)
        safe_max = tl.where(updated > -float('inf'), updated, 0.)
        correction = tl.exp(maximum - safe_max)
        prob = tl.exp(scores - safe_max)
        denominator = denominator * correction + tl.sum(prob, 0)
        vv = tl.load(V + (ids[:, None] * H + head) * D + d[None, :],
                     valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
        drop = _drop(Seed, query, head, r, H, P)
        numerator = numerator * correction + tl.sum((prob * drop)[:, None] * vv, 0)
        maximum = updated
    result = numerator / tl.where(denominator > 0, denominator, 1.)
    tl.store(Out + (query * H + head) * D + d, result, d < D)
    if KEEP:
        tl.store(Saved + (query * H + head) * D + d, result, d < D)
        lse = tl.where(denominator > 0, maximum + tl.log(denominator), 0.)
        tl.store(LSE + query * H + head, lse)


@triton.jit(do_not_specialize=['NL'])
def _backward_q(Q, K, V, Bias, Slopes, Seed, Real, Coord, Window, ShiftWindow,
              Members, ShiftMembers, Saved, LSE, DO, DQ, DBias, DSlopes,
              NL, NG: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              SIDE: tl.constexpr, LOCAL: tl.constexpr, UNION: tl.constexpr,
              GLOBAL: tl.constexpr, CAP: tl.constexpr, RPB: tl.constexpr,
              ALIBI: tl.constexpr, P: tl.constexpr, SHIFT: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    query, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    qvec = tl.load(Q + (query * H + head) * D + d, d < D, other=0).to(tl.float32)
    dout = tl.load(DO + (query * H + head) * D + d, d < D, other=0).to(tl.float32)
    out = tl.load(Saved + (query * H + head) * D + d, d < D, other=0)
    delta = tl.sum(dout * out, 0)
    lse = tl.load(LSE + query * H + head)
    length = tl.where(query < NL, CAP + (NG if GLOBAL else 0), NL + NG if GLOBAL else 0)
    dq = tl.full((BD,), 0., tl.float32)
    dslope = tl.full((), 0., tl.float32)
    for begin in range(0, length, BK):
        r = begin + tl.arange(0, BK)
        ids, valid = _keys(Real, Coord, Window, ShiftWindow, Members, ShiftMembers,
                           query, r, NL, NG, SIDE, LOCAL, UNION, GLOBAL, CAP)
        scores, kv, bi, dist, ll = _scores(K, Bias, Slopes, Coord, qvec, query, head,
                                          ids, valid, d, NL, H, D, SIDE, RPB, ALIBI)
        prob = tl.exp(scores - lse)
        vv = tl.load(V + (ids[:, None] * H + head) * D + d[None, :],
                     valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
        drop = _drop(Seed, query, head, r, H, P)
        dp = tl.sum(vv * dout[None, :], 1) * drop
        ds = prob * (dp - delta)
        dq += tl.sum(ds[:, None] * kv, 0) * (D ** -.5)
        if RPB:
            tl.atomic_add(DBias + bi, ds, ll, sem='relaxed')
        if ALIBI:
            dslope -= tl.sum(tl.where(ll, ds * dist, 0.), 0)
    tl.store(DQ + (query * H + head) * D + d, dq, d < D)
    if ALIBI:
        tl.atomic_add(DSlopes + head, dslope, sem='relaxed')


@triton.jit
def _reverse_slot(Coord, Window, key, queries, valid, NL,
                  SIDE: tl.constexpr, SHIFT: tl.constexpr,
                  LOCAL: tl.constexpr, UNION: tl.constexpr, CAP: tl.constexpr):
    # Attention adjacency is symmetric, but dropout counters are query-oriented.
    # Reconstruct each source query's original forward slot without a reverse CSR.
    slot = tl.full(queries.shape, key, tl.int64)
    if key >= NL:
        slot = tl.where(queries < NL, CAP + key - NL, slot)
    elif LOCAL:
        ky = tl.load(Coord + key * 2)
        kx = tl.load(Coord + key * 2 + 1)
        local_slot = ((ky + SHIFT) % SIDE) * SIDE + (kx + SHIFT) % SIDE
        if UNION:
            kwid = tl.load(Window + key)
            qwid = tl.load(Window + queries, valid & (queries < NL), other=-1)
            qy = tl.load(Coord + queries * 2, valid & (queries < NL), other=0)
            qx = tl.load(Coord + queries * 2 + 1, valid & (queries < NL), other=0)
            half = SIDE // 2
            area = half * half
            shared = ((qy + half) % SIDE // half) * 2 + (qx + half) % SIDE // half
            sy, sx = (ky + half) % SIDE, (kx + half) % SIDE
            quadrant = sy // half * 2 + sx // half
            group = quadrant - (quadrant > shared).to(tl.int32)
            alternate = SIDE * SIDE + group * area + sy % half * half + sx % half
            local_slot = tl.where(qwid == kwid, local_slot, alternate)
        slot = tl.where(queries < NL, local_slot, slot)
    return slot


@triton.jit(do_not_specialize=['NL'])
def _backward_kv(Q, K, V, Bias, Slopes, Seed, Real, Coord, Window, ShiftWindow,
                 Members, ShiftMembers, Saved, LSE, DO, DK, DV,
                 NL, NG: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 SIDE: tl.constexpr, LOCAL: tl.constexpr, UNION: tl.constexpr,
                 GLOBAL: tl.constexpr, CAP: tl.constexpr, RPB: tl.constexpr,
                 ALIBI: tl.constexpr, P: tl.constexpr, SHIFT: tl.constexpr,
                 BK: tl.constexpr, BD: tl.constexpr):
    key, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    kvec = tl.load(K + (key * H + head) * D + d, d < D, other=0).to(tl.float32)
    vvec = tl.load(V + (key * H + head) * D + d, d < D, other=0).to(tl.float32)
    length = tl.where(key < NL, CAP + (NG if GLOBAL else 0), NL + NG if GLOBAL else 0)
    dk = tl.full((BD,), 0., tl.float32)
    dv = tl.full((BD,), 0., tl.float32)
    for begin in range(0, length, BK):
        r = begin + tl.arange(0, BK)
        queries, valid = _keys(Real, Coord, Window, ShiftWindow, Members, ShiftMembers,
                               key, r, NL, NG, SIDE, LOCAL, UNION, GLOBAL, CAP)
        scores, qv, _, _, _ = _scores(Q, Bias, Slopes, Coord, kvec, key, head,
                                     queries, valid, d, NL, H, D, SIDE, RPB, ALIBI, True)
        lse = tl.load(LSE + queries * H + head, valid, other=0)
        prob = tl.exp(scores - lse)
        mask = valid[:, None] & (d[None, :] < D)
        dout = tl.load(DO + (queries[:, None] * H + head) * D + d[None, :], mask, other=0).to(tl.float32)
        out = tl.load(Saved + (queries[:, None] * H + head) * D + d[None, :], mask, other=0)
        delta = tl.sum(dout * out, 1)
        if P != 0.:
            slot = _reverse_slot(Coord, Window, key, queries, valid, NL, SIDE, SHIFT, LOCAL, UNION, CAP)
            drop = _drop(Seed, queries, head, slot, H, P)
        else:
            drop = tl.full(r.shape, 1., tl.float32)
        dp = tl.sum(dout * vvec[None, :], 1) * drop
        ds = prob * (dp - delta)
        dk += tl.sum(ds[:, None] * qv, 0) * (D ** -.5)
        dv += tl.sum((prob * drop)[:, None] * dout, 0)
    tl.store(DK + (key * H + head) * D + d, dk, d < D)
    tl.store(DV + (key * H + head) * D + d, dv, d < D)


@triton.jit(do_not_specialize=['NL'])
def _forward_edges(Q, K, V, Bias, Slopes, Seed, Coord, Ptr, Idx, Slot, Out, Saved, LSE,
                   NL, H: tl.constexpr, D: tl.constexpr, SIDE: tl.constexpr,
                   RPB: tl.constexpr, ALIBI: tl.constexpr, P: tl.constexpr,
                   KEEP: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    query, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    start, end = tl.load(Ptr + query).to(tl.int32), tl.load(Ptr + query + 1).to(tl.int32)
    qvec = tl.load(Q + (query * H + head) * D + d, d < D, other=0).to(tl.float32)
    maximum = tl.full((), -float('inf'), tl.float32)
    denominator = tl.full((), 0., tl.float32)
    numerator = tl.full((BD,), 0., tl.float32)
    pos = start
    while pos < end:
        r = pos + tl.arange(0, BK)
        valid = r < end
        ids = tl.load(Idx + r, valid, other=-1).to(tl.int32)
        scores, _, _, _, _ = _scores(K, Bias, Slopes, Coord, qvec, query, head, ids, valid,
                                     d, NL, H, D, SIDE, RPB, ALIBI)
        block_max = tl.max(scores, 0)
        updated = tl.maximum(maximum, block_max)
        safe_max = tl.where(updated > -float('inf'), updated, 0.)
        correction = tl.exp(maximum - safe_max)
        prob = tl.exp(scores - safe_max)
        denominator = denominator * correction + tl.sum(tl.where(valid, prob, 0.), 0)
        vv = tl.load(V + (ids[:, None] * H + head) * D + d[None, :],
                     valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
        slots = tl.load(Slot + r, valid, other=0)
        drop = _drop(Seed, query, head, slots, H, P)
        numerator = numerator * correction + tl.sum((tl.where(valid, prob, 0.) * drop)[:, None] * vv, 0)
        maximum = updated
        pos += BK
    result = numerator / tl.where(denominator > 0, denominator, 1.)
    tl.store(Out + (query * H + head) * D + d, result, d < D)
    if KEEP:
        tl.store(Saved + (query * H + head) * D + d, result, d < D)
        lse = tl.where(denominator > 0, maximum + tl.log(denominator), 0.)
        tl.store(LSE + query * H + head, lse)


@triton.jit(do_not_specialize=['NL'])
def _backward_q_edges(Q, K, V, Bias, Slopes, Seed, Coord, Ptr, Idx, Slot, LSE, DO, Delta,
                      DQ, DBias, DSlopes, NL, H: tl.constexpr, D: tl.constexpr,
                      SIDE: tl.constexpr, RPB: tl.constexpr, ALIBI: tl.constexpr,
                      P: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    query, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    start, end = tl.load(Ptr + query).to(tl.int32), tl.load(Ptr + query + 1).to(tl.int32)
    qvec = tl.load(Q + (query * H + head) * D + d, d < D, other=0).to(tl.float32)
    dout = tl.load(DO + (query * H + head) * D + d, d < D, other=0).to(tl.float32)
    delta = tl.load(Delta + query * H + head)
    lse = tl.load(LSE + query * H + head)
    dq = tl.full((BD,), 0., tl.float32)
    dslope = tl.full((), 0., tl.float32)
    pos = start
    while pos < end:
        r = pos + tl.arange(0, BK)
        valid = r < end
        ids = tl.load(Idx + r, valid, other=-1).to(tl.int32)
        scores, kv, bi, dist, ll = _scores(K, Bias, Slopes, Coord, qvec, query, head, ids, valid,
                                          d, NL, H, D, SIDE, RPB, ALIBI)
        prob = tl.exp(scores - lse)
        vv = tl.load(V + (ids[:, None] * H + head) * D + d[None, :],
                     valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
        slots = tl.load(Slot + r, valid, other=0)
        drop = _drop(Seed, query, head, slots, H, P)
        dp = tl.sum(vv * dout[None, :], 1) * drop
        ds = tl.where(valid, prob * (dp - delta), 0.)
        dq += tl.sum(ds[:, None] * kv, 0) * (D ** -.5)
        if RPB:
            tl.atomic_add(DBias + bi, ds, ll, sem='relaxed')
        if ALIBI:
            dslope -= tl.sum(tl.where(ll, ds * dist, 0.), 0)
        pos += BK
    tl.store(DQ + (query * H + head) * D + d, dq, d < D)
    if ALIBI:
        tl.atomic_add(DSlopes + head, dslope, sem='relaxed')


@triton.jit(do_not_specialize=['NL'])
def _backward_kv_edges(Q, K, V, Bias, Slopes, Seed, Coord, Ptr, Idx, Slot, LSE, DO, Delta,
                       DK, DV, NL, H: tl.constexpr, D: tl.constexpr, SIDE: tl.constexpr,
                       RPB: tl.constexpr, ALIBI: tl.constexpr, P: tl.constexpr,
                       BK: tl.constexpr, BD: tl.constexpr):
    key, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    start, end = tl.load(Ptr + key).to(tl.int32), tl.load(Ptr + key + 1).to(tl.int32)
    kvec = tl.load(K + (key * H + head) * D + d, d < D, other=0).to(tl.float32)
    vvec = tl.load(V + (key * H + head) * D + d, d < D, other=0).to(tl.float32)
    dk = tl.full((BD,), 0., tl.float32)
    dv = tl.full((BD,), 0., tl.float32)
    pos = start
    while pos < end:
        r = pos + tl.arange(0, BK)
        valid = r < end
        queries = tl.load(Idx + r, valid, other=-1).to(tl.int32)
        scores, qv, _, _, _ = _scores(Q, Bias, Slopes, Coord, kvec, key, head, queries, valid,
                                     d, NL, H, D, SIDE, RPB, ALIBI, True)
        lse = tl.load(LSE + queries * H + head, valid, other=0.)
        prob = tl.where(valid, tl.exp(scores - lse), 0.)
        mask = valid[:, None] & (d[None, :] < D)
        dout = tl.load(DO + (queries[:, None] * H + head) * D + d[None, :], mask, other=0).to(tl.float32)
        delta = tl.load(Delta + queries * H + head, valid, other=0.)
        slots = tl.load(Slot + r, valid, other=0)
        drop = _drop(Seed, queries, head, slots, H, P)
        dp = tl.sum(dout * vvec[None, :], 1) * drop
        ds = prob * (dp - delta)
        dk += tl.sum(ds[:, None] * qv, 0) * (D ** -.5)
        dv += tl.sum((prob * drop)[:, None] * dout, 0)
        pos += BK
    tl.store(DK + (key * H + head) * D + d, dk, d < D)
    tl.store(DV + (key * H + head) * D + d, dv, d < D)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, slopes, seed, plan, p, keep):
        shape = q.shape
        out = torch.empty_like(q)
        saved = (out if q.dtype == torch.float32 else torch.empty(shape, dtype=torch.float32, device=q.device)) if keep else None
        lse = torch.empty(shape[:2], dtype=torch.float32, device=q.device) if keep else None
        w = plan.windows
        topology = (plan.real, plan.positions,
                    w.window if w is not None else plan.real,
                    w.shifted_window if w is not None else plan.real,
                    w.members if w is not None else plan.real,
                    w.shifted_members if w is not None else plan.real)
        constants = (plan.num_local, plan.num_globals, shape[1], shape[2],
                     w.side if w is not None else ((bias.shape[-1] + 1) // 2 if bias is not None else 2),
                     w is not None, w.union if w is not None else False,
                     plan.global_edges, plan.local_capacity, bias is not None,
                     slopes is not None, p, w.shift if w is not None else 0)
        bk = min(128, triton.next_power_of_2(max(plan.local_capacity + plan.num_globals, 32)))
        bd = triton.next_power_of_2(shape[2])
        edges = None
        if plan.windows is not None:
            from .attention import cached_local_edges
            edges = cached_local_edges(plan)
            bk = 32
        if len(q):
            with torch.cuda.device(q.device):
                if edges is None:
                    _forward[(shape[0], shape[1])](
                        q, k, v, bias, slopes, seed, *topology, out, saved, lse,
                        *constants, keep, bk, bd, num_warps=4, enable_fp_fusion=False)
                else:
                    _forward_edges[(shape[0], shape[1])](
                        q, k, v, bias, slopes, seed, plan.positions, *edges[:3], out, saved, lse,
                        plan.num_local, shape[1], shape[2], constants[4],
                        bias is not None, slopes is not None, p, keep, bk, bd,
                        num_warps=2, enable_fp_fusion=False)
        saved_edges = edges if edges is not None else ()
        ctx.save_for_backward(q, k, v, bias, slopes, seed, *topology, *saved_edges, saved, lse)
        ctx.constants, ctx.bk, ctx.bd = constants, bk, bd
        ctx.use_edges = edges is not None
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        q, k, v, bias, slopes, seed, *rest = ctx.saved_tensors
        if ctx.use_edges:
            topology, edges, saved, lse = rest[:6], rest[6:12], rest[12], rest[13]
        else:
            topology, edges, saved, lse = rest[:6], None, rest[6], rest[7]
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.empty_like(k, dtype=torch.float32)
        dv = torch.empty_like(v, dtype=torch.float32)
        db = torch.zeros_like(bias, dtype=torch.float32) if bias is not None else None
        ds = torch.zeros_like(slopes, dtype=torch.float32) if slopes is not None else None
        grad = grad.contiguous()
        if len(q):
            with torch.cuda.device(q.device):
                if edges is None:
                    _backward_q[(q.shape[0], q.shape[1])](
                        q, k, v, bias, slopes, seed, *topology, saved, lse, grad,
                        dq, db, ds, *ctx.constants, ctx.bk, ctx.bd,
                        num_warps=4, enable_fp_fusion=False)
                    _backward_kv[(q.shape[0], q.shape[1])](
                        q, k, v, bias, slopes, seed, *topology, saved, lse, grad,
                        dk, dv, *ctx.constants, ctx.bk, ctx.bd,
                        num_warps=4, enable_fp_fusion=False)
                else:
                    delta = torch.sum(grad.float() * saved, -1).contiguous()
                    coord = topology[1]
                    _backward_q_edges[(q.shape[0], q.shape[1])](
                        q, k, v, bias, slopes, seed, coord, *edges[:3], lse, grad, delta,
                        dq, db, ds, ctx.constants[0], q.shape[1], q.shape[2], ctx.constants[4],
                        bias is not None, slopes is not None, ctx.constants[11], ctx.bk, ctx.bd,
                        num_warps=2, enable_fp_fusion=False)
                    _backward_kv_edges[(q.shape[0], q.shape[1])](
                        q, k, v, bias, slopes, seed, coord, *edges[3:], lse, grad, delta,
                        dk, dv, ctx.constants[0], q.shape[1], q.shape[2], ctx.constants[4],
                        bias is not None, slopes is not None, ctx.constants[11], ctx.bk, ctx.bd,
                        num_warps=2, enable_fp_fusion=False)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                db.to(bias.dtype) if db is not None else None,
                ds.to(slopes.dtype) if ds is not None else None, None, None, None, None)


def fused_attention(q, k, v, plan, bias, slopes, p, seed):
    tensors = (q, k, v)
    # Turing lacks native bfloat16 arithmetic. Conversion kernels preserve BF16
    # rounding while this operator's accumulation stays FP32 on every GPU.
    software_bf16 = q.dtype == torch.bfloat16 and torch.cuda.get_device_capability(q.device)[0] < 8
    if software_bf16:
        tensors = tuple(x.float() for x in tensors)
        bias = bias.float() if bias is not None else None
        slopes = slopes.float() if slopes is not None else None
    out = _Attention.apply(*(x.contiguous() for x in tensors),
                            bias.contiguous() if bias is not None else None,
                            slopes.contiguous().reshape(-1) if slopes is not None else None,
                            seed.contiguous(), plan, p,
                            torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v) + ((bias,) if bias is not None else ()) + ((slopes,) if slopes is not None else ())))
    return out.to(q.dtype)
