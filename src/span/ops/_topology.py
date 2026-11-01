import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N', 'WIDTH', 'SHIFT_WIDTH'])
def _window_tables(Coord, Sentinel, Window, ShiftWindow, Members, ShiftMembers,
                   N, SIDE: tl.constexpr, WIDTH,
                   SHIFT_WIDTH, UNION: tl.constexpr,
                   SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    real = (i < N) & ~tl.load(Sentinel + i, i < N, other=1).to(tl.int1)
    y = tl.load(Coord + i * 2, i < N, other=0)
    x = tl.load(Coord + i * 2 + 1, i < N, other=0)
    wid = (y + SHIFT) // SIDE * WIDTH + (x + SHIFT) // SIDE
    slot = (y + SHIFT) % SIDE * SIDE + (x + SHIFT) % SIDE
    tl.store(Window + i, tl.where(real, wid, -1), i < N)
    tl.store(Members + wid * SIDE * SIDE + slot, i, real)
    if UNION:
        shift = SIDE // 2
        swid = (y + shift) // SIDE * SHIFT_WIDTH + (x + shift) // SIDE
        sslot = (y + shift) % SIDE * SIDE + (x + shift) % SIDE
        tl.store(ShiftWindow + i, tl.where(real, swid, -1), i < N)
        tl.store(ShiftMembers + swid * SIDE * SIDE + sslot, i, real)


def window_tables(coord, spatial_shape, sentinel_mask, side, union, shift):
    from .window_attention import WindowPlan
    h, w = int(spatial_shape[0]), int(spatial_shape[1])
    coord = coord.contiguous()
    sentinel_mask = sentinel_mask.contiguous()
    with torch.cuda.device(coord.device):
        width = triton.cdiv(w + shift, side)
        window = torch.empty(len(coord), dtype=torch.int32, device=coord.device)
        members = torch.full((triton.cdiv(h + shift, side) * width, side * side),
                             -1, dtype=torch.int32, device=coord.device)
        if union:
            swidth = triton.cdiv(w + side // 2, side)
            swindow = torch.empty_like(window)
            smembers = torch.full((triton.cdiv(h + side // 2, side) * swidth, side * side),
                                  -1, dtype=torch.int32, device=coord.device)
        else:
            swidth, swindow, smembers = width, window, members
        if len(coord):
            _window_tables[(triton.cdiv(len(coord), 256),)](
                coord, sentinel_mask, window, swindow, members, smembers,
                len(coord), side, width, swidth, union, shift, 256)
    return WindowPlan(coord, window, swindow, members, smembers, side, union, shift)


@triton.jit(do_not_specialize=['N', 'H', 'W'])
def _scatter_children(Coord, Sentinel, Dense, Occupied, N,
                      H, W, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(Coord + i * 2, i < N, other=-1)
    x = tl.load(Coord + i * 2 + 1, i < N, other=-1)
    valid = (i < N) & (y >= 0) & (x >= 0) & (y // 2 < H) & (x // 2 < W)
    p, slot = y // 2 * W + x // 2, y % 2 * 2 + x % 2
    sentinel = tl.load(Sentinel + i, i < N, other=1)
    tl.store(Dense + p * 4 + slot, tl.where(sentinel, -1, i), valid)
    tl.atomic_or(Occupied + p, 1, valid, sem='relaxed')


@triton.jit(do_not_specialize=['P', 'W'])
def _compact_children(Dense, Occupied, Flat, Children, Coord, Sentinel,
                      P, W, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    flat = tl.load(Flat + p, p < P, other=0)
    any_real = tl.full((BLOCK,), False, tl.int1)
    for k in tl.static_range(4):
        child = tl.load(Dense + flat * 4 + k, p < P, other=-1)
        tl.store(Children + p * 4 + k, child, p < P)
        any_real |= child >= 0
    tl.store(Sentinel + p, ~any_real, p < P)
    tl.store(Coord + p * 2, flat // W, p < P)
    tl.store(Coord + p * 2 + 1, flat % W, p < P)
    # Reuse the occupancy bitmap as a compact parent lookup after nonzero.
    tl.store(Occupied + flat, p + 1, p < P)


@triton.jit(do_not_specialize=['N', 'H', 'W'])
def _reverse_children(Coord, Sentinel, Occupied, Parent, Slot,
                      N, H, W, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(Coord + i * 2, i < N, other=-1)
    x = tl.load(Coord + i * 2 + 1, i < N, other=-1)
    sentinel = tl.load(Sentinel + i, i < N, other=1)
    valid = (i < N) & (y >= 0) & (x >= 0) & (y // 2 < H) & (x // 2 < W) & ~sentinel.to(tl.int1)
    p = tl.load(Occupied + y // 2 * W + x // 2, valid, other=0) - 1
    tl.store(Parent + i, p, i < N)
    tl.store(Slot + i, y % 2 * 2 + x % 2, i < N)


def downsample_tables(coord, spatial_shape, sentinel_mask):
    from .spatial import DownsamplePlan
    h, w = int(spatial_shape[0]) // 2, int(spatial_shape[1]) // 2
    coord, sentinel_mask = coord.contiguous(), sentinel_mask.contiguous()
    with torch.cuda.device(coord.device):
        dense = torch.full((h * w, 4), -1, dtype=torch.int32, device=coord.device)
        occupied = torch.zeros(h * w, dtype=torch.int32, device=coord.device)
        if len(coord):
            _scatter_children[(triton.cdiv(len(coord), 256),)](
                coord, sentinel_mask, dense, occupied, len(coord), h, w, 256)
        flat = occupied.nonzero().flatten()
        p = len(flat)
        children = torch.empty((p, 4), dtype=torch.int32, device=coord.device)
        out_coord = torch.empty((p, 2), dtype=torch.int64, device=coord.device)
        sentinel = torch.empty(p, dtype=torch.bool, device=coord.device)
        parent = torch.empty(len(coord), dtype=torch.int32, device=coord.device)
        slot = torch.empty_like(parent)
        if p:
            _compact_children[(triton.cdiv(p, 256),)](
                dense, occupied, flat, children, out_coord, sentinel, p, w, 256)
        if len(coord):
            _reverse_children[(triton.cdiv(len(coord), 256),)](
                coord, sentinel_mask, occupied, parent, slot, len(coord), h, w, 256)
    return DownsamplePlan(out_coord, (h, w), children, parent, slot, sentinel)
