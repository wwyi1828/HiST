import torch
from torch import nn
from typing import Optional, Tuple
from einops import rearrange
from timm.layers import trunc_normal_
import math


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')

def _build_inv_freq(dim: int, base: float, device=None, dtype=None) -> torch.Tensor:
    assert dim % 2 == 0, "Rotary dimension must be even"
    idx = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (idx / dim))
    return inv_freq.to(dtype if dtype is not None else torch.float32)

def build_2d_rope_cos_sin(
    ins_pos: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
    device=None,
    dtype=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 4 == 0, "For 2D RoPE, head_dim must be divisible by 4"
    device = ins_pos.device if device is None else device
    dtype = dtype if dtype is not None else torch.get_default_dtype()

    D = head_dim
    Dh = D // 2
    assert Dh % 2 == 0

    h_idx, w_idx = ins_pos.unbind(dim=1)

    inv_freq_h = _build_inv_freq(Dh, base, device=device, dtype=torch.float32)
    freqs_h = torch.outer(h_idx.to(torch.float32), inv_freq_h)
    freqs_h = torch.repeat_interleave(freqs_h, 2, dim=1)

    inv_freq_w = inv_freq_h
    freqs_w = torch.outer(w_idx.to(torch.float32), inv_freq_w)
    freqs_w = torch.repeat_interleave(freqs_w, 2, dim=1)

    freqs = torch.cat([freqs_h, freqs_w], dim=1)
    cos = freqs.cos().to(dtype=dtype).unsqueeze(1)
    sin = freqs.sin().to(dtype=dtype).unsqueeze(1)
    return cos, sin

def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    return (x * cos * scale) + (rotate_half(x) * sin * scale)

def apply_rope_2d_partial(
    x: torch.Tensor,
    ins_pos: torch.Tensor,
    rotary_dim: Optional[int] = None,
    base: float = 10000.0,
    scale: float = 1.0,
) -> torch.Tensor:
    D = x.size(-1)
    if rotary_dim is None:
        rotary_dim = (D // 4) * 4
    if rotary_dim <= 0:
        return x
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos, sin = build_2d_rope_cos_sin(
        ins_pos, head_dim=rotary_dim, base=base, device=x.device, dtype=x.dtype
    )
    x_rot = apply_rotary_pos_emb(x_rot, cos, sin, scale)
    return torch.cat([x_rot, x_pass], dim=-1)


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return _get_interleave_power_of_2(closest_power_of_2) +               _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, size, learned_pos, pos_std):
        super(RelativePositionBias, self).__init__()
        self.size = size
        self.num_heads = num_heads
        self.adjustment_tables = nn.Parameter(torch.zeros(num_heads, 2 * size - 1, 2 * size - 1), requires_grad=learned_pos)
        if learned_pos:
            trunc_normal_(self.adjustment_tables, mean=0.0, std=pos_std)

    def forward(self, x_diff, y_diff):

        x_index = x_diff + (self.size - 1)
        y_index = y_diff + (self.size - 1)

        adjustment_tables = self.adjustment_tables
        adjustments = adjustment_tables[:, x_index, y_index].permute(1, 0).unsqueeze(-1)
        return adjustments

class ALiBiPositionBias(nn.Module):
    def __init__(self, num_heads):
        super(ALiBiPositionBias, self).__init__()
        self.num_heads = num_heads
        slopes = _get_interleave(num_heads)
        self.register_buffer('slopes', torch.tensor(slopes).view(num_heads, 1, 1))

    def forward(self, x_diff, y_diff):
        dist = torch.sqrt((x_diff.float() ** 2 + y_diff.float() ** 2))
        dist = dist.unsqueeze(-1).unsqueeze(-1)
        bias = -self.slopes.transpose(0, 1) * dist
        return bias
