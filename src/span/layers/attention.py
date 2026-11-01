import torch
from torch import nn


class AttentionBuilder(nn.Module):
    def __init__(self, window_size=1, mode='long', backend='auto'):
        super().__init__()
        self.window_size = window_size
        self.mode = mode
        self.backend = backend
        self.positions = None
        self.sentinel_mask = None

    def forward(self, positions, valid_globals, mode=None, sentinel_mask=None,
                spatial_shape=None):
        from ..ops.attention import build_attention_plans
        self.positions = positions
        self.sentinel_mask = sentinel_mask
        if sentinel_mask is not None:
            self.sentinel_mask = sentinel_mask.to(device=positions.device, dtype=torch.bool)
        return build_attention_plans(
            positions, valid_globals, mode=mode or self.mode,
            window_size=self.window_size, sentinel_mask=self.sentinel_mask,
            spatial_shape=spatial_shape, backend=self.backend,
        )
