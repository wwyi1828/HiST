from __future__ import annotations

import math
from typing import Optional

from src.span.preprocessing import (
    ComposeTransforms,
    RandomDrop,
    RandomMirror,
    RandomRotate,
    RandomShear,
)


def build_coord_aug_transform(cfg) -> Optional[ComposeTransforms]:
    training_cfg = cfg.get("training", {})
    coord_aug_cfg = training_cfg.get("coord_aug")
    if coord_aug_cfg is None:
        return None
    if not coord_aug_cfg.get("enabled", False):
        return None

    transforms = []

    mirror_cfg = coord_aug_cfg.get("mirror", {})
    if mirror_cfg.get("enabled", True):
        transforms.append(RandomMirror(p=mirror_cfg.get("p", 0.75)))

    shear_cfg = coord_aug_cfg.get("shear", {})
    if shear_cfg.get("enabled", True):
        transforms.append(
            RandomShear(
                max_angle=shear_cfg.get("max_angle", 5),
                p=shear_cfg.get("p", 0.75),
            )
        )

    rotate_cfg = coord_aug_cfg.get("rotate", {})
    if rotate_cfg.get("enabled", True):
        transforms.append(
            RandomRotate(
                p=rotate_cfg.get("p", 0.75),
                fixed_angle=rotate_cfg.get("fixed_angle", False),
                max_angle=rotate_cfg.get("max_angle", 45),
            )
        )

    drop_cfg = coord_aug_cfg.get("drop", {})
    if drop_cfg.get("enabled", False):
        transforms.append(RandomDrop(p=drop_cfg.get("p", 0.0)))

    return ComposeTransforms(transforms) if transforms else None


def compute_coord_aug_multiplier(schedule_cfg, epoch: int, total_epochs: int) -> float:
    mode = str(schedule_cfg.get("mode", "cosine")).lower()
    start_factor = float(schedule_cfg.get("start_factor", 1.0))
    end_factor = float(schedule_cfg.get("end_factor", 0.0))
    progress = (epoch - 1) / max(total_epochs, 1)

    if mode == "cosine":
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return end_factor + (start_factor - end_factor) * cosine_decay
    return start_factor - (start_factor - end_factor) * progress


def update_coord_aug_multiplier(cfg, train_data, epoch: int, total_epochs: int) -> None:
    training_cfg = cfg.get("training", {})
    coord_aug_cfg = training_cfg.get("coord_aug", {})
    schedule_cfg = coord_aug_cfg.get("schedule", {})

    if not coord_aug_cfg.get("enabled", False):
        return
    if not schedule_cfg.get("enabled", False):
        return

    target = getattr(train_data, "dataset", train_data)
    if not hasattr(target, "set_aug_multiplier"):
        return

    multiplier = compute_coord_aug_multiplier(schedule_cfg, epoch, total_epochs)
    target.set_aug_multiplier(multiplier)
