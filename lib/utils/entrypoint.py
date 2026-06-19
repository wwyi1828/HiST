from __future__ import annotations

from typing import Callable, Iterable, Optional

from omegaconf import DictConfig, OmegaConf

from lib.utils.seed import setup_seeds
from lib.utils.wandb_helper import inject_sweep_params


def print_hydra_config(
    title: str,
    cfg: DictConfig,
    drop_keys: Optional[Iterable[str]] = None,
) -> None:
    printable_cfg = cfg
    if drop_keys:
        cfg_dict = OmegaConf.to_container(cfg, resolve=False)
        if isinstance(cfg_dict, dict):
            for key in drop_keys:
                cfg_dict.pop(key, None)
            printable_cfg = OmegaConf.create(cfg_dict)

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(OmegaConf.to_yaml(printable_cfg))
    print("=" * 60 + "\n")


def initialize_hydra_run(
    cfg: DictConfig,
    title: str,
    *,
    seed: Optional[int] = None,
    mutate_cfg: Optional[Callable[[DictConfig], None]] = None,
    drop_keys: Optional[Iterable[str]] = None,
) -> DictConfig:
    cfg = inject_sweep_params(cfg)
    if mutate_cfg is not None:
        mutate_cfg(cfg)
    if seed is not None:
        setup_seeds(seed)
    print_hydra_config(title=title, cfg=cfg, drop_keys=drop_keys)
    return cfg
