import logging
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from lib.utils.logging import WandBLogger, wandb


def inject_sweep_params(cfg: DictConfig, verbose: bool = True) -> DictConfig:
    if wandb is None or wandb.run is None:
        return cfg

    if verbose:
        print("\n" + "=" * 60)
        print("WandB Sweep Detected - Overriding Parameters")
        print("=" * 60)

    for key, value in wandb.config.items():
        try:
            OmegaConf.update(cfg, key, value, merge=False)
            if verbose:
                print(f"  {key} = {value}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not set {key} = {value}: {e}")

    if verbose:
        print("=" * 60 + "\n")

    return cfg


def _to_container(cfg: DictConfig) -> Dict[str, Any]:
    data = OmegaConf.to_container(cfg, resolve=True)
    data.pop("hydra", None)
    return data


def create_logger(cfg: DictConfig) -> Optional[WandBLogger]:
    if not cfg.logging.wandb.enabled:
        return None
    if wandb is None:
        logging.getLogger(__name__).warning('W&B is enabled but wandb is not installed')
        return None
    return WandBLogger(cfg.logging.wandb, _to_container(cfg))
