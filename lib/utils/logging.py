import os
import json
import fcntl
import logging
from pathlib import Path
from typing import Dict, Any, Union, Callable

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


def _flatten_config(cfg, parent_key='', sep='_'):
    items = {}
    for key, value in cfg.items():
        if key.startswith('_'):
            continue
        name = f'{parent_key}{sep}{key}' if parent_key else key
        if isinstance(value, dict):
            items.update(_flatten_config(value, name, sep))
        else:
            items[name] = value
    return items


class WandBLogger:
    def __init__(self, cfg, training_config):
        self.run = None
        try:
            self.run = wandb.run if wandb.run is not None else wandb.init(
                project=cfg.project,
                entity=cfg.entity,
                config=_flatten_config(training_config),
                tags=list(cfg.tags),
                notes=cfg.notes,
                mode=cfg.mode,
            )
        except Exception as error:
            logger.error(f'Failed to initialize W&B: {error}')

    def log_epoch(self, epoch, metrics):
        if self.run is not None:
            try:
                self.run.log({'epoch': epoch, **metrics})
            except Exception as error:
                logger.error(f'Failed to log epoch {epoch}: {error}')

    def log_final(self, metrics):
        if self.run is not None:
            try:
                self.run.summary.update({
                    f'final/{key}': value for key, value in _flatten_config(metrics, sep='/').items()
                })
            except Exception as error:
                logger.error(f'Failed to log final metrics: {error}')

    def finish(self):
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as error:
                logger.error(f'Error finishing W&B: {error}')


def get_results_path(results_dir: Union[str, Path], filename: str) -> Path:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    return results_path / filename


def save_results(data: Dict[str, Any], results_dir: Union[str, Path], filename: str) -> Path:
    result_path = get_results_path(results_dir, filename)
    tmp_path = result_path.with_name(f"{result_path.name}.tmp.{os.getpid()}")
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, result_path)
    logger.info(f"Results saved to {result_path}")
    return result_path




def update_results_file(
    results_dir: Union[str, Path],
    filename: str,
    updater: Callable[[Dict[str, Any]], None],
) -> Path:
    result_path = get_results_path(results_dir, filename)
    lock_path = result_path.with_suffix(result_path.suffix + ".lock")

    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        if result_path.exists():
            with open(result_path, 'r') as f:
                results = json.load(f)
        else:
            results = {}

        updater(results)
        save_results(results, results_dir, filename)

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return result_path
