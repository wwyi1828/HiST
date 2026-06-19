from __future__ import annotations

import os
import random

import numpy as np
import torch


def setup_seeds(seed: int, set_hash_seed: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if set_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(seed)
