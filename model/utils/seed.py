from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    value = int(seed)
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)

    try:
        import numpy as np

        np.random.seed(value)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

