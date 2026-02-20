from __future__ import annotations

import random
from typing import Dict, List, Tuple


def select_calib_indices(
    *,
    requested: int,
    available: int,
    policy: str,
    seed: int,
) -> Tuple[List[int], Dict[str, int | str]]:
    req = max(0, int(requested))
    avail = max(0, int(available))
    pol = str(policy).strip().lower()
    if avail <= 0:
        raise ValueError("available calibration samples must be > 0")
    if pol not in {"downscale", "replacement"}:
        raise ValueError(f"unsupported calib policy: {policy}")

    if pol == "downscale":
        used = min(req, avail)
        indices = list(range(used))
        sampling_method = "without_replacement"
    else:
        if req <= avail:
            used = req
            indices = list(range(used))
            sampling_method = "without_replacement"
        else:
            used = req
            rng = random.Random(int(seed))
            indices = [rng.randrange(avail) for _ in range(req)]
            sampling_method = "with_replacement"

    meta: Dict[str, int | str] = {
        "calib_n_requested": req,
        "calib_n_available": avail,
        "calib_n_used": int(used),
        "calib_policy": pol,
        "sampling_method": sampling_method,
        "seed": int(seed),
    }
    return indices, meta
