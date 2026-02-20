from __future__ import annotations

import itertools
from typing import Any, Dict, List


def parse_int_csv(csv_text: str) -> List[int]:
    values = [int(chunk.strip()) for chunk in csv_text.split(",") if chunk.strip()]
    values = sorted(set(values))
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_float_csv(csv_text: str) -> List[float]:
    values = [float(chunk.strip()) for chunk in csv_text.split(",") if chunk.strip()]
    values = sorted(set(values))
    if not values:
        raise ValueError("expected at least one float value")
    return values


def build_deterministic_gptq_trials(
    group_sizes_csv: str,
    dampening_csv: str,
    block_size: int,
    calib_samples: int,
    calib_seq_len: int,
    search_budget: int,
    disable_grid: bool = False,
) -> List[Dict[str, Any]]:
    groups = parse_int_csv(group_sizes_csv)
    dampening_values = parse_float_csv(dampening_csv)
    if disable_grid:
        groups = groups[:1]
        dampening_values = dampening_values[:1]

    trials = []
    for group_size, dampening in itertools.product(groups, dampening_values):
        trials.append(
            {
                "group_size": int(group_size),
                "dampening": float(dampening),
                "block_size": int(block_size),
                "calib_samples": int(calib_samples),
                "calib_seq_len": int(calib_seq_len),
            }
        )
    trials.sort(key=lambda row: (row["group_size"], row["dampening"]))
    limit = min(6, max(1, int(search_budget)))
    return trials[:limit]
