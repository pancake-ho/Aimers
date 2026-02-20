from __future__ import annotations

import pytest

from model.pipeline import validate_lm_eval_args


def test_pipeline_requires_tasks_when_lm_eval_enabled() -> None:
    with pytest.raises(ValueError):
        validate_lm_eval_args(run_lm_eval=True, lm_eval_tasks="")


def test_pipeline_allows_lm_eval_with_tasks() -> None:
    validate_lm_eval_args(run_lm_eval=True, lm_eval_tasks="arc_challenge")

