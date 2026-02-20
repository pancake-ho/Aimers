from __future__ import annotations

import ast
from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "model" / "main.py"


def _iter_add_argument_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
            yield node


def _extract_quant_method_choices(tree: ast.AST):
    for call in _iter_add_argument_calls(tree):
        arg0 = call.args[0] if call.args else None
        if not isinstance(arg0, ast.Constant) or arg0.value != "--quant_method":
            continue
        for kw in call.keywords:
            if kw.arg != "choices":
                continue
            if isinstance(kw.value, (ast.List, ast.Tuple)):
                values = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        values.append(elt.value)
                return values
    return []


def test_main_cli_has_awq_and_no_legacy_quant_method() -> None:
    source = MAIN_PATH.read_text(encoding="utf-8")
    legacy_token = "".join(["auto", "round"])
    assert legacy_token not in source.lower()

    tree = ast.parse(source)
    choices = _extract_quant_method_choices(tree)
    assert set(choices) == {"gptq", "awq"}

    for flag in [
        "--mix_streaming",
        "--teacher_model_id",
        "--calib_shortfall_policy",
        "--gptq_scheme",
        "--gptq_targets",
        "--gptq_ignore_patterns",
        "--quant_small_grid",
        "--gptq_grid_block_sizes",
        "--gptq_grid_dampening_values",
        "--awq_group_size",
        "--awq_symmetric",
        "--awq_duo_scaling",
        "--awq_offload_device",
        "--awq_targets",
        "--awq_ignore_patterns",
        "--quant_two_stage_eval",
        "--quant_proxy_eval_count",
        "--quant_two_stage_top_k",
        "--rehearsal",
    ]:
        assert flag in source
