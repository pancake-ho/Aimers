from __future__ import annotations

import importlib
import inspect
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llmcompressor import oneshot


def _resolve_awq_modifier():
    candidates = [
        ("llmcompressor.modifiers.quantization", "AWQModifier"),
        ("llmcompressor.modifiers.awq", "AWQModifier"),
    ]
    last_error: Optional[Exception] = None
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            modifier_cls = getattr(module, class_name)
            return modifier_cls
        except Exception as exc:
            last_error = exc
    raise ImportError(f"failed to import AWQModifier from llmcompressor: {last_error}")


def _static_mapping_pairs() -> List[Tuple[List[str], List[str]]]:
    return [
        (
            ["re:.*(input_layernorm|self_attn_layer_norm|ln_1)$"],
            [
                "re:.*self_attn\\.q_proj$",
                "re:.*self_attn\\.k_proj$",
                "re:.*self_attn\\.v_proj$",
            ],
        ),
        (
            ["re:.*self_attn\\.v_proj$"],
            ["re:.*self_attn\\.o_proj$"],
        ),
        (
            ["re:.*(post_attention_layernorm|post_attn_layer_norm|ln_2)$"],
            [
                "re:.*mlp\\.gate_proj$",
                "re:.*mlp\\.up_proj$",
            ],
        ),
        (
            ["re:.*mlp\\.up_proj$"],
            ["re:.*mlp\\.down_proj$"],
        ),
    ]


def _mapping_supported_by_model(mapping_pairs: Sequence[Tuple[List[str], List[str]]], module_names: Sequence[str]) -> bool:
    for input_patterns, output_patterns in mapping_pairs:
        has_input = any(re.search(pattern.removeprefix("re:"), name) for pattern in input_patterns for name in module_names)
        has_output = any(re.search(pattern.removeprefix("re:"), name) for pattern in output_patterns for name in module_names)
        if not has_input or not has_output:
            return False
    return True


def _pick_existing(candidates: Sequence[str], module_set: set) -> Optional[str]:
    for name in candidates:
        if name in module_set:
            return name
    return None


def _as_exact_regex(names: Sequence[str]) -> List[str]:
    return [f"re:^{re.escape(name)}$" for name in names]


def _infer_mapping_pairs(module_names: Sequence[str]) -> Tuple[List[Tuple[List[str], List[str]]], Dict[str, Any]]:
    module_set = set(module_names)
    q_proj_suffix = ".self_attn.q_proj"
    attn_prefixes = [name[: -len(q_proj_suffix)] + ".self_attn" for name in module_names if name.endswith(q_proj_suffix)]
    if not attn_prefixes:
        raise RuntimeError("could not find attention q_proj modules for AWQ mapping")

    input_norms: List[str] = []
    q_names: List[str] = []
    k_names: List[str] = []
    v_names: List[str] = []
    o_names: List[str] = []
    post_norms: List[str] = []
    gate_names: List[str] = []
    up_names: List[str] = []
    down_names: List[str] = []

    for attn_prefix in sorted(set(attn_prefixes)):
        block_prefix = attn_prefix[: -len(".self_attn")]

        q_name = f"{attn_prefix}.q_proj"
        k_name = f"{attn_prefix}.k_proj"
        v_name = f"{attn_prefix}.v_proj"
        o_name = _pick_existing(
            [f"{attn_prefix}.o_proj", f"{attn_prefix}.out_proj"],
            module_set,
        )
        in_norm = _pick_existing(
            [
                f"{block_prefix}.input_layernorm",
                f"{block_prefix}.self_attn_layer_norm",
                f"{block_prefix}.ln_1",
            ],
            module_set,
        )
        post_norm = _pick_existing(
            [
                f"{block_prefix}.post_attention_layernorm",
                f"{block_prefix}.post_attn_layer_norm",
                f"{block_prefix}.ln_2",
            ],
            module_set,
        )
        gate_name = _pick_existing(
            [f"{block_prefix}.mlp.gate_proj"],
            module_set,
        )
        up_name = _pick_existing(
            [f"{block_prefix}.mlp.up_proj"],
            module_set,
        )
        down_name = _pick_existing(
            [f"{block_prefix}.mlp.down_proj"],
            module_set,
        )

        required = [q_name, k_name, v_name, o_name, in_norm, post_norm, gate_name, up_name, down_name]
        if any(item is None for item in required):
            continue

        q_names.append(q_name)
        k_names.append(k_name)
        v_names.append(v_name)
        o_names.append(o_name)
        input_norms.append(in_norm)
        post_norms.append(post_norm)
        gate_names.append(gate_name)
        up_names.append(up_name)
        down_names.append(down_name)

    if not q_names:
        raise RuntimeError("no complete transformer blocks found for inferred AWQ mapping")

    mapping_pairs = [
        (_as_exact_regex(input_norms), _as_exact_regex(q_names + k_names + v_names)),
        (_as_exact_regex(v_names), _as_exact_regex(o_names)),
        (_as_exact_regex(post_norms), _as_exact_regex(gate_names + up_names)),
        (_as_exact_regex(up_names), _as_exact_regex(down_names)),
    ]
    meta = {
        "mapping_mode": "inferred",
        "mapping_blocks": len(q_names),
    }
    return mapping_pairs, meta


def build_awq_mappings(model) -> Tuple[List[Tuple[List[str], List[str]]], Dict[str, Any]]:
    module_names = [name for name, _ in model.named_modules() if name]
    static_pairs = _static_mapping_pairs()
    if _mapping_supported_by_model(static_pairs, module_names):
        return static_pairs, {"mapping_mode": "regex_static", "mapping_blocks": None}

    inferred_pairs, inferred_meta = _infer_mapping_pairs(module_names)
    return inferred_pairs, inferred_meta


def _build_awq_modifier_kwargs(
    modifier_cls,
    mapping_pairs: List[Tuple[List[str], List[str]]],
    *,
    group_size: int,
    symmetric: bool,
    duo_scaling: bool,
    offload_device: Optional[str],
    targets: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    signature = inspect.signature(modifier_cls)
    names = set(signature.parameters.keys())
    kwargs: Dict[str, Any] = {}

    resolved_targets = [item.strip() for item in (targets or ["Linear"]) if str(item).strip()]
    resolved_ignore = [item.strip() for item in (ignore_patterns or ["lm_head"]) if str(item).strip()]

    if "ignore" in names:
        kwargs["ignore"] = resolved_ignore
    if "targets" in names:
        kwargs["targets"] = resolved_targets
    if "duo_scaling" in names:
        kwargs["duo_scaling"] = bool(duo_scaling)
    if "offload_device" in names and offload_device:
        kwargs["offload_device"] = offload_device

    if "config_groups" in names:
        kwargs["config_groups"] = {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": bool(symmetric),
                    "strategy": "group",
                    "group_size": int(group_size),
                },
            }
        }
    else:
        if "scheme" in names:
            kwargs["scheme"] = "W4A16"
        if "num_bits" in names:
            kwargs["num_bits"] = 4
        if "bits" in names:
            kwargs["bits"] = 4
        if "group_size" in names:
            kwargs["group_size"] = int(group_size)
        if "symmetric" in names:
            kwargs["symmetric"] = bool(symmetric)
        if "weights" in names:
            kwargs["weights"] = {
                "num_bits": 4,
                "type": "int",
                "symmetric": bool(symmetric),
                "strategy": "group",
                "group_size": int(group_size),
            }

    mapping_target_name = None
    for candidate in ("mappings", "mapping", "awq_mappings"):
        if candidate in names:
            mapping_target_name = candidate
            break
    if mapping_target_name is not None:
        kwargs[mapping_target_name] = mapping_pairs

    return kwargs


def run_awq_oneshot(
    *,
    model,
    calib_dataset,
    max_seq_length: int,
    num_calibration_samples: int,
    group_size: int = 128,
    symmetric: bool = False,
    duo_scaling: bool = False,
    offload_device: Optional[str] = None,
    targets: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
):
    mapping_pairs, mapping_meta = build_awq_mappings(model)
    modifier_cls = _resolve_awq_modifier()
    modifier_kwargs = _build_awq_modifier_kwargs(
        modifier_cls,
        mapping_pairs,
        group_size=group_size,
        symmetric=symmetric,
        duo_scaling=duo_scaling,
        offload_device=offload_device,
        targets=targets,
        ignore_patterns=ignore_patterns,
    )

    try:
        modifier = modifier_cls(**modifier_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"AWQModifier init failed. kwargs={sorted(modifier_kwargs.keys())}, error={exc}"
        ) from exc

    oneshot_kwargs = dict(
        model=model,
        dataset=calib_dataset,
        recipe=[modifier],
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
        concatenate_data=True,
        pad_to_max_length=False,
        shuffle_calibration_samples=True,
    )
    if output_dir:
        oneshot_kwargs["output_dir"] = str(output_dir)
    oneshot(**oneshot_kwargs)

    meta = {
        **mapping_meta,
        "awq_group_size": int(group_size),
        "awq_symmetric": bool(symmetric),
        "awq_duo_scaling": bool(duo_scaling),
        "awq_offload_device": offload_device,
        "awq_modifier_kwargs": sorted(modifier_kwargs.keys()),
        "awq_targets": [item.strip() for item in (targets or ["Linear"]) if str(item).strip()],
        "awq_ignore_patterns": [item.strip() for item in (ignore_patterns or ["lm_head"]) if str(item).strip()],
        "output_dir": output_dir,
    }
    return model, meta
