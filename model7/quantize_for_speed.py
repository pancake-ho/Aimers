#!/usr/bin/env python
"""
Model7 stage-2:
1) Load merged full-weight model (no adapter dependency).
2) Quantize with AutoRound -> uniform 4-bit GPTQ (iters=1000).
3) Export auto_gptq with Marlin when supported.
4) Apply speed-safe runtime configs for vLLM.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import shutil
from typing import Any, Dict, List

import torch
from auto_round import AutoRound
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model7 final 4-bit quantization for speed")
    p.add_argument("--merged_model_dir", type=str, default="./merged_model7")
    p.add_argument("--output_dir", type=str, default="./model")

    p.add_argument("--calib_dataset", type=str, default="./curated_train_dataset_model7")
    p.add_argument("--calib_config", type=str, default=None)
    p.add_argument("--calib_split", type=str, default="train")
    p.add_argument("--calib_samples", type=int, default=512)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sym", action="store_true", default=True)

    p.add_argument("--max_model_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--repetition_penalty", type=float, default=1.08)
    p.add_argument("--use-marlin", dest="use_marlin", action="store_true")
    p.add_argument("--no-use-marlin", dest="use_marlin", action="store_false")
    p.set_defaults(use_marlin=True)
    return p.parse_args()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]).strip())
                elif "content" in item:
                    parts.append(str(item["content"]).strip())
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"]).strip()
        if "content" in content:
            return str(content["content"]).strip()
    return str(content).strip()


def _normalize_messages(raw_messages: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []
    role_map = {"human": "user", "gpt": "assistant", "bot": "assistant"}
    out: List[Dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", msg.get("from", "user"))).strip().lower()
        role = role_map.get(role, role)
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = _content_to_text(msg.get("content", msg.get("value", msg.get("text", ""))))
        if content:
            out.append({"role": role, "content": content})
    return out


def _example_to_prompt(ex: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    messages: List[Dict[str, str]] = []
    if "conversations" in ex:
        messages = _normalize_messages(ex["conversations"])
    elif "messages" in ex:
        messages = _normalize_messages(ex["messages"])
    elif "question" in ex and "answer" in ex:
        messages = [
            {"role": "user", "content": str(ex["question"]).strip()},
            {"role": "assistant", "content": str(ex["answer"]).strip()},
        ]
    elif "problem" in ex and "solution" in ex:
        messages = [
            {"role": "user", "content": str(ex["problem"]).strip()},
            {"role": "assistant", "content": str(ex["solution"]).strip()},
        ]
    elif "prompt" in ex and "response" in ex:
        messages = [
            {"role": "user", "content": str(ex["prompt"]).strip()},
            {"role": "assistant", "content": str(ex["response"]).strip()},
        ]
    elif "input" in ex and "output" in ex:
        messages = [
            {"role": "user", "content": str(ex["input"]).strip()},
            {"role": "assistant", "content": str(ex["output"]).strip()},
        ]
    elif "text" in ex and str(ex["text"]).strip():
        messages = [{"role": "user", "content": str(ex["text"]).strip()}]

    if messages:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join([f"<|{m['role']}|>\n{m['content']}" for m in messages])
    return ""


def _load_split(dataset_name_or_path: str, dataset_config: str | None, split: str) -> Dataset:
    if os.path.isdir(dataset_name_or_path):
        ds = load_from_disk(dataset_name_or_path)
        if isinstance(ds, DatasetDict):
            if split in ds:
                return ds[split]
            return ds[next(iter(ds.keys()))]
        if isinstance(ds, Dataset):
            return ds
        raise TypeError("Unsupported local dataset object.")

    if dataset_config:
        try:
            return load_dataset(dataset_name_or_path, dataset_config, split=split)
        except ValueError as e:
            msg = str(e)
            if "BuilderConfig" not in msg or "not found" not in msg:
                raise
    ds = load_dataset(dataset_name_or_path)
    if isinstance(ds, DatasetDict):
        if split in ds:
            return ds[split]
        return ds[next(iter(ds.keys()))]
    return ds


def _prepare_calibration(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> List[Dict[str, torch.Tensor]]:
    dataset = dataset.shuffle(seed=seed).select(range(min(nsamples, len(dataset))))

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = _example_to_prompt(ex, tokenizer)
        if not prompt:
            prompt = "user: summarize this sample."
        tok = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=seqlen,
            return_tensors="pt",
        )
        return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}

    dataset = dataset.map(_map, remove_columns=dataset.column_names, desc="Preparing calibration samples")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return [dataset[i] for i in range(len(dataset))]


def _sanitize_tokenizer_config(tokenizer: AutoTokenizer, output_dir: str) -> None:
    if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
        tokenizer.init_kwargs.pop("fix_mistral_regex", None)
    cfg_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "fix_mistral_regex" in data:
            data.pop("fix_mistral_regex", None)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


def _update_model_config(model_dir: str, max_model_len: int) -> None:
    cfg = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg):
        return
    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["max_model_len"] = int(max_model_len)
    data["use_cache"] = True
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _update_generation_config(
    model_dir: str,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    repetition_penalty: float,
) -> None:
    gen_path = os.path.join(model_dir, "generation_config.json")
    if os.path.isfile(gen_path):
        with open(gen_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data["max_new_tokens"] = int(max_new_tokens)
    data["repetition_penalty"] = float(repetition_penalty)
    data["do_sample"] = False
    data["temperature"] = 0.0
    data["top_p"] = 1.0
    data["pad_token_id"] = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    if tokenizer.eos_token_id is not None:
        data["eos_token_id"] = int(tokenizer.eos_token_id)

    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_autoround_kwargs(args: argparse.Namespace, calib: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "bits": 4,
        "group_size": args.group_size,
        "sym": args.sym,
        "dataset": calib,
        "seqlen": args.seqlen,
        "nsamples": len(calib),
        "iters": args.iters,
        "lr": 1e-2,
        "minmax_lr": 1e-2,
        "enable_quanted_input": True,
        "enable_minmax_tuning": True,
        "batch_size": 1,
        "gradient_accumulate_steps": 8,
        "scale_dtype": torch.float32,
    }
    init_params = inspect.signature(AutoRound.__init__).parameters
    if args.use_marlin:
        if "use_marlin" in init_params:
            kwargs["use_marlin"] = True
        elif "enable_marlin" in init_params:
            kwargs["enable_marlin"] = True
    return kwargs


def _save_quantized(autoround: AutoRound, output_dir: str, use_marlin: bool) -> None:
    kwargs: Dict[str, Any] = {"output_dir": output_dir, "format": "auto_gptq", "inplace": True}
    params = inspect.signature(autoround.save_quantized).parameters
    if use_marlin:
        if "use_marlin" in params:
            kwargs["use_marlin"] = True
        elif "enable_marlin" in params:
            kwargs["enable_marlin"] = True
        else:
            print("[WARN] Marlin flag not found in AutoRound save API; exporting standard auto_gptq.")
    autoround.save_quantized(**kwargs)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    bf16 = torch.cuda.is_bf16_supported()

    print("[1/5] Loading merged model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.merged_model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.merged_model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False

    print("[2/5] Building calibration set...")
    calib_ds = _load_split(args.calib_dataset, args.calib_config, args.calib_split)
    calib = _prepare_calibration(
        dataset=calib_ds,
        tokenizer=tokenizer,
        nsamples=args.calib_samples,
        seqlen=args.seqlen,
        seed=args.seed,
    )
    if len(calib) == 0:
        raise RuntimeError("Calibration set is empty after preprocessing.")

    print("[3/5] AutoRound 4-bit GPTQ quantization...")
    autoround_kwargs = _build_autoround_kwargs(args, calib)
    autoround = AutoRound(model, tokenizer, **autoround_kwargs)
    autoround.quantize()

    print("[4/5] Exporting quantized model for vLLM...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    _save_quantized(autoround, args.output_dir, args.use_marlin)
    tokenizer.save_pretrained(args.output_dir)
    _sanitize_tokenizer_config(tokenizer, args.output_dir)
    _update_model_config(args.output_dir, args.max_model_len)
    _update_generation_config(
        args.output_dir,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    files = os.listdir(args.output_dir)
    weight_files = [f for f in files if f.endswith(".safetensors")]
    if len(weight_files) != 1:
        raise RuntimeError(f"Expected one quantized safetensors file, found: {weight_files}")
    if any("adapter" in f.lower() for f in files):
        raise RuntimeError("Adapter artifact detected in quantized output.")

    print("[5/5] Done.")
    print(f"[DONE] Quantized model dir: {args.output_dir}")


if __name__ == "__main__":
    main()
