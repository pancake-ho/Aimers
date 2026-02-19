#!/usr/bin/env python
"""
Model6 stage-2 pipeline:
1) Load merged full-weight model (no LoRA adapters)
2) AutoRound uniform 4-bit GPTQ (iters=1000)
3) Export auto_gptq with optional Marlin flag when supported
4) Package submit.zip with root directory model/
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
    p = argparse.ArgumentParser(description="Model6 final quantization and packaging")
    p.add_argument("--merged_model_dir", type=str, default="./merged_model6")
    p.add_argument("--output_dir", type=str, default="./model")
    p.add_argument("--zip_name", type=str, default="submit")

    p.add_argument("--calib_dataset", type=str, default="LGAI-EXAONE/MANTA-1M")
    p.add_argument("--calib_config", type=str, default=None)
    p.add_argument("--calib_split", type=str, default="train")
    p.add_argument("--calib_samples", type=int, default=384)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_model_len", type=int, default=512)
    p.add_argument("--sym", action="store_true", default=True)
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


def _example_to_prompt(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    messages: List[Dict[str, str]] = []
    if "conversations" in example:
        messages = _normalize_messages(example["conversations"])
    elif "messages" in example:
        messages = _normalize_messages(example["messages"])
    elif "question" in example and "answer" in example:
        messages = [
            {"role": "user", "content": str(example["question"]).strip()},
            {"role": "assistant", "content": str(example["answer"]).strip()},
        ]
    elif "problem" in example and "solution" in example:
        messages = [
            {"role": "user", "content": str(example["problem"]).strip()},
            {"role": "assistant", "content": str(example["solution"]).strip()},
        ]
    elif "prompt" in example and "response" in example:
        messages = [
            {"role": "user", "content": str(example["prompt"]).strip()},
            {"role": "assistant", "content": str(example["response"]).strip()},
        ]
    elif "input" in example and "output" in example:
        messages = [
            {"role": "user", "content": str(example["input"]).strip()},
            {"role": "assistant", "content": str(example["output"]).strip()},
        ]
    elif "text" in example and str(example["text"]).strip():
        messages = [{"role": "user", "content": str(example["text"]).strip()}]

    if messages:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return "\n".join([f"<|{m['role']}|>\n{m['content']}" for m in messages])
    return ""


def _load_split(dataset_name_or_path: str, dataset_config: str | None, split: str) -> Dataset:
    if os.path.isdir(dataset_name_or_path):
        ds = load_from_disk(dataset_name_or_path)
        if isinstance(ds, DatasetDict):
            if split in ds:
                return ds[split]
            first_key = next(iter(ds.keys()))
            return ds[first_key]
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
        first_key = next(iter(ds.keys()))
        return ds[first_key]
    return ds


def _prepare_calibration(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> List[Dict[str, torch.Tensor]]:
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(nsamples, len(dataset))))

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
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }

    dataset = dataset.map(_map, remove_columns=dataset.column_names, desc="Preparing calibration samples")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return [dataset[i] for i in range(len(dataset))]


def _update_vllm_config_hints(model_dir: str, max_model_len: int) -> None:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["max_model_len"] = int(max_model_len)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _build_autoround_kwargs(args: argparse.Namespace, calib_list: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "bits": 4,
        "group_size": args.group_size,
        "sym": args.sym,
        "dataset": calib_list,
        "seqlen": args.seqlen,
        "nsamples": len(calib_list),
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
    save_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "format": "auto_gptq",
        "inplace": True,
    }
    save_params = inspect.signature(autoround.save_quantized).parameters
    if use_marlin:
        if "use_marlin" in save_params:
            save_kwargs["use_marlin"] = True
        elif "enable_marlin" in save_params:
            save_kwargs["enable_marlin"] = True
        else:
            print("[WARN] AutoRound save_quantized() has no Marlin flag; exporting standard auto_gptq.")
    autoround.save_quantized(**save_kwargs)


def _package_zip(output_dir: str, zip_name: str) -> None:
    if os.path.basename(os.path.abspath(output_dir)) != "model":
        if os.path.isdir("./model"):
            shutil.rmtree("./model")
        shutil.copytree(output_dir, "./model")
        base_dir = "model"
    else:
        base_dir = "model"
    zip_path = f"{zip_name}.zip"
    if os.path.isfile(zip_path):
        os.remove(zip_path)
    shutil.make_archive(base_name=zip_name, format="zip", root_dir=".", base_dir=base_dir)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    bf16 = torch.cuda.is_bf16_supported()

    print("[1/6] Loading merged model and tokenizer...")
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

    print("[2/6] Building calibration dataset...")
    calib_ds = _load_split(args.calib_dataset, args.calib_config, args.calib_split)
    calib_list = _prepare_calibration(
        calib_ds,
        tokenizer=tokenizer,
        nsamples=args.calib_samples,
        seqlen=args.seqlen,
        seed=args.seed,
    )
    if len(calib_list) == 0:
        raise RuntimeError("Calibration dataset is empty after preprocessing.")

    print("[3/6] AutoRound 4-bit quantization...")
    autoround_kwargs = _build_autoround_kwargs(args, calib_list)
    autoround = AutoRound(model, tokenizer, **autoround_kwargs)
    autoround.quantize()

    print("[4/6] Exporting auto_gptq model...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    _save_quantized(autoround, args.output_dir, use_marlin=args.use_marlin)
    tokenizer.save_pretrained(args.output_dir)
    _update_vllm_config_hints(args.output_dir, args.max_model_len)

    weight_files = [f for f in os.listdir(args.output_dir) if f.endswith(".safetensors")]
    if len(weight_files) != 1:
        raise RuntimeError(
            f"Expected one quantized safetensors file in {args.output_dir}, found: {weight_files}"
        )
    if any("adapter" in f.lower() for f in os.listdir(args.output_dir)):
        raise RuntimeError("Adapter artifact detected in final model directory.")

    print("[5/6] Packaging submit.zip...")
    _package_zip(args.output_dir, args.zip_name)

    print("[6/6] Done.")
    print(f"[DONE] Quantized model dir: {args.output_dir}")
    print(f"[DONE] Zip: {args.zip_name}.zip")


if __name__ == "__main__":
    main()

