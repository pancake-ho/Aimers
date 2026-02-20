from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from datasets import Dataset

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

try:
    from .utils.chat_preprocess import normalize_conversations, render_chat_prompt
    from .utils.seed import set_seed
except ImportError:
    from utils.chat_preprocess import normalize_conversations, render_chat_prompt
    from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KD student with LoRA SFT and merge.")
    parser.add_argument("--base_model", type=str, default="./base_model")
    parser.add_argument("--kd_data_path", type=str, default="./kd_data/train.jsonl")
    parser.add_argument(
        "--data_format",
        type=str,
        choices=["prompt_completion", "conversations"],
        default="prompt_completion",
    )
    parser.add_argument("--output_lora_dir", type=str, default="./distilled_lora")
    parser.add_argument("--output_merged_dir", type=str, default="./distilled_merged")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def runtime_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def bf16_fp16_flags(dtype: torch.dtype) -> tuple[bool, bool]:
    if dtype == torch.bfloat16:
        return True, False
    if dtype == torch.float16:
        return False, True
    return False, False


def load_jsonl(path: str | Path) -> List[Dict]:
    records: List[Dict] = []
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"KD data file not found: {src}")
    with src.open("r", encoding="utf-8") as reader:
        for line in reader:
            text = line.strip()
            if not text:
                continue
            records.append(json.loads(text))
    if not records:
        raise ValueError("KD data is empty.")
    return records


def build_labels_for_prompt_completion(tokenizer, prompt: str, completion: str, max_seq_len: int) -> Dict[str, List[int]]:
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_text = f"{prompt}{completion}{eos}"
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=int(max_seq_len),
    )
    input_ids = list(enc["input_ids"])
    attention_mask = list(enc["attention_mask"])
    labels = list(input_ids)
    cutoff = min(len(prompt_ids), len(labels))
    labels[:cutoff] = [-100] * cutoff
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_prompt_completion_record(tokenizer, record: Dict, max_seq_len: int) -> Dict[str, List[int]]:
    prompt = str(record.get("prompt", ""))
    completion = str(record.get("completion", ""))
    if not prompt or not completion:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    return build_labels_for_prompt_completion(
        tokenizer=tokenizer,
        prompt=prompt,
        completion=completion,
        max_seq_len=max_seq_len,
    )


def tokenize_conversations_record(tokenizer, record: Dict, max_seq_len: int) -> Dict[str, List[int]]:
    conversations = normalize_conversations(record)
    if not conversations:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    if conversations[-1]["role"] == "assistant":
        prompt_conv = conversations[:-1]
        completion = conversations[-1]["content"]
        prompt = render_chat_prompt(tokenizer, prompt_conv, add_generation_prompt=True)
        return build_labels_for_prompt_completion(
            tokenizer=tokenizer,
            prompt=prompt,
            completion=completion,
            max_seq_len=max_seq_len,
        )

    full_text = render_chat_prompt(tokenizer, conversations, add_generation_prompt=False)
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=int(max_seq_len),
    )
    labels = list(enc["input_ids"])
    return {
        "input_ids": list(enc["input_ids"]),
        "attention_mask": list(enc["attention_mask"]),
        "labels": labels,
    }


def _module_parent(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for key in parts[:-1]:
        parent = getattr(parent, key)
    return parent, parts[-1]


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear requires nn.Linear")
        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha / r) if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features
        device = self.base_layer.weight.device
        base_dtype = self.base_layer.weight.dtype

        if self.r > 0:
            A = torch.empty((self.r, in_features), device=device, dtype=torch.float32)
            B = torch.empty((out_features, self.r), device=device, dtype=torch.float32)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)
            self.lora_A = nn.Parameter(A.to(dtype=base_dtype))
            self.lora_B = nn.Parameter(B.to(dtype=base_dtype))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if self.r <= 0:
            return base
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base + delta * self.scaling


def inject_lora(model: nn.Module, target_modules: List[str], r: int, alpha: int, dropout: float) -> int:
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("lm_head") or "embed_tokens" in name:
            continue
        if not any(name.endswith(suffix) for suffix in target_modules):
            continue
        parent, child_name = _module_parent(model, name)
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced += 1
    return replaced


def merge_lora(model: nn.Module) -> int:
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        base = module.base_layer
        if module.r > 0:
            with torch.no_grad():
                delta = (module.lora_B.float() @ module.lora_A.float()) * module.scaling
                delta = delta.to(dtype=base.weight.dtype, device=base.weight.device)
                base.weight.add_(delta)
        parent, child_name = _module_parent(model, name)
        setattr(parent, child_name, base)
        merged += 1
    return merged


def collect_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    payload: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        if module.lora_A is None or module.lora_B is None:
            continue
        payload[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        payload[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return payload


def auto_detect_target_modules(model: nn.Module, user_value: str) -> List[str]:
    if user_value.strip():
        return [token.strip() for token in user_value.split(",") if token.strip()]

    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    selected = sorted({suffix for suffix in preferred if any(name.endswith(suffix) for name in names)})
    if selected:
        return selected

    fallback = []
    for name in names:
        leaf = name.split(".")[-1]
        lowered = name.lower()
        if leaf == "lm_head" or "embed_tokens" in lowered:
            continue
        if ("attn" in lowered) or ("attention" in lowered) or ("mlp" in lowered) or ("ffn" in lowered):
            fallback.append(leaf)
    if fallback:
        return sorted(set(fallback))

    generic = [name.split(".")[-1] for name in names if "embed_tokens" not in name and not name.endswith("lm_head")]
    return sorted(set(generic))


def enable_input_require_grads(model: nn.Module) -> None:
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
            return
        except Exception:
            pass

    emb = None
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
        except Exception:
            emb = None
    if emb is None:
        return

    def _hook(_module, _inputs, output):
        if torch.is_tensor(output):
            output.requires_grad_(True)
        return output

    if not hasattr(emb, "_lora_input_hook"):
        emb._lora_input_hook = emb.register_forward_hook(_hook)


@dataclass
class DataCollatorForCausalLM:
    tokenizer: any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pad_id = int(self.tokenizer.pad_token_id)

        def to_tensor(value):
            if torch.is_tensor(value):
                return value.to(dtype=torch.long)
            return torch.tensor(value, dtype=torch.long)

        input_ids = [to_tensor(item["input_ids"]) for item in features]
        attention_mask = [to_tensor(item["attention_mask"]) for item in features]
        labels = [to_tensor(item["labels"]) for item in features]

        max_len = max(seq.numel() for seq in input_ids)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            if max_len % multiple != 0:
                max_len = ((max_len // multiple) + 1) * multiple

        def pad(seq: torch.Tensor, fill: int) -> torch.Tensor:
            if seq.numel() == max_len:
                return seq
            tail = torch.full((max_len - seq.numel(),), fill, dtype=seq.dtype)
            return torch.cat([seq, tail], dim=0)

        batch_ids = torch.stack([pad(seq, pad_id) for seq in input_ids], dim=0)
        batch_mask = torch.stack([pad(seq, 0) for seq in attention_mask], dim=0)
        batch_labels = torch.stack([pad(seq, -100) for seq in labels], dim=0)
        batch_labels = batch_labels.masked_fill(batch_labels == pad_id, -100)

        return {"input_ids": batch_ids, "attention_mask": batch_mask, "labels": batch_labels}


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    args = parse_args()
    set_seed(args.seed)

    dtype = runtime_dtype()
    bf16_flag, fp16_flag = bf16_fp16_flags(dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if hasattr(model, "config"):
        model.config.use_cache = False

    records = load_jsonl(args.kd_data_path)
    dataset = Dataset.from_list(records)

    if args.data_format == "prompt_completion":
        tokenize_fn = lambda row: tokenize_prompt_completion_record(tokenizer, row, args.max_seq_len)
    else:
        tokenize_fn = lambda row: tokenize_conversations_record(tokenizer, row, args.max_seq_len)

    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    tokenized = tokenized.filter(lambda row: len(row["input_ids"]) > 0 and any(v != -100 for v in row["labels"]))
    tokenized.set_format(type="torch")

    for param in model.parameters():
        param.requires_grad_(False)

    target_modules = auto_detect_target_modules(model, args.target_modules)
    injected = inject_lora(
        model=model,
        target_modules=target_modules,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if injected == 0:
        raise RuntimeError(f"No LoRA target module was injected. target_modules={target_modules}")

    enable_input_require_grads(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] trainable params: {trainable} / {total} ({(100.0 * trainable / total):.4f}%)")
    print(f"[INFO] injected modules: {injected}")
    print(f"[INFO] target modules: {','.join(target_modules)}")

    training_args = TrainingArguments(
        output_dir="./trainer_kd_lora",
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.per_device_batch_size),
        gradient_accumulation_steps=int(args.grad_accum_steps),
        bf16=bf16_flag,
        fp16=fp16_flag,
        gradient_checkpointing=True,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=int(args.seed),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer, pad_to_multiple_of=8),
    )
    trainer.train()

    lora_dir = Path(args.output_lora_dir).resolve()
    lora_dir.mkdir(parents=True, exist_ok=True)
    lora_state = collect_lora_state(model)
    torch.save(lora_state, lora_dir / "adapter_model.bin")
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": target_modules,
                "trainable_params": trainable,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tokenizer.save_pretrained(lora_dir)

    merged_count = merge_lora(model)
    merged_dir = Path(args.output_merged_dir).resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    print(f"[INFO] merged LoRA modules: {merged_count}")
    print(f"[INFO] adapter saved: {lora_dir}")
    print(f"[INFO] merged model saved: {merged_dir}")


if __name__ == "__main__":
    main()
