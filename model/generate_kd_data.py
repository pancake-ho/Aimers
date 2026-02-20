from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .utils.chat_preprocess import normalize_conversations, render_chat_prompt
    from .utils.seed import set_seed
    from .utils.teacher_select import select_teacher_model
except ImportError:
    from utils.chat_preprocess import normalize_conversations, render_chat_prompt
    from utils.seed import set_seed
    from utils.teacher_select import select_teacher_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sequence-level KD pseudo labels.")
    parser.add_argument("--student_model", type=str, default="./base_model")
    parser.add_argument("--teacher_model", type=str, default="")
    parser.add_argument("--teacher_preset_32b", type=str, default="LGAI-EXAONE/EXAONE-4.0-32B")
    parser.add_argument("--teacher_preset_24b", type=str, default="LGAI-EXAONE/EXAONE-4.0-2.4B")
    parser.add_argument("--teacher_preset_12b", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument("--dataset_id", type=str, default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--kd_format",
        type=str,
        choices=["prompt_completion", "conversations", "both"],
        default="prompt_completion",
    )
    parser.add_argument("--output_path", type=str, default="./kd_data/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def runtime_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def first_param_device(model) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def load_teacher_model(model_id: str, load_hint: str):
    dtype = runtime_torch_dtype()
    common_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        common_kwargs["device_map"] = "auto"

    if load_hint == "try_4bit_then_fp16":
        try:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant,
                **common_kwargs,
            )
        except Exception as exc:
            print(f"[WARN] 4bit teacher load failed, fallback to dense load: {exc}")

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        **common_kwargs,
    )


@torch.inference_mode()
def generate_completion(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = first_param_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    generated = model.generate(
        **inputs,
        do_sample=temperature > 0.0,
        temperature=float(temperature) if temperature > 0.0 else 1.0,
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        eos_token_id=eos,
        pad_token_id=pad,
    )

    input_len = int(inputs["input_ids"].shape[-1])
    new_tokens = generated[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_conversation_with_completion(conversations: List[Dict[str, str]], completion: str) -> List[Dict[str, str]]:
    output = [{"role": turn["role"], "content": turn["content"]} for turn in conversations]
    output.append({"role": "assistant", "content": completion})
    return output


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_path = Path(args.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    teacher_info = select_teacher_model(
        teacher_model=args.teacher_model,
        teacher_preset_32b=args.teacher_preset_32b,
        teacher_preset_24b=args.teacher_preset_24b,
        teacher_preset_12b=args.teacher_preset_12b,
    )

    teacher_model_id = str(teacher_info["teacher_model"])
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)
    teacher_model = load_teacher_model(teacher_model_id, str(teacher_info["load_hint"]))
    teacher_model.eval()

    dataset = load_dataset(args.dataset_id, split=args.dataset_split)
    dataset = dataset.shuffle(seed=args.seed)
    take_n = min(int(args.num_samples), len(dataset))
    dataset = dataset.select(range(take_n))

    produced = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as writer:
        for idx, example in enumerate(dataset):
            conversations = normalize_conversations(example)
            if not conversations:
                skipped += 1
                continue

            student_prompt = render_chat_prompt(
                tokenizer=student_tokenizer,
                conversations=conversations,
                add_generation_prompt=True,
            )
            try:
                teacher_prompt = render_chat_prompt(
                    tokenizer=teacher_tokenizer,
                    conversations=conversations,
                    add_generation_prompt=True,
                )
            except Exception:
                teacher_prompt = student_prompt

            completion = generate_completion(
                model=teacher_model,
                tokenizer=teacher_tokenizer,
                prompt_text=teacher_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            if not completion:
                skipped += 1
                continue

            if args.kd_format == "prompt_completion":
                record = {"prompt": student_prompt, "completion": completion}
            elif args.kd_format == "conversations":
                record = {"conversations": build_conversation_with_completion(conversations, completion)}
            else:
                record = {
                    "prompt": student_prompt,
                    "completion": completion,
                    "conversations": build_conversation_with_completion(conversations, completion),
                }

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            produced += 1

            if (idx + 1) % 100 == 0:
                print(f"[INFO] processed={idx + 1} produced={produced} skipped={skipped}")

    meta = {
        "output_path": str(out_path),
        "teacher_selection": teacher_info,
        "dataset_id": args.dataset_id,
        "dataset_split": args.dataset_split,
        "requested_samples": int(args.num_samples),
        "produced_samples": int(produced),
        "skipped_samples": int(skipped),
        "kd_format": args.kd_format,
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] KD data written: {out_path}")
    print(f"[INFO] KD metadata written: {meta_path}")
    print(f"[INFO] teacher_model={teacher_model_id} mode={teacher_info['selection_mode']}")


if __name__ == "__main__":
    main()
