"""LoRA fine-tuning (NO peft / NO trl)

이 파일은 기존 `peft`, `trl` 의존성을 제거하고, LoRA를 **직접 구현**한 버전입니다.

핵심 포인트
1) 학습 시에만 LoRA 파라미터를 추가하고(base weight는 freeze)
2) 학습 종료 후 LoRA를 베이스 가중치에 **merge**해서
3) 최종 결과물은 **표준 HF 모델(Linear만 존재)** 로 저장/양자화에 사용 가능

대회(Phase2) 관점 최적화
- 추론 환경에는 peft/trl이 없다고 가정 → 제출 모델은 순수 HF 모델이어야 안전
- LoRA는 "많이"가 아니라 "안정적으로" (양자화 후 성능 유지가 더 중요)
- 기본값은 attention projection 위주(q/k/v/o) + 낮은 rank(r=8)
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments


class LoRALinear(nn.Module):
    """A minimal LoRA wrapper for nn.Linear.

    Forward: y = xW^T + b + (dropout(x) A^T B^T) * (alpha/r)
    - base_layer parameters are frozen.
    - only lora_A / lora_B are trainable.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear only supports nn.Linear, got: {type(base_layer)}")

        self.base_layer = base_layer
        self.r = int(r)
        self.lora_alpha = int(lora_alpha)
        self.scaling = (self.lora_alpha / self.r) if self.r > 0 else 0.0
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()

        # Freeze base weights
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA 파라미터는 base weight와 동일한 device/dtype에 생성 (GPU 왕복 방지)
        dev = base_layer.weight.device
        base_dtype = base_layer.weight.dtype

        if self.r > 0:
            # LoRA params (trainable)
            # A: [r, in_features], B: [out_features, r]
            A = torch.empty(self.r, in_features, device=dev, dtype=torch.float32)
            B = torch.empty(out_features, self.r, device=dev, dtype=torch.float32)
            # init: A kaiming, B zeros (standard LoRA init)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)
            self.lora_A = nn.Parameter(A.to(dtype=base_dtype))
            self.lora_B = nn.Parameter(B.to(dtype=base_dtype))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_layer(x)
        if self.r <= 0:
            return y

        # Compute delta in the same dtype/device as input
        x_d = self.dropout(x)
        delta = (x_d @ self.lora_A.t()) @ self.lora_B.t()
        return y + delta * self.scaling


def _get_parent_module(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_module_names: List[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    verbose: bool = True,
) -> int:
    """Replace target Linear layers with LoRALinear.

    Returns number of layers replaced.
    """

    replaced = 0
    # named_modules gives hierarchical names
    for name, module in list(model.named_modules()):
        if not any(name.endswith(t) for t in target_module_names):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("lm_head"):
            continue

        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, r=r, lora_alpha=alpha, lora_dropout=dropout))
        replaced += 1

    if verbose:
        print(f"[LoRA] injected into {replaced} Linear layers")
    return replaced


def merge_lora(model: nn.Module, verbose: bool = True) -> nn.Module:
    """Merge LoRA weights into base Linear weights, then unwrap LoRALinear -> nn.Linear."""

    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        base = module.base_layer
        if module.r > 0:
            # W <- W + (B @ A) * scaling
            # Shapes: B[out, r] @ A[r, in] = [out, in]
            # merge는 1회성 연산이므로 float32로 계산해 수치 오차를 줄입니다.
            delta_w = (module.lora_B.float() @ module.lora_A.float()) * float(module.scaling)
            # Ensure dtype/device match
            delta_w = delta_w.to(dtype=base.weight.dtype, device=base.weight.device)
            with torch.no_grad():
                base.weight.add_(delta_w)
        # Replace wrapper with merged base
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, base)
        merged += 1

    if verbose:
        print(f"[LoRA] merged and unwrapped {merged} layers")
    return model


@dataclass
class DataCollatorForCausalLM:
    """Robust padding collator for Causal LM.

    - `datasets.set_format(type="torch")` 여부와 무관하게 동작하도록
      feature가 list[int] / torch.Tensor 둘 다 처리합니다.
    - labels는 padding 구간을 -100으로 채워 loss에서 제외합니다.
    """

    tokenizer: any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pad_id = int(self.tokenizer.pad_token_id)

        def _to_1d_long(x):
            if torch.is_tensor(x):
                return x.to(dtype=torch.long)
            return torch.tensor(x, dtype=torch.long)

        input_ids = [_to_1d_long(f["input_ids"]) for f in features]
        attn = [_to_1d_long(f.get("attention_mask", [1] * len(f["input_ids"]))) for f in features]
        labels = [_to_1d_long(f.get("labels", f["input_ids"])) for f in features]

        max_len = max(x.numel() for x in input_ids)
        if self.pad_to_multiple_of:
            m = int(self.pad_to_multiple_of)
            if max_len % m != 0:
                max_len = ((max_len // m) + 1) * m

        def _pad(seq, value):
            if seq.numel() == max_len:
                return seq
            pad = torch.full((max_len - seq.numel(),), value, dtype=seq.dtype)
            return torch.cat([seq, pad], dim=0)

        input_ids = torch.stack([_pad(x, pad_id) for x in input_ids], dim=0)
        attn = torch.stack([_pad(x, 0) for x in attn], dim=0)
        labels = torch.stack([_pad(x, -100) for x in labels], dim=0)

        # In case tokenizer.pad_token_id appears in labels (shouldn't after masking), fix it
        labels = labels.masked_fill(labels == pad_id, -100)

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


class Fine_tuning:
    """Fine_tuning 클래스 (대회용: 의존성 최소, merge 보장)"""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        seq_length: int = 2048,
        train_ds=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = int(seq_length)
        self.train_dataset = train_ds

    def _enable_gradient_checkpointing_for_frozen_base(self) -> None:
        """LoRA(+frozen base) + gradient checkpointing에서 흔한 gradient 끊김을 방지.

        증상
        - torch.utils.checkpoint 경고: "None of the inputs have requires_grad=True"
        - 이후 loss.backward 에서 "tensors does not require grad" 에러

        원인
        - base 파라미터를 전부 freeze 하면, 첫 transformer block으로 들어가는
          hidden_states가 requires_grad=False가 되기 쉬움
        - HF의 gradient checkpointing 구현은 checkpoint 입력 중 하나라도
          requires_grad=True가 아니면 그래프를 만들지 않아 trainable LoRA까지
          gradient가 사라질 수 있음

        해결
        - 가능하면 model.enable_input_require_grads() 호출
        - 없으면 input embedding 출력에 requires_grad_(True) 훅을 걸어줌
        """

        # 모델이 제공하면 켜기
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        # 가장 정석 (Transformers 최신 버전)
        if hasattr(self.model, "enable_input_require_grads"):
            try:
                self.model.enable_input_require_grads()
                return
            except Exception:
                # 아래 fallback으로
                pass

        # Fallback: embedding 출력에 requires_grad를 강제로 부여
        emb = None
        if hasattr(self.model, "get_input_embeddings"):
            try:
                emb = self.model.get_input_embeddings()
            except Exception:
                emb = None

        if emb is None:
            return

        def _make_out_require_grad(_module, _inp, out):
            if torch.is_tensor(out):
                out.requires_grad_(True)
            return out

        # 중복 등록 방지
        if not hasattr(emb, "_lora_require_grad_hook"):
            emb._lora_require_grad_hook = emb.register_forward_hook(_make_out_require_grad)

    def _tokenize_for_sft(self, example: Dict) -> Dict:
        """Completion-only SFT 스타일 토크나이징.

        - prompt: 마지막 assistant 응답을 제외한 대화 + generation prompt
        - labels: assistant 응답 구간만 loss 계산(-100 masking)
        """
        conv = example["conversations"]
        if not conv or conv[-1].get("role") != "assistant":
            # fallback: 전체 대화에 대해 LM loss
            full_text = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_seq_length,
                add_special_tokens=False,
            )
            enc["labels"] = enc["input_ids"].copy()
            return enc

        prompt_conv = conv[:-1]
        answer = conv[-1].get("content", "")

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_conv,
            tokenize=False,
            add_generation_prompt=True,
        )
        # assistant answer + eos
        full_text = prompt_text + answer
        if self.tokenizer.eos_token:
            full_text += self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False,
        )["input_ids"]

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False,
        )

        labels = enc["input_ids"].copy()
        cutoff = min(len(prompt_ids), len(labels))
        labels[:cutoff] = [-100] * cutoff
        enc["labels"] = labels
        return enc

    def setup_lora(
        self,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        epochs: float = 1.0,
        lr: float = 1e-4,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        warmup_ratio: float = 0.03,
        seed: int = 42,
        target_modules: Optional[List[str]] = None,
    ):
        """LoRA SFT 학습 후 merge 해서 self.model 반환.

        기본값은 **대회(양자화 후 성능 유지)** 관점에서 안정적으로 잡아두었습니다.

        추천 튜닝 레시피 (Phase2)
        - target_modules: attention projection(q/k/v/o)만
        - r=8, alpha=16, lr=1e-4, epochs=1
        - 너무 세게 하면 양자화 후 성능이 더 무너질 수 있음
        """

        print("==========================================")
        print("[Step 1] LoRA Fine-Tuning 시작 (NO peft/trl)")
        print("==========================================")

        if target_modules is None:
            # 대회용 추천: attention 위주
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # vLLM/transformers warning 방지
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        # 0) 안전장치: 전체 파라미터 freeze (LoRA만 학습)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 1) LoRA 주입
        inject_lora(
            self.model,
            target_module_names=target_modules,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

        # (중요) frozen-base LoRA + gradient checkpointing 조합에서
        # checkpoint 입력이 requires_grad=False가 되어 gradient가 끊기는 문제가 흔합니다.
        # -> 아래 가드가 없으면 trainer.train()에서
        #    "loss does not require grad" 류 에러가 날 수 있어요.
        self._enable_gradient_checkpointing_for_frozen_base()

        # 2) trainable params만 optimizer에 들어가도록 확인
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        total = sum(p.numel() for p in self.model.parameters())
        trainable_n = sum(p.numel() for p in trainable)
        print(f"[LoRA] trainable params: {trainable_n:,} / total: {total:,} ({trainable_n/total*100:.4f}%)")

        # 3) 데이터 토크나이징
        print("[Step 1] 데이터 토크나이징 중...")
        tokenized = self.train_dataset.map(
            self._tokenize_for_sft,
            remove_columns=self.train_dataset.column_names,
        )
        tokenized.set_format(type="torch")

        # 4) Trainer 설정
        args = TrainingArguments(
            output_dir="./trainer_nopeft",
            num_train_epochs=float(epochs),
            per_device_train_batch_size=int(per_device_train_batch_size),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            learning_rate=float(lr),
            warmup_ratio=float(warmup_ratio),
            logging_steps=20,
            save_strategy="no",
            bf16=True,
            fp16=False,
            optim="adamw_torch",
            report_to=[],
            seed=int(seed),
            gradient_checkpointing=True,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized,
            data_collator=DataCollatorForCausalLM(self.tokenizer, pad_to_multiple_of=8),
        )

        trainer.train()

        # 5) merge & unwrap
        print("[Step 1] 학습 완료, LoRA merge 중...")
        self.model = merge_lora(self.model)

        # 메모리 정리
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        print("[Step 1] merge 완료. 이제 모델은 순수 HF 모델(Linear)입니다.")
        return self.model
