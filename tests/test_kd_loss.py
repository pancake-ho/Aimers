from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kd_loss_mod = _load_module(MODEL_ROOT / "tuning" / "kd_loss.py", "kd_loss_mod")
token_compat_mod = _load_module(MODEL_ROOT / "tuning" / "tokenizer_compat.py", "token_compat_mod")

compute_kd_ce_loss = kd_loss_mod.compute_kd_ce_loss
tokenizers_compatible = token_compat_mod.tokenizers_compatible


def test_kd_mask_ignores_prompt_positions() -> None:
    vocab = 3
    student = torch.zeros((1, 4, vocab), dtype=torch.float32)
    teacher_a = torch.zeros((1, 4, vocab), dtype=torch.float32)
    teacher_b = torch.zeros((1, 4, vocab), dtype=torch.float32)

    # Shifted labels become [1, -100, 2], so position 1 is masked.
    labels = torch.tensor([[0, 1, -100, 2]], dtype=torch.long)
    teacher_b[0, 1, :] = torch.tensor([20.0, -20.0, -20.0])

    loss_a = compute_kd_ce_loss(
        student_logits=student,
        teacher_logits=teacher_a,
        labels=labels,
        temperature=2.0,
        alpha=1.0,
        use_logit_kd=True,
    )
    loss_b = compute_kd_ce_loss(
        student_logits=student,
        teacher_logits=teacher_b,
        labels=labels,
        temperature=2.0,
        alpha=1.0,
        use_logit_kd=True,
    )
    assert torch.allclose(loss_a["kd_loss"], loss_b["kd_loss"], atol=1e-6)


def test_kl_direction_teacher_to_student() -> None:
    student = torch.tensor([[[0.0, 0.0, 0.0], [0.2, -0.1, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    teacher = torch.tensor([[[0.0, 0.0, 0.0], [1.0, -1.0, 0.5], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    labels = torch.tensor([[0, 1, 2]], dtype=torch.long)

    out = compute_kd_ce_loss(
        student_logits=student,
        teacher_logits=teacher,
        labels=labels,
        temperature=2.0,
        alpha=1.0,
        use_logit_kd=True,
    )

    s = student[:, :-1, :] / 2.0
    t = teacher[:, :-1, :] / 2.0
    expected = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(s, dim=-1),
        torch.nn.functional.softmax(t, dim=-1),
        reduction="none",
    ).sum(-1).mean()
    assert torch.allclose(out["kd_loss"], expected, atol=1e-6)


def test_tokenizer_mismatch_disables_logit_kd_fallback() -> None:
    class MockTokenizer:
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size
            self.pad_token_id = 0
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.unk_token_id = 3

        def __call__(self, text: str, add_special_tokens: bool = False):
            _ = add_special_tokens
            return {"input_ids": [ord(c) % 7 for c in text]}

    ok, reason = tokenizers_compatible(MockTokenizer(100), MockTokenizer(101))
    assert ok is False
    assert "vocab_size mismatch" in reason

    student = torch.zeros((1, 3, 5), dtype=torch.float32)
    teacher = torch.randn((1, 3, 5), dtype=torch.float32)
    labels = torch.tensor([[0, 1, 2]], dtype=torch.long)
    out = compute_kd_ce_loss(
        student_logits=student,
        teacher_logits=teacher,
        labels=labels,
        temperature=2.0,
        alpha=0.7,
        use_logit_kd=False,
    )
    assert torch.allclose(out["kd_loss"], torch.tensor(0.0), atol=1e-8)
    assert torch.allclose(out["loss"], out["ce_loss"], atol=1e-8)
