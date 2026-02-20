"""KD loss utilities without Trainer dependency."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def compute_kd_ce_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    use_logit_kd: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute CE + masked logit KD loss on answer tokens only.

    KD term follows KL(teacher || student):
      kl = KL( softmax(teacher/T) || softmax(student/T) )
    """
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("student_logits and teacher_logits must be [B, S, V]")
    if labels.ndim != 2:
        raise ValueError("labels must be [B, S]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logits shape mismatch")

    temp = float(temperature)
    alpha = float(alpha)

    s_logits = student_logits[:, :-1, :].contiguous()
    t_logits = teacher_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    ce = F.cross_entropy(
        s_logits.view(-1, s_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    mask = (shift_labels != -100).float()
    kd = torch.zeros_like(ce)
    if use_logit_kd:
        # KL(teacher || student) with log_target=False:
        # F.kl_div(log_p_student, p_teacher)
        log_p_student = F.log_softmax(s_logits / temp, dim=-1)
        p_teacher = F.softmax(t_logits / temp, dim=-1)
        kl_tok = F.kl_div(log_p_student, p_teacher, reduction="none").sum(-1)
        kd = (kl_tok * mask).sum() / mask.sum().clamp_min(1.0)
        total = (1.0 - alpha) * ce + alpha * (kd * (temp * temp))
    else:
        total = ce
    return {
        "loss": total,
        "ce_loss": ce,
        "kd_loss": kd,
        "mask": mask,
    }
