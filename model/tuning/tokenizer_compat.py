from __future__ import annotations

from typing import Tuple


def tokenizers_compatible(student_tokenizer, teacher_tokenizer) -> Tuple[bool, str]:
    """Check compatibility for logit KD."""
    try:
        student_vocab = int(getattr(student_tokenizer, "vocab_size", -1))
        teacher_vocab = int(getattr(teacher_tokenizer, "vocab_size", -1))
        if student_vocab != teacher_vocab:
            return False, f"vocab_size mismatch (student={student_vocab}, teacher={teacher_vocab})"

        for token_name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            student_id = getattr(student_tokenizer, token_name, None)
            teacher_id = getattr(teacher_tokenizer, token_name, None)
            if student_id != teacher_id:
                return False, f"{token_name} mismatch (student={student_id}, teacher={teacher_id})"

        probe_text = "KD tokenizer compatibility probe."
        s_ids = student_tokenizer(probe_text, add_special_tokens=False)["input_ids"]
        t_ids = teacher_tokenizer(probe_text, add_special_tokens=False)["input_ids"]
        if s_ids != t_ids:
            return False, "tokenization probe mismatch"
    except Exception as exc:
        return False, f"tokenizer compatibility check failed: {exc}"

    return True, "compatible"
