from __future__ import annotations

from model.train_kd_lora import build_labels_for_prompt_completion


class TinyTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        _ = add_special_tokens
        ids = [ord(ch) % 97 for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }


def test_prompt_tokens_are_masked_with_minus_100() -> None:
    tok = TinyTokenizer()
    prompt = "USER:"
    completion = "ANSWER"
    out = build_labels_for_prompt_completion(tok, prompt, completion, max_seq_len=128)
    prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])

    assert len(out["input_ids"]) == len(out["labels"])
    assert all(v == -100 for v in out["labels"][:prompt_len])
    assert any(v != -100 for v in out["labels"][prompt_len:])

