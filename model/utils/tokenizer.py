from __future__ import annotations

from typing import Any, Dict


def build_tokenizer_kwargs(*, trust_remote_code: bool = True, local_files_only: bool = False) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(trust_remote_code),
    }
    if local_files_only:
        kwargs["local_files_only"] = True
    return kwargs


def load_tokenizer(
    model_id_or_path: str,
    *,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
):
    from transformers import AutoTokenizer

    kwargs = build_tokenizer_kwargs(
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    # transformers versions that include this flag fix known mistral regex tokenizer issues.
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id_or_path,
            fix_mistral_regex=True,
            **kwargs,
        )
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_id_or_path, **kwargs)

    pad_token = getattr(tok, "pad_token", None)
    eos_token = getattr(tok, "eos_token", None)
    if pad_token is None and eos_token is not None:
        try:
            tok.pad_token = eos_token
        except Exception:
            pass
    return tok
