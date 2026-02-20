from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"
TOKENIZER_PATH = MODEL_ROOT / "utils" / "tokenizer.py"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tok:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"


def test_load_tokenizer_uses_fix_mistral_when_supported(monkeypatch) -> None:
    calls = []

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id_or_path, **kwargs):
            _ = model_id_or_path
            calls.append(kwargs)
            return _Tok()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer))
    mod = _load_module(TOKENIZER_PATH, "tokenizer_mod_supported")
    tok = mod.load_tokenizer("dummy-model")
    assert calls
    assert calls[0]["fix_mistral_regex"] is True
    assert tok.pad_token == tok.eos_token


def test_load_tokenizer_falls_back_when_fix_flag_unsupported(monkeypatch) -> None:
    calls = []

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id_or_path, **kwargs):
            _ = model_id_or_path
            calls.append(kwargs)
            if "fix_mistral_regex" in kwargs:
                raise TypeError("unexpected keyword")
            return _Tok()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer))
    mod = _load_module(TOKENIZER_PATH, "tokenizer_mod_fallback")
    tok = mod.load_tokenizer("dummy-model")
    assert len(calls) == 2
    assert "fix_mistral_regex" in calls[0]
    assert "fix_mistral_regex" not in calls[1]
    assert tok.pad_token == tok.eos_token
