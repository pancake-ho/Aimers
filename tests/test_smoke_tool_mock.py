from __future__ import annotations

import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.smoke_test_vllm import run_smoke_test


class _FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def reset_peak_memory_stats():
        return None

    @staticmethod
    def max_memory_reserved():
        return 0


class _FakeTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        _ = args, kwargs
        return _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = tokenize, add_generation_prompt
        return str(messages)


class _FakeModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        _ = args, kwargs
        return _FakeModel()


class _FakeSamplingParams:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs


class _FakeLLM:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs

    def chat(self, messages, sampling_params):
        _ = messages, sampling_params
        token_ids = [1, 2, 3]
        out = types.SimpleNamespace(token_ids=token_ids, text="ok")
        return [types.SimpleNamespace(outputs=[out])]


def test_smoke_tool_with_mocked_dependencies(monkeypatch, tmp_path: Path) -> None:
    fake_torch = types.SimpleNamespace(cuda=_FakeCuda())
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        AutoModelForCausalLM=_FakeModel,
    )
    fake_vllm = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    result = run_smoke_test(model_dir=model_dir, max_new_tokens=8)
    assert result["hf_load_success"] is True
    assert result["vllm_load_success"] is True
    assert result["generated_tokens"] == 3
    assert result["tokens_per_sec"] > 0.0
