from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pack_model_dir import create_submit_zip, pack_model_dir
from tools.submit_guardrails import validate_submit_zip


def _write_minimal_model_dir(path: Path, weight_name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": "exaone4", "architectures": ["Exaone4ForCausalLM"]}),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / weight_name).write_bytes(b"dummy-weights")


def test_pack_and_validate_submit_zip(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    weights_dir = tmp_path / "weights"
    out_model_dir = tmp_path / "out" / "model"
    zip_path = tmp_path / "submit.zip"

    _write_minimal_model_dir(base_dir, "old.safetensors")
    _write_minimal_model_dir(weights_dir, "model.safetensors")
    (weights_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}), encoding="utf-8")

    summary = pack_model_dir(base_dir, weights_dir, out_model_dir)
    assert summary["copied_weight_files"] >= 1
    assert not (out_model_dir / "old.safetensors").exists()
    assert (out_model_dir / "model.safetensors").exists()

    created = create_submit_zip(out_model_dir, zip_path)
    result = validate_submit_zip(created)
    assert result.ok, result.errors


def test_validate_submit_zip_fails_with_extra_root_entries(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_minimal_model_dir(model_dir, "model.safetensors")
    bad_zip = tmp_path / "bad_submit.zip"
    with zipfile.ZipFile(bad_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_dir / "config.json", arcname="model/config.json")
        zf.write(model_dir / "tokenizer_config.json", arcname="model/tokenizer_config.json")
        zf.write(model_dir / "tokenizer.json", arcname="model/tokenizer.json")
        zf.write(model_dir / "model.safetensors", arcname="model/model.safetensors")
        zf.writestr("extra.txt", "not allowed")

    result = validate_submit_zip(bad_zip)
    assert result.ok is False
    assert any("top-level" in msg for msg in result.errors)
