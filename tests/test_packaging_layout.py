from __future__ import annotations

import zipfile
from pathlib import Path

from model.quantize import make_submit_archive


def test_packaging_layout_uses_model_folder(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"123")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    zip_path = make_submit_archive(str(tmp_path / "submit"), model_dir)
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    assert any(name.endswith("model/config.json") for name in names)
    assert any(name.endswith("model/model.safetensors") for name in names)

