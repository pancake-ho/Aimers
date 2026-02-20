from __future__ import annotations

from pathlib import Path

from model.utils.size_report import gather_safetensor_size


def test_size_report_sums_only_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "a.safetensors").write_bytes(b"a" * 10)
    (model_dir / "b.safetensors").write_bytes(b"b" * 20)
    (model_dir / "ignore.bin").write_bytes(b"c" * 99)

    stats = gather_safetensor_size(model_dir)
    assert stats["file_count"] == 2
    assert stats["total_bytes"] == 30
    assert stats["total_mib"] > 0.0

