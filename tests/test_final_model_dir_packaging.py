from __future__ import annotations

from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "model" / "main.py"


def test_main_packages_submit_from_selected_final_model_dir() -> None:
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "final_model_dir, quant_records, best_note = run_quantization_with_grid(" in source
    assert 'create_submit_zip(final_model_dir, out_dir / "submit.zip")' in source
    assert "FINAL_SUBMISSION final_model_dir=" in source
    assert "pack_model_dir(" not in source
