from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def gather_safetensor_size(model_dir: str | Path) -> Dict[str, object]:
    root = Path(model_dir)
    files: List[Path] = sorted(root.rglob("*.safetensors"))
    total_bytes = sum(int(path.stat().st_size) for path in files if path.is_file())
    total_mib = float(total_bytes / (1024 * 1024))
    return {
        "model_dir": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": int(total_bytes),
        "total_mib": total_mib,
        "files": [str(path.resolve()) for path in files],
    }


def print_size_report(stats: Dict[str, object]) -> None:
    print("[MEMORY] safetensors summary")
    print(f"  model_dir: {stats['model_dir']}")
    print(f"  file_count: {stats['file_count']}")
    print(f"  total_bytes: {stats['total_bytes']}")
    print(f"  total_mib: {stats['total_mib']:.2f}")


def write_size_report(stats: Dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

