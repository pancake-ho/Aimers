from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class GpuInventory:
    cuda_available: bool
    gpu_count: int
    max_vram_gb: float
    total_vram_gb: float


def detect_gpu_inventory() -> GpuInventory:
    try:
        import torch
    except Exception:
        return GpuInventory(cuda_available=False, gpu_count=0, max_vram_gb=0.0, total_vram_gb=0.0)

    if not torch.cuda.is_available():
        return GpuInventory(cuda_available=False, gpu_count=0, max_vram_gb=0.0, total_vram_gb=0.0)

    count = int(torch.cuda.device_count())
    values = []
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        values.append(float(props.total_memory) / (1024 ** 3))
    return GpuInventory(
        cuda_available=True,
        gpu_count=count,
        max_vram_gb=max(values) if values else 0.0,
        total_vram_gb=sum(values),
    )


def select_teacher_model(
    teacher_model: str,
    teacher_preset_32b: str,
    teacher_preset_24b: str,
    teacher_preset_12b: str,
    inventory: Optional[GpuInventory] = None,
) -> Dict[str, object]:
    inv = inventory or detect_gpu_inventory()

    if teacher_model.strip():
        return {
            "teacher_model": teacher_model.strip(),
            "selection_mode": "manual",
            "load_hint": "bf16_fp16",
            "inventory": asdict(inv),
            "reason": "teacher_model explicitly provided",
        }

    can_run_32b = inv.max_vram_gb >= 80.0 or (inv.gpu_count >= 2 and inv.total_vram_gb >= 80.0)
    if can_run_32b:
        return {
            "teacher_model": teacher_preset_32b,
            "selection_mode": "auto_vram_32b",
            "load_hint": "bf16_fp16",
            "inventory": asdict(inv),
            "reason": ">=80GB single GPU or multi-GPU total >=80GB",
        }

    if 24.0 <= inv.max_vram_gb < 80.0:
        return {
            "teacher_model": teacher_preset_24b,
            "selection_mode": "auto_vram_24b",
            "load_hint": "bf16_fp16",
            "inventory": asdict(inv),
            "reason": "24GB~80GB max GPU memory",
        }

    return {
        "teacher_model": teacher_preset_12b,
        "selection_mode": "auto_low_resource",
        "load_hint": "try_4bit_then_fp16",
        "inventory": asdict(inv),
        "reason": "low resource profile",
    }

