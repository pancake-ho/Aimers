from __future__ import annotations

from model.utils.teacher_select import GpuInventory, select_teacher_model


def test_teacher_manual_override() -> None:
    info = select_teacher_model(
        teacher_model="local/teacher",
        teacher_preset_32b="a32",
        teacher_preset_24b="a24",
        teacher_preset_12b="a12",
        inventory=GpuInventory(cuda_available=True, gpu_count=1, max_vram_gb=8.0, total_vram_gb=8.0),
    )
    assert info["teacher_model"] == "local/teacher"
    assert info["selection_mode"] == "manual"


def test_teacher_selects_32b_on_80gb() -> None:
    info = select_teacher_model(
        teacher_model="",
        teacher_preset_32b="a32",
        teacher_preset_24b="a24",
        teacher_preset_12b="a12",
        inventory=GpuInventory(cuda_available=True, gpu_count=1, max_vram_gb=80.0, total_vram_gb=80.0),
    )
    assert info["teacher_model"] == "a32"
    assert info["selection_mode"] == "auto_vram_32b"


def test_teacher_selects_24b_on_mid_vram() -> None:
    info = select_teacher_model(
        teacher_model="",
        teacher_preset_32b="a32",
        teacher_preset_24b="a24",
        teacher_preset_12b="a12",
        inventory=GpuInventory(cuda_available=True, gpu_count=1, max_vram_gb=48.0, total_vram_gb=48.0),
    )
    assert info["teacher_model"] == "a24"
    assert info["selection_mode"] == "auto_vram_24b"


def test_teacher_selects_low_resource_profile() -> None:
    info = select_teacher_model(
        teacher_model="",
        teacher_preset_32b="a32",
        teacher_preset_24b="a24",
        teacher_preset_12b="a12",
        inventory=GpuInventory(cuda_available=True, gpu_count=1, max_vram_gb=12.0, total_vram_gb=12.0),
    )
    assert info["teacher_model"] == "a12"
    assert info["load_hint"] == "try_4bit_then_fp16"

