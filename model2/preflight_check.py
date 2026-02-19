#!/usr/bin/env python
import argparse
import gc
import os
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from uuid import uuid4

ZIP_LIMIT_BYTES = 10 * 1024**3
UNZIPPED_LIMIT_BYTES = 32 * 1024**3


def bytes_to_gib(n: int) -> float:
    return n / (1024**3)


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def validate_zip(zip_path: Path) -> tuple[int, int]:
    if not zip_path.exists():
        fail(f"Zip not found: {zip_path}")
    if zip_path.suffix.lower() != ".zip":
        fail(f"Expected a .zip file: {zip_path}")

    compressed_size = zip_path.stat().st_size
    if compressed_size > ZIP_LIMIT_BYTES:
        fail(
            f"Compressed size exceeded: {bytes_to_gib(compressed_size):.2f} GiB "
            f"(limit: {bytes_to_gib(ZIP_LIMIT_BYTES):.2f} GiB)"
        )
    ok(
        f"Compressed zip size is within limit: "
        f"{bytes_to_gib(compressed_size):.2f} GiB <= {bytes_to_gib(ZIP_LIMIT_BYTES):.2f} GiB"
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = [i for i in zf.infolist() if i.filename and not i.filename.endswith("/")]
        if not entries:
            fail("Zip has no file entries.")

        top_levels = {item.filename.split("/", 1)[0] for item in entries}
        if top_levels != {"model"}:
            fail(
                "Zip root must contain only model/ directory. "
                f"Found root entries: {sorted(top_levels)}"
            )
        ok("Zip root structure is valid: only model/ at top level.")

        uncompressed_size = sum(item.file_size for item in entries)
        if uncompressed_size > UNZIPPED_LIMIT_BYTES:
            fail(
                f"Uncompressed size exceeded: {bytes_to_gib(uncompressed_size):.2f} GiB "
                f"(limit: {bytes_to_gib(UNZIPPED_LIMIT_BYTES):.2f} GiB)"
            )
        ok(
            f"Uncompressed size is within limit: "
            f"{bytes_to_gib(uncompressed_size):.2f} GiB <= {bytes_to_gib(UNZIPPED_LIMIT_BYTES):.2f} GiB"
        )

    return compressed_size, uncompressed_size


def extract_model_dir(zip_path: Path) -> Path:
    tmp = Path.cwd() / f"preflight_model_{uuid4().hex[:12]}"
    tmp.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for item in zf.infolist():
            name = item.filename.replace("\\", "/")
            if not name:
                continue
            target = tmp / name
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(item, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    model_dir = tmp / "model"
    if not model_dir.exists() or not model_dir.is_dir():
        fail("Extracted zip does not contain model/ directory.")
    info(f"Extracted model directory: {model_dir}")
    return model_dir


def offline_loading_test(model_dir: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info("Running offline loading test with trust_remote_code=True, local_files_only=True")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    ok("Offline tokenizer/model loading succeeded.")

    del model
    del tokenizer
    gc.collect()


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or "oom" in msg
    )


def vllm_boot_test(model_dir: Path) -> None:
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        fail("CUDA GPU not available. vLLM OOM validation requires GPU (L4 target).")

    info(
        "Running vLLM boot simulation with "
        "tensor_parallel_size=1, gpu_memory_utilization=0.85, max_gen_toks=16384"
    )

    # Evaluation guide exposes max_gen_toks=16384. In vLLM LLM init, this maps to max_model_len.
    try:
        llm = LLM(
            model=str(model_dir),
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=16384,
        )
        params = SamplingParams(max_tokens=1, temperature=0.0)
        _ = llm.generate(["preflight"], params)
    except Exception as exc:
        if is_oom_error(exc):
            fail(f"vLLM boot failed due to OOM: {exc}")
        fail(f"vLLM boot failed: {exc}")
    finally:
        try:
            del llm
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    ok("vLLM boot and tiny generation succeeded without OOM.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for Aimers submit.zip")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("submit.zip"),
        help="Path to submit.zip (default: ./submit.zip)",
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Skip vLLM boot simulation.",
    )
    args = parser.parse_args()

    tmp_extract_dir = None
    try:
        validate_zip(args.zip_path)
        tmp_extract_dir = extract_model_dir(args.zip_path)
        # offline_loading_test(tmp_extract_dir)
        if args.skip_vllm:
            info("Skipped vLLM boot test (--skip-vllm).")
        else:
            vllm_boot_test(tmp_extract_dir)
        print("[PASS] All preflight checks passed.")
    except SystemExit:
        raise
    except Exception as exc:
        print("[FAIL] Unexpected error during preflight.")
        print(exc)
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        if tmp_extract_dir is not None:
            try:
                shutil.rmtree(tmp_extract_dir.parent, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
