from pathlib import Path
import shutil


def save(out_dir, model, tokenizer):
    """
    산출물 규약:
      - <out_dir>/model/*
      - <out_dir>/submit.zip (zip 최상위: model/)
    """
    out_dir = Path(out_dir)
    model_dir = out_dir / "model"
    tmp_root = out_dir / "_pack_tmp"
    staging_dir = tmp_root / "model"

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    current_step = "model.save_pretrained"
    try:
        # llmcompressor quantized model은 save_compressed를 지원할 수 있으므로 먼저 시도
        try:
            model.save_pretrained(staging_dir, save_compressed=True)
        except TypeError:
            model.save_pretrained(staging_dir, safe_serialization=True)

        current_step = "tokenizer.save_pretrained"
        tokenizer.save_pretrained(staging_dir)

        current_step = "copy model dir"
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.copytree(staging_dir, model_dir)

        current_step = "shutil.make_archive"
        zip_base = out_dir / "submit"
        zip_path = Path(
            shutil.make_archive(
                base_name=str(zip_base),
                format="zip",
                root_dir=tmp_root,
                base_dir="model",
            )
        )
    except Exception as e:
        print(f"[ERROR] save 실패 ({current_step}): {e}")
        raise
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return model_dir, zip_path
