from pathlib import Path
import shutil

def save(out_dir, model, tokenizer):
    out_dir = Path(out_dir)
    tmp_root = Path("./_pack")
    staging_dir = tmp_root / out_dir.name
    shutil.rmtree(tmp_root, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    current_step = "model.save_pretrained"
    try:
        model.save_pretrained(output_dir=staging_dir, save_compressed=True)
        current_step = "tokenizer.save_pretrained"
        tokenizer.save_pretrained(staging_dir)
        current_step = "shutil.make_archive"
        shutil.make_archive(
            base_name="submit",
            format="zip",
            root_dir=tmp_root,
            base_dir=out_dir.name,
        )
    except Exception as e:
        print(f"[ERROR] save 실패 ({current_step}): {e}")
        raise
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)