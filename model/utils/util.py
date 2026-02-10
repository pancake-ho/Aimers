import os
import shutil

def save(out_dir, autoround_wrapper, tokenizer):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Result] GPTQ 포맷으로 저장 중: {out_dir}")
    autoround_wrapper.save_quantized(output_dir=out_dir, format="auto_gptq", inplace=True)
    tokenizer.save_pretrained(out_dir)

    zip_name = "submit"
    print(f"[Result] {zip_name}.zip 압축 중...")
    shutil.make_archive(base_name=zip_name, format="zip", root_dir=".", base_dir=out_dir)

    print("6. 모든 작업 완료.")