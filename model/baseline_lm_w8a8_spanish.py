import os
import torch
import shutil
from pathlib import Path
from dotenv import load_dotenv

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"     
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = str(BASE_DIR / "model")
ZIP_NAME = "submit_lm_w8a8_spanish_newdata"

# 실행 전에 기존 산출물 보호
output_path = Path(OUT_DIR)
archive_path = BASE_DIR / f"{ZIP_NAME}.zip"

if output_path.exists() or output_path.is_symlink():
    raise FileExistsError(
        f"Output already exists: {output_path}. "
        "Existing model artifacts will not be overwritten."
    )

if archive_path.exists() or archive_path.is_symlink():
    raise FileExistsError(
        f"Archive already exists: {archive_path}. "
        "Existing ZIP will not be overwritten."
    )

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
NUM_GENERAL = 95

NUM_ES = 7
NUM_MATH = 77
NUM_CODE = 77

SHUFFLE_POOL = 2048 
GROUP_SIZE = 128

print(f"NUM_GENERAL={NUM_GENERAL}, NUM_MATH={NUM_MATH}, NUM_CODE={NUM_CODE}, NUM_ES={NUM_ES}")

load_dotenv(Path(__file__).resolve().parent / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set in .env")

# Quantization
TARGETS = ["Linear"]
ignore  = ["embed_tokens"]

PROFILE = os.getenv("PROFILE", "balanced")  # speed / balanced / safe
GROUP_SIZE = int(os.getenv("GROUP_SIZE", "128"))
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQ", "1536"))

# ------------------------------
# 중복제거
# ------------------------------
import unicodedata
import re  # ✅ dedup 함수 위로 올리기
from typing import List, Optional

def _normalize_text_for_dedup(s: str) -> str:
    """중복 판정을 위한 정규화: 공백/유니코드/케이스 최소 정리"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)  # 모든 공백을 1칸으로
    return s

def dedup_dataset_by_text(ds, text_col: str = "text", mode: str = "exact", ngram: int = 5) :
    """
    mode="exact": 완전 중복 제거
    mode="weak_near": 매우 가벼운 near-dedup (ngram fingerprint 기반, 빠르고 의존성 없음)
    """
    if ds is None or len(ds) == 0:
        return ds

    texts: List[str] = ds[text_col]
    seen = set()
    keep_idx = []

    for i, t in enumerate(texts):
        nt = _normalize_text_for_dedup(t)

        if mode == "exact":
            # ✅ (4번) 너무 짧은 텍스트는 과도하게 합쳐지지 않게 키를 강화
            if len(nt) < 50:
                key = (nt, len(t or ""), i % 7)  # i%7은 동일 짧은 문장이 너무 뭉치는 걸 완화(약한 분산)
            else:
                key = nt

        elif mode == "weak_near":
            # 약한 near-dedup: 문자 ngram들의 해시 일부를 fingerprint로 사용
            # (완벽하진 않지만 “거의 같은 템플릿/문장” 중복을 꽤 줄여줌)
            if len(nt) <= ngram:
                key = nt
            else:
                grams = [nt[j:j+ngram] for j in range(0, len(nt) - ngram + 1, ngram)]
                # fingerprint: ngram 해시의 상위 일부만 모아 키 구성
                fp = tuple(sorted(hash(g) for g in grams)[:16])
                key = fp
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)

    if len(keep_idx) < len(ds):
        print(f"[INFO][DEDUP] {len(ds)} -> {len(keep_idx)} (mode={mode})")
    return ds.select(keep_idx)


def safe_select_first(ds, n: int, *, seed: int = 42):
    """ds 길이가 n보다 짧아도 절대 안 터지게 앞에서 n개(or 전부)만 선택"""
    if ds is None or len(ds) == 0:
        return ds
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))

def fill_to_n(ds_current, ds_pool, n: int, *, text_col="text", seed: int = 123, dedup_mode: str = "exact"):
    """
    ds_current가 n보다 짧으면 ds_pool에서 추가로 가져와 dedup 유지하며 채움.
    - ds_pool: general_pool처럼 충분히 큰 풀을 넣으면 됨
    - 절대 에러 안 나게 구성
    """
    if ds_current is None:
        return ds_current
    if len(ds_current) >= n:
        return safe_select_first(ds_current, n, seed=seed)

    need = n - len(ds_current)

    # pool에서 여유 있게 뽑기 (need * 5 정도)
    take = min(len(ds_pool), max(need * 5, need))
    extra = ds_pool.shuffle(seed=seed).select(range(take))

    merged = concatenate_datasets([ds_current, extra])
    merged = dedup_dataset_by_text(merged, text_col=text_col, mode=dedup_mode)

    # 그래도 부족할 수 있으니 안전하게 마무리
    return safe_select_first(merged, n, seed=seed + 1)


# ------------------------------
# Calibration formatting policy
# ------------------------------
# 0.0 ~ 1.0 : Math/Code에서 teacher-forcing(정답 포함) 샘플을 섞고 싶으면 >0으로
TEACHER_FORCING_RATIO = 0.0 # 추천 시작: 0.0 (완전 prompt-only)
# 예: 0.3 으로 두면 30%는 user+assistant(answer) + add_generation_prompt=False로 구성

def _apply_template_and_count_tokens(conversations, add_generation_prompt: bool):
    text = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=add_generation_prompt,
        tokenize=False
    )
    # 길이 객관화: 너무 긴 샘플이 truncate로 왜곡되지 않도록 토큰 길이도 같이 계산
    n_tokens = len(tokenizer(text, add_special_tokens=False).input_ids)
    return {"text": text, "n_tokens": n_tokens}

def _deterministic_pick_teacher_forcing(idx: int, ratio: float) -> bool:
    """재현성(=객관성) 위해 랜덤 대신 index 기반으로 비율 결정"""
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    # 예: ratio=0.3 -> idx % 10 < 3 이면 teacher forcing
    bucket = int(round(ratio * 10))
    bucket = max(0, min(10, bucket))
    return (idx % 10) < bucket

print("[INFO] 모델 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=HF_TOKEN,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)

print("[INFO] 모델/토크나이저 로드 완료")

print("[INFO] 다양한 도메인의 캘리브레이션 데이터 혼합 중...")
print("[INFO] 캘리브레이션 데이터 로드 중...")

# -----------------------
# A. General (MANTA-1M)
# -----------------------
print("[INFO] Loading General pool...")

# 1. 충분히 큰 Pool 확보 (예: NUM_GENERAL의 20배)
ds_general_pool = load_dataset(
    "LGAI-EXAONE/MANTA-1M",
    split="train",
    token=HF_TOKEN
).shuffle(seed=42).select(range(max(SHUFFLE_POOL, NUM_GENERAL * 20)))

def format_general(example):
    return _apply_template_and_count_tokens(
        example["conversations"],
        add_generation_prompt=True
    )

ds_general_pool = ds_general_pool.map(format_general, remove_columns=ds_general_pool.column_names)
ds_general_pool = ds_general_pool.filter(lambda x: x["n_tokens"] <= MAX_SEQUENCE_LENGTH)

import re
hangul = re.compile(r"[가-힣]")

NUM_GENERAL_KO = 44     # general 중 한국어/혼합 강제 포함 개수
NUM_GENERAL_LONG = 22  # general 중 긴 샘플 강제 포함 개수

# 2. Pool 전체에서 조건에 맞는 데이터 필터링
ds_ko = ds_general_pool.filter(lambda x: hangul.search(x["text"]) is not None)
ds_long = ds_general_pool.filter(lambda x: x["n_tokens"] >= int(0.75 * MAX_SEQUENCE_LENGTH))

ko_n = min(NUM_GENERAL_KO, len(ds_ko))
long_n = min(NUM_GENERAL_LONG, len(ds_long))
rest_n = max(0, NUM_GENERAL - ko_n - long_n)

# 3. 필요한 개수만큼 추출하여 최종 병합
ds_general_final = concatenate_datasets([
    ds_ko.shuffle(seed=42).select(range(ko_n)) if ko_n > 0 else ds_general_pool.select([]),
    ds_long.shuffle(seed=43).select(range(long_n)) if long_n > 0 else ds_general_pool.select([]),
    ds_general_pool.shuffle(seed=44).select(range(rest_n)) if rest_n > 0 else ds_general_pool.select([]),
]).shuffle(seed=45)
ds_general_final = dedup_dataset_by_text(ds_general_final, mode="exact")

ds_general_final = fill_to_n(
    ds_current=ds_general_final,
    ds_pool=ds_general_pool,
    n=NUM_GENERAL,
    text_col="text",
    seed=99,
    dedup_mode="exact",
)

ds_general = safe_select_first(ds_general_final, NUM_GENERAL, seed=45)

print(f"[INFO] General ready: {len(ds_general)}")

# ---------------------------------------------------------
# B. Math 데이터 (OpenAI의 GSM8K)
# ---------------------------------------------------------
print("[INFO] Loading Math...")
ds_math = load_dataset(
    "gsm8k",
    "main",
    split="train"
).shuffle(seed=42).select(range(max(SHUFFLE_POOL, NUM_MATH)))

import hashlib

def pick_teacher_forcing(text: str, ratio: float) -> bool:
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF  # 0~1
    return v < ratio

def format_math(example, idx):
    prompt = example["question"].strip()
    use_tf = pick_teacher_forcing(prompt, TEACHER_FORCING_RATIO)

    if use_tf:
        conversations = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["answer"].strip()},
        ]
        return _apply_template_and_count_tokens(conversations, add_generation_prompt=False)
    else:
        conversations = [
            {"role": "user", "content": prompt},
        ]
        return _apply_template_and_count_tokens(conversations, add_generation_prompt=True)

ds_math = ds_math.map(format_math, with_indices=True, remove_columns=ds_math.column_names)
ds_math = ds_math.filter(lambda x: x["n_tokens"] <= MAX_SEQUENCE_LENGTH)
ds_math = dedup_dataset_by_text(ds_math, mode="exact")  # ✅ 추가
ds_math = safe_select_first(ds_math, NUM_MATH, seed=42)
print(f"[INFO] Math ready: {len(ds_math)}")


# ---------------------------------------------------------
# C. Code 데이터 (CodeAlpaca-20k)
# ---------------------------------------------------------
print("[INFO] Loading Code...")
ds_code = load_dataset(
    "sahil2801/CodeAlpaca-20k",
    split="train"
).shuffle(seed=42).select(range(max(SHUFFLE_POOL, NUM_CODE)))



def format_code(example, idx):
    prompt = (example["instruction"] or "").strip()
    user_in = (example["input"] or "").strip()
    if user_in:
        prompt += f"\n\nInput:\n{user_in}"

    use_tf = pick_teacher_forcing(prompt, TEACHER_FORCING_RATIO)
    if use_tf:
        conversations = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": (example["output"] or "").strip()},
        ]
        return _apply_template_and_count_tokens(conversations, add_generation_prompt=False)
    else:
        conversations = [
            {"role": "user", "content": prompt},
        ]
        return _apply_template_and_count_tokens(conversations, add_generation_prompt=True)

ds_code = ds_code.map(format_code, with_indices=True, remove_columns=ds_code.column_names)
ds_code = ds_code.filter(lambda x: x["n_tokens"] <= MAX_SEQUENCE_LENGTH)
ds_code = dedup_dataset_by_text(ds_code, mode="exact")  # ✅ 추가
ds_code = safe_select_first(ds_code, NUM_CODE, seed=42)
print(f"[INFO] Code ready: {len(ds_code)}")


# ---------------------------------------------------------
# D. Spanish 데이터 (bertin-project/alpaca-spanish)
# ---------------------------------------------------------
def build_spanish_ds():
    if NUM_ES <= 0:
        return None

    print("[INFO] Loading Spanish...")
    ds_es = load_dataset("FreedomIntelligence/sharegpt-spanish", split="train")
    print(f"[INFO] {ds_es} Spanish Dataset")

    # 필터/길이컷 때문에 pool을 좀 크게
    pool_n = max(SHUFFLE_POOL, NUM_ES * 50)
    pool_n = min(pool_n, len(ds_es))
    ds_es = ds_es.shuffle(seed=123).select(range(pool_n))

    cols = set(ds_es.column_names)

    if "conversations" in cols:
    # ShareGPT류: conversations = [{"from":"human","value":...}, {"from":"gpt","value":...}, ...]
        def _extract_first_turn(conv_list):
            if not conv_list:
                return ("Hola", "")
            user_msg = ""
            asst_msg = ""
            for t in conv_list:
                frm = str(t.get("from") or "").lower()
                val = (t.get("value") or "").strip()
                if not val:
                    continue
                if (not user_msg) and frm in {"human", "user"}:
                    user_msg = val
                elif user_msg and (not asst_msg) and frm in {"gpt", "assistant"}:
                    asst_msg = val
                    break
            if not user_msg:
                user_msg = "Hola"
            return (user_msg, asst_msg)

        # ⚠️ lang 메타가 'en'으로 찍히는 샘플이 보여서(lang 메타 신뢰도 낮음),
        #     실험 통제 목적이면 텍스트 기반으로 스페인어를 거르는 게 안전함. :contentReference[oaicite:4]{index=4}
        #     (최소 휴리스틱: 스페인어 특수문자/빈출 단어)
        es_hint = re.compile(r"[ñáéíóúü¡¿]|(\b(el|la|de|que|y|para|por|con|una|un)\b)", re.IGNORECASE)

        def format_es_sharegpt(example, idx):
            conv = example.get("conversations") or []
            user_msg, asst_msg = _extract_first_turn(conv)

            # 스페인어 여부(가벼운 필터) - 7개만 쓸 거라 과하게 엄격할 필요는 없음
            # 필터를 map 안에서 하려면, 빈 prompt로 떨어뜨리는 것보다 아래처럼 "Hola"로 최소화하는 게 안전
            if not es_hint.search(user_msg):
                # 완전 배제 대신, filter 단계에서 걸러지게 n_tokens 크게 만들거나,
                # 그냥 그대로 두고 후단에서 shuffle+select로 7개 뽑아도 됨.
                pass

            use_tf = pick_teacher_forcing(user_msg, TEACHER_FORCING_RATIO)
            if use_tf and asst_msg:
                conversations = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ]
                return _apply_template_and_count_tokens(conversations, add_generation_prompt=False)
            else:
                conversations = [{"role": "user", "content": user_msg}]
                return _apply_template_and_count_tokens(conversations, add_generation_prompt=True)

        ds_es = ds_es.map(format_es_sharegpt, with_indices=True, remove_columns=ds_es.column_names)
        ds_es = ds_es.filter(lambda x: es_hint.search(x["text"]) is not None)
        ds_es = ds_es.filter(lambda x: x["n_tokens"] <= MAX_SEQUENCE_LENGTH)
        ds_es = dedup_dataset_by_text(ds_es, mode="exact")
        ds_es = ds_es.shuffle(seed=124)
        ds_es = safe_select_first(ds_es, NUM_ES, seed=42)
        print(f"[INFO] Spanish ready: {len(ds_es)} (requested {NUM_ES})")
        return ds_es

    # language 필터(가능하면)
    if "lang" in cols:
        ds_es = ds_es.filter(lambda x: str(x.get("lang") or "").startswith("es"))
    elif "language" in cols:
        ds_es = ds_es.filter(lambda x: str(x.get("language") or "").startswith("es"))

    # 스키마 자동 감지:
    #  - oasst1류: text/role/lang 형태
    #  - alpaca류: instruction/input/output 형태
    if ("text" in cols) and ("role" in cols):
        # OpenAssistant/oasst1 같은 경우: prompter/user만 prompt-only로 사용 (안전/간단)
        allow_roles = {"prompter", "user", "human"}
        ds_es = ds_es.filter(lambda x: str(x.get("role") or "").lower() in allow_roles)

        def format_es_oasst(example, idx):
            prompt = (example.get("text") or "").strip()
            if not prompt:
                prompt = "Hola"  # 빈 샘플 방지용 최소값
            conversations = [{"role": "user", "content": prompt}]
            return _apply_template_and_count_tokens(conversations, add_generation_prompt=True)

        ds_es = ds_es.map(format_es_oasst, with_indices=True, remove_columns=ds_es.column_names)

    else:
        # alpaca-spanish 류
        def format_es_alpaca(example, idx):
            inst = (example.get("instruction") or example.get("prompt") or example.get("question") or "").strip()
            inp  = (example.get("input") or example.get("context") or "").strip()
            out  = (example.get("output") or example.get("response") or example.get("answer") or "").strip()

            prompt = inst if inst else inp
            if inst and inp:
                prompt = f"{inst}\n\nInput:\n{inp}"
            if not prompt:
                prompt = "Hola"

            use_tf = pick_teacher_forcing(prompt, TEACHER_FORCING_RATIO)
            if use_tf and out:
                conversations = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": out},
                ]
                return _apply_template_and_count_tokens(conversations, add_generation_prompt=False)
            else:
                conversations = [{"role": "user", "content": prompt}]
                return _apply_template_and_count_tokens(conversations, add_generation_prompt=True)

        ds_es = ds_es.map(format_es_alpaca, with_indices=True, remove_columns=ds_es.column_names)

    ds_es = ds_es.filter(lambda x: x["n_tokens"] <= MAX_SEQUENCE_LENGTH)
    ds_es = dedup_dataset_by_text(ds_es, mode="exact")  # ✅ 추가
    ds_es = ds_es.shuffle(seed=124)
    ds_es = safe_select_first(ds_es, NUM_ES, seed=42)
    print(f"[INFO] Spanish ready: {len(ds_es)} (requested {NUM_ES})")
    return ds_es

ds_es = build_spanish_ds()


# ---------------------------------------------------------
# E. 데이터셋 병합 및 최종 셔플링
# ---------------------------------------------------------
# 3개의 데이터셋을 하나로 합치고 (총 512개), 순서가 편향되지 않도록 섞어줍니다.
parts = [ds_general, ds_math, ds_code]
if ds_es is not None:
    parts.append(ds_es)

mixed_ds = concatenate_datasets(parts).shuffle(seed=999)

print(f"[INFO] 혼합 완료: General({NUM_GENERAL}) + Math({NUM_MATH}) + Code({NUM_CODE}) + ES({NUM_ES}) = 총 {len(mixed_ds)}개")
print("[INFO] length stats:", mixed_ds["n_tokens"][:10])

# 안전하게 text만 남김
if "n_tokens" in mixed_ds.column_names:
    mixed_ds = mixed_ds.remove_columns(["n_tokens"])

print("[INFO] 데이터 전처리 완료")

print(f"[INFO] W8A8 시작 (samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})...")

ignore_layers = ["embed_tokens", "lm_head"]

print(f"ignore_layers: {ignore_layers}")

recipe = [
    QuantizationModifier(
        targets=["Linear"],
        scheme="FP8", # Static FP8 W8A8 양자화 적용, 즉 weight과 input activation을 모두 8-bit로 scaling 처리
        ignore=ignore_layers
    )
]

oneshot(
    model=model,
    dataset=mixed_ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] W8A8 완료")

os.makedirs(OUT_DIR, exist_ok=False)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

zip_name = ZIP_NAME
print(f"[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=str(BASE_DIR/zip_name),
    format="zip",
    root_dir=str(BASE_DIR),
    base_dir="model",
)

print(f"[INFO] 생성 완료: {zip_name}.zip")
