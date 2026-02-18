# Aimers 8th: EXAONE 4.0 1.2B Optimization Pipeline

파이프라인 단계:

`load -> (optional KD) -> (optional LoRA) -> quantize(gptq/autoround) -> eval -> package -> rehearsal`

## Requirements

1. Python 3.10+
2. GPU 환경(BF16 권장)
3. PyTorch CUDA wheel 먼저 설치(환경에 맞게)
4. 의존성 설치

```bash
pip install -r requirements.txt
```

## 실행 예시

```bash
cd model
python3 main.py \
  --base_model LGAI-EXAONE/EXAONE-4.0-1.2B \
  --quant_method gptq \
  --do_kd \
  --do_lora \
  --search_budget 6 \
  --selection_metric lb_proxy \
  --score_perf_weight 0.5 \
  --score_speed_weight 0.5 \
  --eval_dataset_id LGAI-EXAONE/MANTA-1M \
  --eval_dataset_split train \
  --eval_start 200000 \
  --eval_count 128 \
  --report_path metrics.csv \
  --out_dir ./artifacts
```

## 주요 인자

- `--quant_method {gptq,autoround}`: 양자화 방식
- `--do_kd`, `--do_lora`: KD/LoRA 활성화
- `--search_budget`: 단계별 trial 수 제한
- `--selection_metric {lb_proxy,ppl}`: trial 선택 기준 (`lb_proxy` 권장)
- `--score_perf_weight`, `--score_speed_weight`: 성능/속도 가중치
- `--eval_dataset_id`, `--eval_dataset_split`: 평가셋 소스 분리
- `--eval_start`, `--eval_count`: 고정 슬라이스 평가
- `--report_path`: 리포트 경로(`.csv` 또는 `.jsonl`)
- `--skip_rehearsal`: 제출 리허설 비활성화
- `--strict_rehearsal`: 리허설 실패 시 즉시 실패 처리

## 산출물

`--out_dir` 아래 생성:

- `model/` : 최종 모델
- `submit.zip` : 제출 zip (최상위 `model/`)
- `metrics.csv`
- `metrics.jsonl`
- `metrics.json`
- `leaderboard_ready_report.json`

## 제출 리허설

기본 동작은 저장 직후 자동 리허설:

`save -> submit.zip -> python3 test.py --mode full`

수동 실행:

```bash
python3 test.py --zip ./model/artifacts/submit.zip --mode package
python3 test.py --zip ./model/artifacts/submit.zip --mode full
```
