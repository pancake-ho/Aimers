# Score Formation Report (job.sh 기준)

## 1) 점검 기준
- 기준 프로파일: `model/job.sh` (`--do_kd`, `--quant_method gptq`, `--search_budget 6`, `--do_lora` 미사용)
- 기준 날짜/컷라인: 2026-02-17 스냅샷, 100등 `0.61214`, 1등 `0.64701`

## 2) 실제 실행 시도 결과
### 2.1 원본 job 인자 실행
- 명령: `python main.py --quant_method gptq --do_kd --search_budget 6 ...`
- 결과: 실패
- 실패 원인: 네트워크 차단으로 `huggingface.co` 이름 해석 실패

### 2.2 오프라인 우회 후 동일 프로파일 재시도
- 우회:
  - `--base_model`을 로컬 HF snapshot 경로로 지정
  - `HF_DATASETS_CACHE=/tmp/hf_datasets`로 캐시 lock 권한 우회
  - `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`
- 결과: base eval 단계에서 CPU 환경 병목(장시간)으로 중단

### 2.3 축소 러닝(동일 파이프라인 구조, 작은 예산)
- 명령: `--do_kd --quant_method gptq --search_budget 1 --kd_steps 2 --kd_samples 64 --calib_samples 16 --eval_count 8 ...`
- 결과: KD 진입 전후 불안정 종료
  - 1차: `bf16/gpu` 요구 오류 (CPU 미지원)
  - 2차(코드 fallback 적용 후): 프로세스 종료 코드 `-1` (리소스/런타임 제한 추정)

## 3) 점수 형성 관점 결론
- 현재 환경에서는 `artifacts/metrics.json`, `leaderboard_ready_report.json`, `submit.zip` 생성이 완료되지 않아 **단계별 수치 분해(base→kd→quant→final)를 확정할 수 없음**.
- 따라서 아래 값은 본 점검에서 산출 불가:
  - `final_perplexity`, `final_tokens_per_sec`, `lb_proxy_score`
  - KD/Quant 단계별 기여량

## 4) 왜 점수 확정이 안 되었는가
- 서버 채점과 다른 실행 환경:
  - 서버: GPU + vLLM 고정 설정 + 20분 제한
  - 현재: GPU 없음(`cuda available: False`), 로컬 CPU only
- 결과적으로, job.sh 기준 학습/양자화 파이프라인을 완료하지 못해 제출 산출물 부재

## 5) 재실행 권장 조건 (점수 확정용)
- GPU 노드에서 재실행(최소 1x GPU)
- 동일 인자 유지:
  - `--do_kd --quant_method gptq --search_budget 6`
- 실행 완료 후 아래 파일로 점수 형성 분해:
  - `artifacts/metrics.json`
  - `artifacts/leaderboard_ready_report.json`
  - `artifacts/submit.zip`
