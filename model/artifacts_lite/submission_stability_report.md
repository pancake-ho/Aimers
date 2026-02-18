# Submission Stability Report

## 점검 대상
- 모델 경로: `/home/pancake_ho/.cache/huggingface/hub/models--LGAI-EXAONE--EXAONE-4.0-1.2B/snapshots/3abf2810673c7c0778df64a73c2d52eab32d91c4`
- 도구: `test.py`

## 1) Package 체크 (`--mode package`)
- 결과: **PASS**
- 확인 사항:
  - `config.json` 존재
  - 가중치 파일 존재(`model.safetensors`)
  - tokenizer 관련 파일 존재
  - 모델 디렉터리 용량: 약 `2450.91 MiB`

## 2) Config 체크 (`--mode config`)
- 결과: **PASS**
- 확인 사항:
  - JSON 파싱/NaN/Inf 이상 없음
  - `model_type`, `architectures` 키 존재

## 3) Tokenizer + Chat Template (`--mode tokenizer`)
- 결과: **PASS**
- 확인 사항:
  - `AutoTokenizer` 오프라인 로드 성공
  - `apply_chat_template` 호출 성공

## 4) vLLM Smoke (`--mode smoke --tp 1 --gpu-mem 0.85`)
- 결과: **FAIL**
- 실패 메시지:
  - `vLLM 모델 로딩 실패 ... Device string must not be empty`
- 환경 상태:
  - `cuda available: False`

## 안정성 판정
- 제출 구조/토크나이저/설정은 정상이나, **vLLM 추론 경로가 현재 환경에서 실패**.
- 즉, 현재 머신 기준으로는 리더보드 채점 경로를 재현하지 못하며, 제출 전 최종 안정성 검증이 완료되지 않음.

## 즉시 조치
1. GPU 환경에서 `test.py --mode full` 재검증
2. 동일 환경에서 `submit.zip` 대상으로 smoke 재실행
3. vLLM 로딩 실패 재발 시 모델/config/remote_code 포함 여부 재점검
