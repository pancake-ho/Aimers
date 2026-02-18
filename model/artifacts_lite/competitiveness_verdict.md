# Competitiveness Verdict (Phase3 컷 기준)

## 기준
- 컷라인(2026-02-17 스냅샷): 100등 `0.61214`
- 안정권 목표: `Public Score >= 0.615`

## 현재 판정
- **위험 (High Risk)**

## 근거
1. `job.sh` 기준 파이프라인 완주 실패로 `submit.zip` 미생성
2. 단계별 메트릭(`metrics.json`) 부재로 점수 기여도 분해 불가
3. vLLM smoke 실패(`Device string must not be empty`), 즉 채점 경로 재현 미완료
4. GPU 없는 환경에서 KD+GPTQ 실행 시간이 실질적으로 계획 범위를 초과

## 판정 로직 적용
- 규칙: `Public Score >= 0.615`가 2회 이상이면 확보
- 현재: 제출/점수 자체가 생성되지 않았으므로 규칙 평가 불가
- 따라서 보수적으로 `위험`으로 분류

## 컷 경쟁력 확보를 위한 최소 조건
1. GPU 환경에서 `job.sh` 프로파일 1회 완주
2. `submit.zip` 생성 + `test.py --mode full` 통과
3. 동일 산출물 2~3회 제출 후 점수 분산 확인
4. 최소 2회 `>=0.615` 달성 확인
