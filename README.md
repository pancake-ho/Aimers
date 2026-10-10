# LG Aimers 8기 | EXAONE 4.0 1.2B 경량화

**Static FP8 W8A8 양자화와 다중 도메인 Calibration 데이터 구성**

LG Aimers 8기 해커톤에서 EXAONE 4.0 1.2B 모델의 추론 효율을 개선하기 위해 진행한 모델 경량화 프로젝트입니다. 최종 구현은 **가중치(Weight)와 입력 활성화값(Activation)을 모두 FP8로 양자화**하고, 일반 대화·수학·코드·스페인어 데이터를 혼합해 Calibration을 수행하는 방식입니다.

> **프로젝트 핵심:** 데이터 구성 → 양자화 → 모델 패키징 → 평가까지 연결되는 파이프라인을 구현하고, 실제 저장된 모델의 양자화 설정을 검증했습니다. 추론 속도 및 정확도 향상에 대한 최종 정량 비교는 아직 재현 검증되지 않았습니다.

## 1. 프로젝트 목표

LLM을 제한된 GPU 메모리와 연산 자원에서 활용하려면 모델의 메모리 요구량과 추론 비용을 고려해야 합니다. 반면 낮은 정밀도로 변환하는 과정에서는 모델의 출력 품질이 달라질 수 있습니다.

이 프로젝트에서는 **EXAONE 4.0 1.2B의 정확도와 추론 효율 사이의 균형을 탐색**하기 위해 다음 작업에 집중했습니다.

- **모델 경량화:** Post-Training Quantization(PTQ) 기반 FP8 W8A8 적용
- **Calibration 데이터 설계:** 다양한 언어 및 태스크를 반영한 데이터 혼합·전처리
- **실행 및 평가 자동화:** Slurm 기반 작업 실행, vLLM 추론 속도 및 `lm-eval` 평가 코드 구성
- **재현성·안전성:** 샘플링 seed 관리, 산출물 설정 확인, 기존 결과 덮어쓰기 방지

## 2. 전체 구현 흐름

```text
EXAONE 4.0 1.2B (원본 모델)
          │
          ▼
Calibration 데이터 구성
General (95) + Math (77) + Code (77) + Spanish (7)
          │
          ▼
Chat Template 적용 → 길이 필터링 → 중복 완화 → Shuffle
          │
          ▼
LLM Compressor · QuantizationModifier(scheme="FP8")
          │
          ▼
Static FP8 W8A8 모델 저장 (compressed-tensors)
          │
          ▼
모델 ZIP 패키징 → vLLM / lm-eval 평가 코드
```

괄호 안 숫자는 **목표 샘플 수**입니다. 중복 제거와 필터링 이후 실제 확보 수량은 별도 실행 로그로 확인해야 합니다.

## 3. 핵심 구현

### 3.1 Static FP8 W8A8 양자화

[최종 양자화 코드](model/baseline_lm_w8a8_spanish.py)에서 `LLM Compressor`의 `QuantizationModifier`와 `oneshot`을 사용합니다. 실제 최종 모델 ZIP의 `config.json`에서 다음 설정을 확인했습니다.

| 항목 | 최종 모델 설정 |
|---|---|
| 대상 모델 | `LGAI-EXAONE/EXAONE-4.0-1.2B` |
| 방식 | PTQ · `scheme="FP8"` |
| Weight | FP8 · 8-bit · Static · Per-tensor |
| Input activation | FP8 · 8-bit · Static · Per-tensor |
| 양자화 대상 | `Linear` 계층 |
| 저장 포맷 | `compressed-tensors` · `float-quantized` |
| KV-cache 양자화 | 별도 적용하지 않음 |

**Static FP8 W8A8은 INT8 W8A8 또는 `FP8_DYNAMIC`과 다릅니다.** 현재 저장된 모델 메타데이터에는 Weight와 Input activation 모두 `dynamic: false`로 명시되어 있습니다.

### 3.2 다중 도메인 Calibration 데이터

단일 도메인 데이터만 사용하는 대신, 일반 대화 외에 수학·코드·스페인어 프롬프트를 포함하도록 데이터를 구성했습니다.

| 영역 | 데이터셋 | 목표 수량 |
|---|---|---:|
| General | `LGAI-EXAONE/MANTA-1M` | 95 |
| Math | `gsm8k` (`main`) | 77 |
| Code | `sahil2801/CodeAlpaca-20k` | 77 |
| Spanish | `FreedomIntelligence/sharegpt-spanish` | 7 |
| **합계** | | **256** |

주요 전처리 방식은 **Chat Template 변환**, 최대 시퀀스 길이 필터링(기본 1,536 tokens), 텍스트 정규화 기반 중복 완화, 고정 seed를 이용한 셔플 및 샘플 선택입니다. General 데이터에는 한국어와 상대적으로 긴 문맥의 예제를 우선 포함하도록 선택 로직을 구성했습니다. 최종 설정의 `TEACHER_FORCING_RATIO`는 `0.0`으로, 수학·코드 데이터는 기본적으로 prompt-only 형식으로 준비합니다.

데이터셋의 정확한 revision과 필터링 후 수량은 아직 별도 고정·검증하지 않았으므로, 모든 실행에서 완전히 동일한 샘플이 확보된다고 주장하지 않습니다.

### 3.3 평가 및 실험 관리

[평가 코드](model/test_speed.py)는 원본 모델과 양자화 모델을 비교하기 위한 기능을 포함합니다.

- **추론 속도:** vLLM 기반 생성 토큰 수·경과 시간·tokens/s 측정 코드
- **출력 품질:** `lm-eval` 기반 태스크 평가 코드(기본 태스크: GSM8K)
- **실험 환경:** Seraph GPU 서버의 Slurm 작업 스크립트 제공
- **산출물 관리:** 기존 출력 디렉터리 및 ZIP을 덮어쓰지 않도록 보호

> 평가 기능의 **구현**과 실제 **성능 향상 검증**은 구분합니다. 동일 GPU·vLLM 커널·프롬프트·생성 설정에서의 정확도 및 지연 시간 비교 수치는 아직 README에 확정 결과로 기재하지 않았습니다.

## 4. 검증 결과 및 한계

| 구분 | 확인된 내용 |
|---|---|
| 소스 코드 | Python 구문 검사 통과 |
| 실행 환경 | CUDA 인식 및 주요 패키지 확인 |
| 최종 산출물 | 약 1.5GB ZIP 존재, `config.json`·`model.safetensors` 확인 |
| 양자화 설정 | 최종 산출물에서 Static FP8 W8A8 메타데이터 확인 |
| 추론 성능 | Native FP8 W8A8 실행 및 속도·정확도 재현은 **미검증** |

검증에 사용한 서버 GPU는 **NVIDIA RTX 3090 (SM 8.6)** 입니다. 해당 GPU에서의 CUDA 인식과 산출물 메타데이터 검증은 성공했지만, 이를 **native FP8 W8A8 추론 가속의 검증 결과로 해석해서는 안 됩니다.** 실제 가속 성능은 지원 GPU와 선택된 추론 커널을 확인한 환경에서 평가해야 합니다.

## 5. 실행 방법

최종 구현은 Python 3.10.19, PyTorch 2.6.0+cu118, Transformers 4.57.3, Datasets 4.4.1, LLM Compressor 0.9.0.2, vLLM 0.14.1 환경에서 정적 점검했습니다. **현재 저장소의 공통 `requirements.txt`와 실제 검증 환경이 완전히 동일하지 않으므로, 새 환경 설치 및 전체 재현은 미검증**입니다.

```bash
# 저장소 준비
git clone https://github.com/pancake-ho/Aimers.git
cd Aimers
mkdir -p logs

# Seraph 환경에서 최종 양자화 작업 제출
sbatch scripts/seraph_w8a8_spanish.sbatch
```

실행 전 `aimers` Conda 환경, 모델·데이터셋 접근 권한, `HF_TOKEN` 환경변수 또는 `model/.env` 설정이 필요합니다. **토큰을 Git에 커밋하면 안 됩니다.** 기존 모델 출력이나 제출 ZIP이 존재하는 경우 스크립트는 덮어쓰지 않고 중단합니다.

최종 제출 ZIP 이름은 `submit_lm_w8a8_spanish_newdata.zip`입니다. 모델 가중치와 ZIP은 Git 저장소에 포함하지 않았습니다.

## 6. 저장소 주요 파일

| 경로 | 역할 |
|---|---|
| [`model/baseline_lm_w8a8_spanish.py`](model/baseline_lm_w8a8_spanish.py) | 최종 데이터 구성 및 FP8 W8A8 양자화 |
| [`model/test_speed.py`](model/test_speed.py) | vLLM 속도·`lm-eval` 정확도 평가 코드 |
| [`scripts/seraph_w8a8_spanish.sbatch`](scripts/seraph_w8a8_spanish.sbatch) | Slurm 양자화 실행 스크립트 |
| [`docs/W8A8_FINAL.md`](docs/W8A8_FINAL.md) | 환경·설정·검증 범위 상세 기록 |
| `model/main.py`, `model/quantizing/` | 과거 AutoRound 중심 실험 코드(최종 W8A8 경로와 구분) |

## 7. 프로젝트를 통해 다룬 기술적 과제

이번 프로젝트에서는 **양자화 설정 자체뿐 아니라 결과를 재현하고 해석할 수 있는 실험 파이프라인**에 중점을 두었습니다. 데이터 도메인 비율 설계, 길이 및 중복 처리, Static/Dynamic 양자화 방식의 구분, 압축 모델 메타데이터 점검, Slurm 실행과 모델 산출물 보호를 함께 다뤘습니다.

추가로 검증할 과제는 **지원 GPU에서의 native FP8 추론**, **원본 대비 정확도·추론 성능 비교**, **데이터셋 revision 고정**입니다.

---

이 README는 **최종 W8A8 구현 기준**으로 작성되었습니다. 이전 AutoRound·LoRA 관련 시도는 저장소의 과거 개발 이력에 해당하며, 최종 모델의 양자화 방식과 혼동하지 않도록 구분했습니다.
