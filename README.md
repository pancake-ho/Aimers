# Aimers 8th: EXAONE 4.0 1.2B Model Optimization


본 리포지토리는 LG AI Research의 **EXAONE 4.0 1.2B** 모델을 경량화하기 위해, AutoRound 방식을 사용한 프로젝트입니다.  
**AutoRound** 알고리즘을 기반으로 **LoRA Fine-tuning**을 결합하여, 모델의 성능 저하를 최소화하면서 압축률을 극대화하는 파이프라인을 구축하였습니다.


## 📌 Project Overview


이 프로젝트의 목표는 제한된 리소스 환경에서도 EXAONE 모델이 높은 성능을 유지하도록 최적화하는 것입니다. 주요 접근 방식은 다음과 같습니다:

1.  **LoRA Fine-tuning (Optional):** 양자화 전, 모델이 데이터셋(MANTA-1M)의 분포를 더 잘 학습하도록 Low-Rank Adaptation(LoRA)을 적용합니다.
2.  **AutoRound Quantization:** Weight-only quantization 기법인 AutoRound를 사용하여 4-bit 양자화를 수행합니다.
3.  **GPTQ Format Export:** 최종 모델을 호환성이 높은 `auto_gptq` 포맷으로 저장 및 압축합니다.


## 🛠️ Project Structure


```bash
Aimers/
├── model/
│   ├── config/             # 모델, 학습, 양자화 설정 관리
│   ├── dataset/            # 데이터셋 로드 및 전처리 (Train/Calib 분리)
│   ├── quantizing/         # AutoRound 알고리즘 실행 모듈
│   ├── tuning/             # LoRA Fine-tuning 및 Merge 모듈
│   ├── utils/              # 결과 저장 및 압축 유틸리티
│   └── main.py             # 전체 파이프라인 실행 엔트리 포인트
├── .gitignore
└── README.md
```


## ⚙️ Requirements


이 프로젝트는 Python 3.10.12 환경에서 테스트되었습니다. 
실행 전 아래 라이브러리들을 설치해주세요.
(권장: requirements.txt 파일을 생성하여 pip install -r requirements.txt로 관리하는 것이 좋습니다.)

```bash
pip install torch transformers datasets peft trl auto-round auto-gptq
```

## 🚀 Usage


전체 파이프라인은 main.py를 통해 한 번에 실행됩니다. (우분투)
실행이 완료되면 model/ 디렉토리에 양자화된 모델 파일과 제출용 zip 파일이 생성됩니다.

```bash
(레포지토리 클론)
git clone [https://github.com/pancake-ho/Aimers.git](https://github.com/pancake-ho/Aimers.git)
cd Aimers/model

(파이프라인 실행)
python3 main.py --num_train (숫자) --tuning (True or False) --num_calib (숫자)
```

## 🧪 Methodology Details


**1. Fine-tuning Stage**

Method: LoRA (Low-Rank Adaptation)

Library: peft, trl

Details: q_proj, k_proj, v_proj 등 모듈에 어댑터를 부착하여 소규모 데이터로 빠르게 학습한 뒤, 원본 모델에 Merge합니다.


**2. Quantization Stage**

Method: AutoRound (Advanced Weight-Rounding)

Bits: 4-bit

Group Size: 128

Algorithm: Calibration 데이터를 사용하여 Layer별 최적의 Weight Rounding을 학습합니다.
