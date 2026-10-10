# Aimers 8th — Final Static FP8 W8A8

## 1. Objective

Quantization of LGAI-EXAONE/EXAONE-4.0-1.2B
using calibration-data mixture with Spanish data.

## 2. Final Implementation

- Quantization script: model/baseline_lm_w8a8_spanish.py
- Benchmark script: model/test_speed.py
- Method: LLM Compressor QuantizationModifier
- Scheme: FP8
- Targets: Linear
- Format: compressed-tensors / float-quantized
- Weight: FP8, static, per-tensor
- Input activation: FP8, static, per-tensor
- KV-cache quantization: not configured

## 3. Calibration Data

Target calibration samples: 256

- General: 95
- Math: 77
- Code: 77
- Spanish: 7

Known sources:
- LGAI-EXAONE/MANTA-1M
- FreedomIntelligence/sharegpt-spanish

The complete dataset selection, deduplication,
templating, and shuffle logic is preserved in
model/baseline_lm_w8a8_spanish.py.

Actual post-dedup sample counts and dataset
revisions require further verification.

## 4. Verified Environment

- Python: 3.10.19
- PyTorch: 2.6.0+cu118
- Transformers: 4.57.3
- Datasets: 4.4.1
- LLM Compressor: 0.9.0.2
- python-dotenv: 1.2.1
- vLLM: 0.14.1
- lm-eval: 0.4.11
- Verification GPU: RTX 3090 (SM 8.6)

## 5. Final Artifact

Expected ZIP filename:
submit_lm_w8a8_spanish_newdata.zip

Original archived artifact was verified to contain:
- config.json
- model.safetensors
- tokenizer files
- recipe.yaml

The model ZIP is stored separately and is not
committed to GitHub.

## 6. Validation Status

Passed:
- Python source compilation
- CUDA visibility
- Package discovery
- Existing model ZIP metadata inspection
- Static FP8 W8A8 configuration verification

Not yet verified:
- Full inference on supported FP8 W8A8 hardware
- Native FP8 W8A8 acceleration
- Final accuracy / throughput reproduction
- Exact calibration dataset revisions
- Fresh-environment dependency installation

RTX 3090 does not support native FP8 W8A8
acceleration under vLLM 0.14.1.

## 7. Reproduction Notes

The quantization script requires HF_TOKEN.
Do not commit .env files or authentication tokens.

Existing artifacts are protected against overwrite.
The final script does not automatically delete
previously generated model directories.

The historical AutoRound implementation remains
separate from this final W8A8 implementation.
