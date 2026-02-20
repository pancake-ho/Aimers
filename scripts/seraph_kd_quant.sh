#!/usr/bin/bash
# Seraph batch template for KD -> Quant pipeline
# IMPORTANT:
# - Do NOT run heavy training/quant jobs on master node.
# - Submit this file via sbatch on a GPU partition.

#SBATCH -J aimers_pancake
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-y6
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out


set -euo pipefail

if [[ -z "${PARTITION:-}" ]]; then
  echo "[ERROR] PARTITION env is required."
  echo "[ERROR] Example: PARTITION=batch_eebme_ugrad sbatch -p batch_eebme_ugrad scripts/seraph_kd_quant.sbatch"
  exit 2
fi

if [[ "${SLURM_JOB_PARTITION:-}" != "${PARTITION}" ]]; then
  echo "[WARN] PARTITION=${PARTITION}, but allocated partition is ${SLURM_JOB_PARTITION:-unknown}."
  echo "[WARN] Submit with: sbatch -p ${PARTITION} scripts/seraph_kd_quant.sbatch"
fi

REPO_DIR="${REPO_DIR:-/data/$USER/repos/aimers}"
cd "${REPO_DIR}"

mkdir -p logs

# Keep cache/data IO on local disk to reduce NAS pressure.
export LOCAL_DATA_ROOT="/local_datasets/$USER"
export HF_HOME="${LOCAL_DATA_ROOT}/hf_home"
export HF_DATASETS_CACHE="${LOCAL_DATA_ROOT}/hf_datasets"
export TRANSFORMERS_CACHE="${LOCAL_DATA_ROOT}/hf_transformers"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ -f "/data/$USER/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "/data/$USER/anaconda3/etc/profile.d/conda.sh"
fi
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  conda activate "${CONDA_ENV_NAME}"
fi

STUDENT_MODEL="${STUDENT_MODEL:-./base_model}"
BASE_MODEL="${BASE_MODEL:-./base_model}"
TEACHER_MODEL="${TEACHER_MODEL:-}"
KD_NUM_SAMPLES="${KD_NUM_SAMPLES:-5000}"
KD_MAX_NEW_TOKENS="${KD_MAX_NEW_TOKENS:-256}"
OUT_DIR="${OUT_DIR:-./model}"
QUANT_METHOD="${QUANT_METHOD:-gptq}"
ZIP_NAME="${ZIP_NAME:-submit}"
SEED="${SEED:-42}"

echo "[INFO] partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[INFO] repo_dir=$(pwd)"
echo "[INFO] hf_home=${HF_HOME}"
echo "[INFO] datasets_cache=${HF_DATASETS_CACHE}"
echo "[INFO] transformers_cache=${TRANSFORMERS_CACHE}"

python -V

python model/pipeline.py \
  --student_model "${STUDENT_MODEL}" \
  --base_model "${BASE_MODEL}" \
  --teacher_model "${TEACHER_MODEL}" \
  --kd_num_samples "${KD_NUM_SAMPLES}" \
  --kd_max_new_tokens "${KD_MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  --quant_method "${QUANT_METHOD}" \
  --out_dir "${OUT_DIR}" \
  --zip_name "${ZIP_NAME}" \
  --run_vllm_smoke

echo "[INFO] pipeline completed"
echo "[INFO] check outputs: ${OUT_DIR}, ${ZIP_NAME}.zip"

