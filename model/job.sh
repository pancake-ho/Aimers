#!/usr/bin/bash

#SBATCH -J aimers_pancake
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-y6
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

set -euo pipefail

QUANT=${QUANT:-gptq}
SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-./artifacts}
BASE_MODEL=${BASE_MODEL:-LGAI-EXAONE/EXAONE-4.0-1.2B}
SELECTION_METRIC=${SELECTION_METRIC:-ppl}
PROXY_EVAL_COUNT=${PROXY_EVAL_COUNT:-64}
TOP_K=${TOP_K:-2}
STRICT=${STRICT:-1}
FAST=${FAST:-0}
MIX_CONFIGS=${MIX_CONFIGS:-}
ALLOW_LEGACY_CLI=${ALLOW_LEGACY_CLI:-0}
SYNC_BRANCH=${SYNC_BRANCH:-}
SYNC_COMMIT=${SYNC_COMMIT:-}
SYNC_STRICT=${SYNC_STRICT:-0}
REPO_DIR=${REPO_DIR:-/data/$USER/repos/aimers}
MODEL_DIR="${REPO_DIR}/model"
SCRIPT_VERSION="2026-02-20-r3"

source /data/$USER/anaconda3/etc/profile.d/conda.sh
conda activate aimers

echo "[INFO] job script version=${SCRIPT_VERSION}"
echo "[INFO] repo sync start: repo=${REPO_DIR} branch=${SYNC_BRANCH:-<current>} commit=${SYNC_COMMIT:-<branch-head>}"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[ERROR] git repo not found at ${REPO_DIR}" >&2
  exit 2
fi

cd "${REPO_DIR}"
echo "[INFO] repo before sync: pwd=$(pwd)"
git rev-parse --short HEAD || true
git branch --show-current || true

if [[ -z "${SYNC_BRANCH}" ]]; then
  SYNC_BRANCH="$(git branch --show-current || true)"
fi

REMOTE_SYNC_OK=1
if ! git fetch --all --prune; then
  REMOTE_SYNC_OK=0
  echo "[WARN] git fetch failed (remote auth/host key/network). continuing with local repo state."
  if [[ "${SYNC_STRICT}" == "1" ]]; then
    echo "[ERROR] SYNC_STRICT=1 and fetch failed; aborting." >&2
    exit 2
  fi
fi

if [[ -n "${SYNC_BRANCH}" ]]; then
  if git show-ref --verify --quiet "refs/heads/${SYNC_BRANCH}"; then
    git checkout "${SYNC_BRANCH}"
  else
    echo "[WARN] local branch '${SYNC_BRANCH}' not found; keeping current branch."
  fi
  if [[ "${REMOTE_SYNC_OK}" == "1" ]]; then
    if ! git pull --ff-only origin "${SYNC_BRANCH}"; then
      echo "[WARN] git pull failed; continuing with local repo state."
      if [[ "${SYNC_STRICT}" == "1" ]]; then
        echo "[ERROR] SYNC_STRICT=1 and pull failed; aborting." >&2
        exit 2
      fi
    fi
  else
    echo "[WARN] skip pull because fetch did not succeed."
  fi
else
  echo "[WARN] unable to detect branch; skipping checkout/pull."
fi

if [[ -n "${SYNC_COMMIT}" ]]; then
  git checkout "${SYNC_COMMIT}"
fi

echo "[INFO] repo after sync:"
git rev-parse --short HEAD || true
git branch --show-current || true

cd "${MODEL_DIR}"

mkdir -p ../logs
echo "[INFO] runtime: pwd=$(pwd)"
echo "[INFO] runtime: which_python=$(which python)"
python -V
python -c "import sys; print(f'[INFO] runtime: python_executable={sys.executable}')"
python -c "import vllm, torch, transformers; print('deps_ok')"

echo "[INFO] run params: quant_method=${QUANT} seed=${SEED} selection_metric=${SELECTION_METRIC} proxy_eval_count=${PROXY_EVAL_COUNT} top_k=${TOP_K} fast=${FAST} strict=${STRICT} allow_legacy_cli=${ALLOW_LEGACY_CLI}"

HELP_DUMP="../logs/main_help_${SLURM_JOB_ID:-manual}.txt"
python main.py --help > "${HELP_DUMP}" 2>&1 || true
HELP_TEXT="$(cat "${HELP_DUMP}")"
echo "[INFO] main.py --help (first 120 lines) from ${HELP_DUMP}"
sed -n '1,120p' "${HELP_DUMP}"

has_arg() {
  local arg="$1"
  grep -Fq -- "$arg" <<< "${HELP_TEXT}"
}

HAS_AWQ_SIG=0
if grep -Eq -- "--quant_method[[:space:]]+\\{gptq,awq\\}" <<< "${HELP_TEXT}" || has_arg "--awq_group_size"; then
  HAS_AWQ_SIG=1
fi
HAS_AUTOROUND_SIG=0
if grep -Fqi -- "autoround" <<< "${HELP_TEXT}"; then
  HAS_AUTOROUND_SIG=1
fi
NEW_CLI_OK=0
if [[ "${HAS_AWQ_SIG}" == "1" && "${HAS_AUTOROUND_SIG}" != "1" ]]; then
  NEW_CLI_OK=1
fi
echo "[INFO] cli detect: has_awq_sig=${HAS_AWQ_SIG} has_autoround_sig=${HAS_AUTOROUND_SIG} new_cli_ok=${NEW_CLI_OK}"

if [[ "${NEW_CLI_OK}" != "1" && "${ALLOW_LEGACY_CLI}" != "1" ]]; then
  echo "[ERROR] 구버전 main.py 실행 중. repo sync 후 재시도하세요." >&2
  echo "[ERROR] repo=$(pwd)" >&2
  echo "[ERROR] commit=$(git rev-parse --short HEAD || true) branch=$(git branch --show-current || true)" >&2
  echo "[ERROR] ALLOW_LEGACY_CLI=${ALLOW_LEGACY_CLI} (temporary bypass: ALLOW_LEGACY_CLI=1)" >&2
  exit 2
fi
if [[ "${NEW_CLI_OK}" != "1" && "${ALLOW_LEGACY_CLI}" == "1" ]]; then
  echo "[WARN] legacy CLI detected, but ALLOW_LEGACY_CLI=1 so continuing."
  echo "[WARN] legacy mode: advanced quant flags and --skip_smoke auto-injection are disabled."
fi

ARGS=(
  --base_model "${BASE_MODEL}"
  --out_dir "${OUT_DIR}"
  --seed "${SEED}"
  --quant_method "${QUANT}"
  --do_kd
  --do_lora
  --selection_metric "${SELECTION_METRIC}"
)

if [[ "${NEW_CLI_OK}" == "1" ]]; then
  if has_arg "--quant_small_grid"; then
    ARGS+=(--quant_small_grid)
  fi
  if has_arg "--quant_two_stage_eval"; then
    ARGS+=(--quant_two_stage_eval)
  fi
  if has_arg "--quant_proxy_eval_count"; then
    ARGS+=(--quant_proxy_eval_count "${PROXY_EVAL_COUNT}")
  fi
  if has_arg "--quant_two_stage_top_k"; then
    ARGS+=(--quant_two_stage_top_k "${TOP_K}")
  fi
fi

if [[ -n "${MIX_CONFIGS}" ]]; then
  if has_arg "--use_external_mix" && has_arg "--mix_dataset_configs"; then
    ARGS+=(--use_external_mix --mix_dataset_configs "${MIX_CONFIGS}")
  else
    echo "[WARN] mix args are not supported by this main.py; ignoring MIX_CONFIGS."
  fi
fi

if [[ "${FAST}" == "1" ]]; then
  echo "[WARN] FAST=1 enabled: skip smoke/rehearsal. Not for final submission."
  if has_arg "--skip_rehearsal"; then
    ARGS+=(--skip_rehearsal)
  fi
  if [[ "${NEW_CLI_OK}" == "1" ]] && has_arg "--skip_smoke"; then
    ARGS+=(--skip_smoke)
  fi
else
  # stable mode: do not add --skip_rehearsal
  if [[ "${STRICT}" == "1" ]]; then
    if has_arg "--strict_rehearsal"; then
      ARGS+=(--strict_rehearsal)
    fi
    if has_arg "--strict_guardrails"; then
      ARGS+=(--strict_guardrails)
    fi
  fi
fi

printf "[INFO] exec: python main.py"
printf " %q" "${ARGS[@]}"
printf "\n"

python main.py "${ARGS[@]}"
python ../tools/check_submit_zip.py --zip "${OUT_DIR}/submit.zip"

python - <<'PY'
import json
import os
from pathlib import Path
summary_path = Path(os.environ.get("OUT_DIR", "./artifacts")) / "final_summary.json"
if summary_path.exists():
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    sel = data.get("selected_quant_trial") or {}
    print(f"[INFO] final_summary.json: {summary_path.resolve()}")
    print(f"[INFO] selected_trial: trial={sel.get('trial')} method={sel.get('method')} block={sel.get('block_size')} group={sel.get('group_size')} damp={sel.get('dampening')}")
else:
    print(f"[WARN] final_summary.json not found at {summary_path}")
PY

exit 0
