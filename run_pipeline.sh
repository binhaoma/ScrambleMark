#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: ./run_pipeline.sh INPUT_DIR [OUTPUT_DIR]

Run preprocessing and inference for every .wav file under INPUT_DIR.
Generated audio is written to OUTPUT_DIR (default: ./output).

Optional environment variables:
  PYTHON          Python executable (default: python3)
  MODEL_DIR       Model directory or weights file (default: ./diffwave/model)
  PREPROCESS_DIR  Root directory for generated spectrograms (default: ./out_preprocess)
  OUTPUT_DIR      Directory for generated audio (default: ./output)
  DEVICE          Inference device: cuda or cpu (default: cuda)
  NUM_SAMPLES     Samples generated per spectrogram (default: 1)
  FAST            Set to 1 to enable fast sampling (default: 0)
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$1"
PYTHON_BIN="${PYTHON:-python3}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/diffwave/model}"
PREPROCESS_ROOT="${PREPROCESS_DIR:-${SCRIPT_DIR}/out_preprocess}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-${SCRIPT_DIR}/output}}"
DEVICE="${DEVICE:-cuda}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
FAST="${FAST:-0}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Error: input directory does not exist: ${INPUT_DIR}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -d "${MODEL_DIR}" ]]; then
  if [[ ! -f "${MODEL_DIR}/weights.pt" ]]; then
    echo "Error: model weights not found: ${MODEL_DIR}/weights.pt" >&2
    exit 1
  fi
elif [[ ! -f "${MODEL_DIR}" ]]; then
  echo "Error: model directory or weights file does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

if ! find "${INPUT_DIR}" -type f -name '*.wav' -print -quit | grep -q .; then
  echo "Error: no .wav files found under: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ "${DEVICE}" != "cuda" && "${DEVICE}" != "cpu" ]]; then
  echo "Error: DEVICE must be 'cuda' or 'cpu', got: ${DEVICE}" >&2
  exit 1
fi

if [[ ! "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: NUM_SAMPLES must be a positive integer, got: ${NUM_SAMPLES}" >&2
  exit 1
fi

if [[ "${FAST}" != "0" && "${FAST}" != "1" ]]; then
  echo "Error: FAST must be 0 or 1, got: ${FAST}" >&2
  exit 1
fi

# A unique directory prevents inference from accidentally processing stale
# spectrograms left by an earlier run.
mkdir -p "${PREPROCESS_ROOT}" "${OUTPUT_DIR}"
RUN_PREPROCESS_DIR="$(mktemp -d "${PREPROCESS_ROOT}/run.XXXXXX")"

echo "[1/2] Preprocessing"
echo "      input:  ${INPUT_DIR}"
echo "      specs:  ${RUN_PREPROCESS_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/preprocess.py" \
  "${INPUT_DIR}" \
  "${RUN_PREPROCESS_DIR}"

INFERENCE_ARGS=(
  --model_dir "${MODEL_DIR}"
  --input_dir "${RUN_PREPROCESS_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --num_samples "${NUM_SAMPLES}"
)

if [[ "${FAST}" == "1" ]]; then
  INFERENCE_ARGS+=(--fast)
fi

echo "[2/2] Inference"
echo "      model:  ${MODEL_DIR}"
echo "      output: ${OUTPUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/inference.py" "${INFERENCE_ARGS[@]}"

echo "Done. Generated audio is in: ${OUTPUT_DIR}"
