#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_full_watermark_gt_pipeline.sh {audioseal|wavmark} INPUT_DIR WATERMARK_DIR [OPTIONS]
  ./run_full_watermark_gt_pipeline.sh audiomarknet [WATERMARK_AUDIO_DIR] [RESULT_DIR] [OPTIONS]

AudioSeal/WavMark arguments:
  INPUT_DIR      Directory containing clean WAV files
  WATERMARK_DIR  Root directory for watermark and GT results

AudioMarkNet arguments:
  WATERMARK_AUDIO_DIR  Existing watermarked dataset
                       (default: ./audiomarknet_audio)
  RESULT_DIR           Root directory for GT results
                       (default: ./audiomarknet_gt_results)

Options:
  --num-samples N       Randomly select N input WAV files (default: all)
  --num-candidates N    GT candidates per audio (default: 10)
  --threshold VALUE     GT watermark/recovery threshold (default: 0.8)
  --max-opt-iters N     Optimization iterations per candidate (default: 20)
  --sample-rate N       GT output sample rate (default: 22050)
  -h, --help            Show this help

Environment variables:
  PYTHON  Python executable (default: python)
  DEVICE  Watermark generation/validation device (default: cuda)

Examples:
  ./run_full_watermark_gt_pipeline.sh audioseal ./clean_audio ./audioseal_gt_results
  ./run_full_watermark_gt_pipeline.sh wavmark ./clean_audio ./wavmark_gt_results --num-samples 10
  ./run_full_watermark_gt_pipeline.sh audiomarknet ./audiomarknet_audio ./audiomarknet_gt_results
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="$1"
shift

TEST_SAMPLE_COUNT=""
GT_NUM_CANDIDATES=10
GT_THRESHOLD=0.8
GT_MAX_OPT_ITERS=20
GT_SAMPLE_RATE=22050
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-samples)
      [[ $# -ge 2 ]] || { echo "Error: --num-samples requires a value." >&2; exit 2; }
      TEST_SAMPLE_COUNT="$2"
      shift 2
      ;;
    --num-candidates)
      [[ $# -ge 2 ]] || { echo "Error: --num-candidates requires a value." >&2; exit 2; }
      GT_NUM_CANDIDATES="$2"
      shift 2
      ;;
    --threshold)
      [[ $# -ge 2 ]] || { echo "Error: --threshold requires a value." >&2; exit 2; }
      GT_THRESHOLD="$2"
      shift 2
      ;;
    --max-opt-iters)
      [[ $# -ge 2 ]] || { echo "Error: --max-opt-iters requires a value." >&2; exit 2; }
      GT_MAX_OPT_ITERS="$2"
      shift 2
      ;;
    --sample-rate)
      [[ $# -ge 2 ]] || { echo "Error: --sample-rate requires a value." >&2; exit 2; }
      GT_SAMPLE_RATE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if [[ -n "${TEST_SAMPLE_COUNT}" ]] && ! is_positive_integer "${TEST_SAMPLE_COUNT}"; then
  echo "Error: --num-samples must be a positive integer; got: ${TEST_SAMPLE_COUNT}" >&2
  exit 2
fi
if ! is_positive_integer "${GT_NUM_CANDIDATES}"; then
  echo "Error: --num-candidates must be a positive integer; got: ${GT_NUM_CANDIDATES}" >&2
  exit 2
fi
if ! is_positive_integer "${GT_MAX_OPT_ITERS}"; then
  echo "Error: --max-opt-iters must be a positive integer; got: ${GT_MAX_OPT_ITERS}" >&2
  exit 2
fi
if ! is_positive_integer "${GT_SAMPLE_RATE}"; then
  echo "Error: --sample-rate must be a positive integer; got: ${GT_SAMPLE_RATE}" >&2
  exit 2
fi
if [[ ! "${GT_THRESHOLD}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "Error: --threshold must be a non-negative number; got: ${GT_THRESHOLD}" >&2
  exit 2
fi

case "${MODEL_NAME}" in
  audioseal|wavmark)
    if [[ ${#POSITIONAL_ARGS[@]} -ne 2 ]]; then
      usage >&2
      exit 2
    fi
    CLEAN_AUDIO_DIR="${POSITIONAL_ARGS[0]}"
    RESULT_ROOT="${POSITIONAL_ARGS[1]}"
    ;;
  audiomarknet)
    if [[ ${#POSITIONAL_ARGS[@]} -gt 2 ]]; then
      usage >&2
      exit 2
    fi
    CLEAN_AUDIO_DIR=""
    WATERMARK_AUDIO_DIR="${POSITIONAL_ARGS[0]:-${SCRIPT_DIR}/audiomarknet_audio}"
    RESULT_ROOT="${POSITIONAL_ARGS[1]:-${SCRIPT_DIR}/audiomarknet_gt_results}"
    ;;
  *)
    echo "Error: MODEL_NAME must be audioseal, wavmark, or audiomarknet; got: ${MODEL_NAME}" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON:-python}"
DEVICE_NAME="${DEVICE:-cuda}"
TEMP_RUN_ROOT=""
ORIGINAL_VALIDATION_LOG=""
GT_VALIDATION_LOG=""
SELECTED_NAMES=()

cleanup() {
  if [[ -n "${TEMP_RUN_ROOT}" && -d "${TEMP_RUN_ROOT}" ]]; then
    rm -rf -- "${TEMP_RUN_ROOT}"
  fi
  if [[ -n "${ORIGINAL_VALIDATION_LOG}" && -f "${ORIGINAL_VALIDATION_LOG}" ]]; then
    rm -f -- "${ORIGINAL_VALIDATION_LOG}"
  fi
  if [[ -n "${GT_VALIDATION_LOG}" && -f "${GT_VALIDATION_LOG}" ]]; then
    rm -f -- "${GT_VALIDATION_LOG}"
  fi
}
trap cleanup EXIT

write_validation_summary() {
  local label="$1"
  local validation_log="$2"
  local results_file="$3"

  awk -v label="${label}" '
    /^Average bit accuracy:/ { acc = $NF }
    /^Average max bit accuracy:/ { acc = $NF }
    /^Average detection score:/ { detection_score = $NF }
    /^ASR:/ { asr = $NF }
    END {
      if (acc == "") acc = detection_score
      if (acc == "") acc = "N/A"
      if (asr == "") asr = "N/A"
      printf "%s: ACC=%s, ASR=%s\n", label, acc, asr
    }
  ' "${validation_log}" >> "${results_file}"
}

collect_wav_files() {
  local source_dir="$1"
  local -n output_array="$2"
  mapfile -d '' output_array < <(
    find "${source_dir}" -maxdepth 1 -type f -name '*.wav' -print0 | sort -z
  )
}

select_random_audio_files() {
  local source_dir="$1"
  local selected_dir="$2"
  local -a candidates=()
  local -a chosen=()

  collect_wav_files "${source_dir}" candidates
  if [[ ${#candidates[@]} -lt ${TEST_SAMPLE_COUNT} ]]; then
    echo "Error: requested ${TEST_SAMPLE_COUNT} samples, but only ${#candidates[@]} WAV files were found in ${source_dir}." >&2
    exit 1
  fi

  mapfile -d '' chosen < <(
    printf '%s\0' "${candidates[@]}" | shuf -z -n "${TEST_SAMPLE_COUNT}"
  )

  mkdir -p "${selected_dir}"
  SELECTED_NAMES=()
  local source_file base_name
  for source_file in "${chosen[@]}"; do
    base_name="$(basename -- "${source_file}")"
    SELECTED_NAMES+=("${base_name}")
    cp -p -- "${source_file}" "${selected_dir}/${base_name}"
  done
}

copy_selected_names() {
  local source_dir="$1"
  local selected_dir="$2"
  local base_name source_file

  mkdir -p "${selected_dir}"
  for base_name in "${SELECTED_NAMES[@]}"; do
    source_file="${source_dir}/${base_name}"
    if [[ ! -f "${source_file}" ]]; then
      echo "Error: expected generated audio was not found: ${source_file}" >&2
      exit 1
    fi
    cp -p -- "${source_file}" "${selected_dir}/${base_name}"
  done
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${DEVICE_NAME}" != "cuda" && "${DEVICE_NAME}" != "cpu" ]]; then
  echo "Error: DEVICE must be cuda or cpu; got: ${DEVICE_NAME}" >&2
  exit 2
fi
if [[ -n "${TEST_SAMPLE_COUNT}" ]] && ! command -v shuf >/dev/null 2>&1; then
  echo "Error: shuf is required for random sample selection but was not found." >&2
  exit 1
fi

if [[ "${MODEL_NAME}" == "audiomarknet" ]]; then
  [[ -d "${WATERMARK_AUDIO_DIR}" ]] || { echo "Error: watermark directory does not exist: ${WATERMARK_AUDIO_DIR}" >&2; exit 1; }
else
  [[ -d "${CLEAN_AUDIO_DIR}" ]] || { echo "Error: clean audio directory does not exist: ${CLEAN_AUDIO_DIR}" >&2; exit 1; }
fi

mkdir -p "${RESULT_ROOT}"
RESULT_ROOT="$(cd -- "${RESULT_ROOT}" && pwd)"
GROUND_TRUTH_DIR="${RESULT_ROOT}/ground_truth"
RESULTS_FILE="${RESULT_ROOT}/gt_results.txt"
TEMP_RUN_ROOT="$(mktemp -d /tmp/watermark_gt_pipeline.XXXXXX)"
GT_RUN_DIR="${TEMP_RUN_ROOT}/ground_truth"
mkdir -p "${GT_RUN_DIR}" "${GROUND_TRUTH_DIR}"

echo "============================================================"
if [[ "${MODEL_NAME}" == "audiomarknet" ]]; then
  echo "[1/4] Use existing AudioMarkNet watermark audio (generation skipped)"
  echo "      input: ${WATERMARK_AUDIO_DIR}"
  FULL_WATERMARK_AUDIO_DIR="${WATERMARK_AUDIO_DIR}"
  if [[ -n "${TEST_SAMPLE_COUNT}" ]]; then
    select_random_audio_files "${FULL_WATERMARK_AUDIO_DIR}" "${TEMP_RUN_ROOT}/watermarked"
    WATERMARK_AUDIO_DIR="${TEMP_RUN_ROOT}/watermarked"
  fi
else
  GENERATION_INPUT_DIR="${CLEAN_AUDIO_DIR}"
  if [[ -n "${TEST_SAMPLE_COUNT}" ]]; then
    select_random_audio_files "${CLEAN_AUDIO_DIR}" "${TEMP_RUN_ROOT}/clean"
    GENERATION_INPUT_DIR="${TEMP_RUN_ROOT}/clean"
  fi

  echo "[1/4] Generate watermarked audio"
  echo "      input:  ${GENERATION_INPUT_DIR}"
  echo "      output: ${RESULT_ROOT}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/watermark_generation.py" \
    --model_name "${MODEL_NAME}" \
    --input_dir "${GENERATION_INPUT_DIR}" \
    --output_dir "${RESULT_ROOT}" \
    --device "${DEVICE_NAME}"

  FULL_WATERMARK_AUDIO_DIR="${RESULT_ROOT}/data"
  WATERMARK_AUDIO_DIR="${FULL_WATERMARK_AUDIO_DIR}"
  if [[ -n "${TEST_SAMPLE_COUNT}" ]]; then
    copy_selected_names "${FULL_WATERMARK_AUDIO_DIR}" "${TEMP_RUN_ROOT}/watermarked"
    WATERMARK_AUDIO_DIR="${TEMP_RUN_ROOT}/watermarked"
  fi
fi

if ! find "${WATERMARK_AUDIO_DIR}" -maxdepth 1 -type f -name '*.wav' -print -quit | grep -q .; then
  echo "Error: no WAV files found for GT generation in: ${WATERMARK_AUDIO_DIR}" >&2
  exit 1
fi

echo "============================================================"
echo "[2/4] Generate ground-truth audio"
echo "      input:      ${WATERMARK_AUDIO_DIR}"
echo "      candidates: ${GT_NUM_CANDIDATES} per audio"
echo "      output:     ${GROUND_TRUTH_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_GT.py" \
  --input_dir "${WATERMARK_AUDIO_DIR}" \
  --output_dir "${GT_RUN_DIR}" \
  --num_candidates "${GT_NUM_CANDIDATES}" \
  --threshold "${GT_THRESHOLD}" \
  --max_opt_iters "${GT_MAX_OPT_ITERS}" \
  --sample_rate "${GT_SAMPLE_RATE}"

mapfile -d '' GENERATED_GT_FILES < <(
  find "${GT_RUN_DIR}" -maxdepth 1 -type f -name '*.wav' -print0 | sort -z
)
if [[ ${#GENERATED_GT_FILES[@]} -eq 0 ]]; then
  echo "Error: generate_GT.py did not produce any successful GT audio." >&2
  exit 1
fi

PAIRED_ORIGINAL_DIR="${TEMP_RUN_ROOT}/paired_original"
PAIRED_GT_DIR="${TEMP_RUN_ROOT}/paired_gt"
mkdir -p "${PAIRED_ORIGINAL_DIR}" "${PAIRED_GT_DIR}"
for gt_file in "${GENERATED_GT_FILES[@]}"; do
  base_name="$(basename -- "${gt_file}")"
  original_file="${WATERMARK_AUDIO_DIR}/${base_name}"
  if [[ ! -f "${original_file}" ]]; then
    echo "Error: GT output has no matching watermarked input: ${base_name}" >&2
    exit 1
  fi
  cp -p -- "${gt_file}" "${GROUND_TRUTH_DIR}/${base_name}"
  cp -p -- "${gt_file}" "${PAIRED_GT_DIR}/${base_name}"
  cp -p -- "${original_file}" "${PAIRED_ORIGINAL_DIR}/${base_name}"
done

: > "${RESULTS_FILE}"
ORIGINAL_VALIDATION_LOG="$(mktemp /tmp/watermark_gt_original_validation.XXXXXX)"
GT_VALIDATION_LOG="$(mktemp /tmp/watermark_gt_generated_validation.XXXXXX)"

echo "============================================================"
echo "[3/4] Validate original watermarked audio (ACC and ASR)"
if [[ "${MODEL_NAME}" == "audiomarknet" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/watermark_validation.py" \
    --model_name "${MODEL_NAME}" \
    --watermark_dir "${PAIRED_ORIGINAL_DIR}" \
    --device "${DEVICE_NAME}" 2>&1 | tee "${ORIGINAL_VALIDATION_LOG}"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/watermark_validation.py" \
    --model_name "${MODEL_NAME}" \
    --watermark_dir "${RESULT_ROOT}" \
    --audio_subdir "${PAIRED_ORIGINAL_DIR}" \
    --device "${DEVICE_NAME}" 2>&1 | tee "${ORIGINAL_VALIDATION_LOG}"
fi
write_validation_summary "Original" "${ORIGINAL_VALIDATION_LOG}" "${RESULTS_FILE}"

echo "============================================================"
echo "[4/4] Validate ground-truth audio (ACC and ASR)"
if [[ "${MODEL_NAME}" == "audiomarknet" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/watermark_validation.py" \
    --model_name "${MODEL_NAME}" \
    --watermark_dir "${PAIRED_GT_DIR}" \
    --device "${DEVICE_NAME}" 2>&1 | tee "${GT_VALIDATION_LOG}"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/watermark_validation.py" \
    --model_name "${MODEL_NAME}" \
    --watermark_dir "${RESULT_ROOT}" \
    --audio_subdir "${PAIRED_GT_DIR}" \
    --device "${DEVICE_NAME}" 2>&1 | tee "${GT_VALIDATION_LOG}"
fi
write_validation_summary "GroundTruth" "${GT_VALIDATION_LOG}" "${RESULTS_FILE}"

echo "============================================================"
echo "Done."
echo "Successfully paired audio: ${#GENERATED_GT_FILES[@]}"
echo "Watermarked audio:         ${FULL_WATERMARK_AUDIO_DIR}"
echo "Ground-truth audio:        ${GROUND_TRUTH_DIR}"
echo "Validation results:        ${RESULTS_FILE}"
cat "${RESULTS_FILE}"
