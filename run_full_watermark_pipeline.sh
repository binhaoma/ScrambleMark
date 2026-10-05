#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_full_watermark_pipeline.sh {audioseal|wavmark} INPUT_DIR WATERMARK_DIR [--num-samples N]
  ./run_full_watermark_pipeline.sh audiomarknet [WATERMARK_AUDIO_DIR] [RESULT_DIR] [--num-samples N]

AudioSeal/WavMark arguments:
  INPUT_DIR      Directory containing the original clean audio
  WATERMARK_DIR  Root directory used for generated watermark results

AudioMarkNet arguments:
  WATERMARK_AUDIO_DIR  Existing watermarked dataset
                       (default: ./audiomarknet_audio)
  RESULT_DIR           Root directory for ScrambleMark results
                       (default: ./audiomarknet_results)

Options:
  --num-samples N  Randomly select N WAV files for the complete pipeline
  -h, --help       Show this help

Pipeline:
  1. Generate watermarked audio with watermark_generation.py
     (AudioMarkNet skips generation and uses its existing dataset)
  2. Run ScrambleMark with run_pipeline.sh
  3. Validate the original watermarked audio
  4. Validate the ScrambleMark audio using the same watermark/message

Optional environment variables:
  PYTHON       Python executable (default: python)
  DEVICE       cuda or cpu (default: cuda)
  MODEL_DIR    DiffWave model directory/file (default: ./diffwave/model)
  FAST         1 enables fast DiffWave sampling; 0 disables it (default: 0)

Examples:
  ./run_full_watermark_pipeline.sh audioseal ./clean_audio ./audioseal_results --num-samples 10
  ./run_full_watermark_pipeline.sh wavmark ./clean_audio ./wavmark_results --num-samples 20
  ./run_full_watermark_pipeline.sh audiomarknet --num-samples 10
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
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-samples)
      if [[ $# -lt 2 ]]; then
        echo "Error: --num-samples requires a positive integer." >&2
        exit 2
      fi
      TEST_SAMPLE_COUNT="$2"
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

if [[ -n "${TEST_SAMPLE_COUNT}" && ! "${TEST_SAMPLE_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --num-samples must be a positive integer; got: ${TEST_SAMPLE_COUNT}" >&2
  exit 2
fi

case "${MODEL_NAME}" in
  audioseal|wavmark)
    if [[ ${#POSITIONAL_ARGS[@]} -ne 2 ]]; then
      usage >&2
      exit 2
    fi
    INPUT_DIR="${POSITIONAL_ARGS[0]}"
    WATERMARK_DIR="${POSITIONAL_ARGS[1]}"
    ;;
  audiomarknet)
    if [[ ${#POSITIONAL_ARGS[@]} -gt 2 ]]; then
      usage >&2
      exit 2
    fi
    INPUT_DIR=""
    WATERMARK_AUDIO_DIR="${POSITIONAL_ARGS[0]:-${SCRIPT_DIR}/audiomarknet_audio}"
    WATERMARK_DIR="${POSITIONAL_ARGS[1]:-${SCRIPT_DIR}/audiomarknet_results}"
    ;;
  *)
    echo "Error: MODEL_NAME must be audioseal, wavmark, or audiomarknet; got: ${MODEL_NAME}" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON:-python}"
DEVICE_NAME="${DEVICE:-cuda}"
DIFFWAVE_MODEL="${MODEL_DIR:-${SCRIPT_DIR}/diffwave/model}"
FAST_MODE="${FAST:-0}"
SCRAMBLEMARK_SUBDIR="scramblemark"
SCRAMBLEMARK_DIR="${WATERMARK_DIR}/${SCRAMBLEMARK_SUBDIR}"
TEMP_SELECTION_ROOT=""
ORIGINAL_VALIDATION_LOG=""
SCRAMBLEMARK_VALIDATION_LOG=""
SELECTED_NAMES=()

cleanup() {
  if [[ -n "${TEMP_SELECTION_ROOT}" && -d "${TEMP_SELECTION_ROOT}" ]]; then
    rm -rf -- "${TEMP_SELECTION_ROOT}"
  fi
  if [[ -n "${ORIGINAL_VALIDATION_LOG}" && -f "${ORIGINAL_VALIDATION_LOG}" ]]; then
    rm -f -- "${ORIGINAL_VALIDATION_LOG}"
  fi
  if [[ -n "${SCRAMBLEMARK_VALIDATION_LOG}" && -f "${SCRAMBLEMARK_VALIDATION_LOG}" ]]; then
    rm -f -- "${SCRAMBLEMARK_VALIDATION_LOG}"
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

select_random_audio_files() {
  local source_dir="$1"
  local selected_dir="$2"
  local -a candidates=()
  local -a random_candidates=()

  mapfile -d '' candidates < <(
    find "${source_dir}" -maxdepth 1 -type f -name '*.wav' -print0 | sort -z
  )

  if [[ ${#candidates[@]} -lt ${TEST_SAMPLE_COUNT} ]]; then
    echo "Error: requested ${TEST_SAMPLE_COUNT} samples, but only ${#candidates[@]} WAV files were found in ${source_dir}." >&2
    exit 1
  fi

  mapfile -d '' random_candidates < <(
    printf '%s\0' "${candidates[@]}" | shuf -z -n "${TEST_SAMPLE_COUNT}"
  )

  mkdir -p "${selected_dir}"
  SELECTED_NAMES=()
  local index source_file base_name
  for ((index = 0; index < TEST_SAMPLE_COUNT; index++)); do
    source_file="${random_candidates[index]}"
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

if [[ "${MODEL_NAME}" == "audiomarknet" ]]; then
  if [[ ! -d "${WATERMARK_AUDIO_DIR}" ]]; then
    echo "Error: AudioMarkNet dataset does not exist: ${WATERMARK_AUDIO_DIR}" >&2
    exit 1
  fi
else
  if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Error: input directory does not exist: ${INPUT_DIR}" >&2
    exit 1
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -n "${TEST_SAMPLE_COUNT}" ]] && ! command -v shuf >/dev/null 2>&1; then
  echo "Error: shuf is required for random sample selection but was not found." >&2
  exit 1
fi

if [[ "${DEVICE_NAME}" != "cuda" && "${DEVICE_NAME}" != "cpu" ]]; then
  echo "Error: DEVICE must be cuda or cpu; got: ${DEVICE_NAME}" >&2
  exit 2
fi

if [[ "${FAST_MODE}" != "0" && "${FAST_MODE}" != "1" ]]; then
  echo "Error: FAST must be 0 or 1; got: ${FAST_MODE}" >&2
  exit 2
fi

mkdir -p "${WATERMARK_DIR}"
WATERMARK_DIR="$(cd -- "${WATERMARK_DIR}" && pwd)"
SCRAMBLEMARK_DIR="${WATERMARK_DIR}/${SCRAMBLEMARK_SUBDIR}"

if [[ -n "${TEST_SAMPLE_COUNT}" ]]; then
  TEMP_SELECTION_ROOT="$(mktemp -d /tmp/scramblemark_test.XXXXXX)"
  echo "Selected test samples: ${TEST_SAMPLE_COUNT}"
else
  echo "Selected test samples: all"
fi

echo "============================================================"
if [[ "${MODEL_NAME}" == "audiomarknet" ]]; then
  echo "[1/4] Use existing AudioMarkNet watermark audio (generation skipped)"
  echo "      input: ${WATERMARK_AUDIO_DIR}"
  FULL_WATERMARK_AUDIO_DIR="${WATERMARK_AUDIO_DIR}"

  if [[ -n "${TEST_SAMPLE_COUNT}" ]]; then
    select_random_audio_files "${FULL_WATERMARK_AUDIO_DIR}" "${TEMP_SELECTION_ROOT}/watermarked"
    WATERMARK_AUDIO_DIR="${TEMP_SELECTION_ROOT}/watermarked"
  fi
else
  GENERATION_INPUT_DIR="${INPUT_DIR}"
  if [[ -n "${TEST_SAMPLE_COUNT}" ]]; then
    select_random_audio_files "${INPUT_DIR}" "${TEMP_SELECTION_ROOT}/clean"
    GENERATION_INPUT_DIR="${TEMP_SELECTION_ROOT}/clean"
  fi

  echo "[1/4] Generate watermark audio"
  echo "      model:  ${MODEL_NAME}"
  echo "      input:  ${GENERATION_INPUT_DIR}"
  echo "      output: ${WATERMARK_DIR}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/watermark_generation.py" \
    --model_name "${MODEL_NAME}" \
    --input_dir "${GENERATION_INPUT_DIR}" \
    --output_dir "${WATERMARK_DIR}" \
    --device "${DEVICE_NAME}"

  FULL_WATERMARK_AUDIO_DIR="${WATERMARK_DIR}/data"
  WATERMARK_AUDIO_DIR="${FULL_WATERMARK_AUDIO_DIR}"
  if [[ -n "${TEST_SAMPLE_COUNT}" ]]; then
    copy_selected_names "${FULL_WATERMARK_AUDIO_DIR}" "${TEMP_SELECTION_ROOT}/watermarked"
    WATERMARK_AUDIO_DIR="${TEMP_SELECTION_ROOT}/watermarked"
  fi
fi

if [[ ! -d "${WATERMARK_AUDIO_DIR}" ]]; then
  echo "Error: watermark audio directory does not exist: ${WATERMARK_AUDIO_DIR}" >&2
  exit 1
fi

if ! find "${WATERMARK_AUDIO_DIR}" -maxdepth 1 \( -type f -o -type l \) -name '*.wav' -print -quit | grep -q .; then
  echo "Error: no .wav files found in: ${WATERMARK_AUDIO_DIR}" >&2
  exit 1
fi

echo "============================================================"
echo "[2/4] Generate ScrambleMark audio"
echo "      input:  ${WATERMARK_AUDIO_DIR}"
echo "      output: ${SCRAMBLEMARK_DIR}"
PYTHON="${PYTHON_BIN}" \
DEVICE="${DEVICE_NAME}" \
MODEL_DIR="${DIFFWAVE_MODEL}" \
FAST="${FAST_MODE}" \
  bash "${SCRIPT_DIR}/run_pipeline.sh" \
  "${WATERMARK_AUDIO_DIR}" \
  "${SCRAMBLEMARK_DIR}"

SCRAMBLEMARK_VALIDATION_DIR="${SCRAMBLEMARK_DIR}"
if [[ -n "${TEST_SAMPLE_COUNT}" ]]; then
  copy_selected_names "${SCRAMBLEMARK_DIR}" "${TEMP_SELECTION_ROOT}/scramblemark"
  SCRAMBLEMARK_VALIDATION_DIR="${TEMP_SELECTION_ROOT}/scramblemark"
fi

RESULTS_FILE="${WATERMARK_DIR}/results.txt"
: > "${RESULTS_FILE}"
ORIGINAL_VALIDATION_LOG="$(mktemp /tmp/scramblemark_original_validation.XXXXXX)"
SCRAMBLEMARK_VALIDATION_LOG="$(mktemp /tmp/scramblemark_attacked_validation.XXXXXX)"

echo "============================================================"
echo "[3/4] Validate original watermarked audio (ACC and ASR)"
if [[ "${MODEL_NAME}" == "audiomarknet" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/watermark_validation.py" \
    --model_name "${MODEL_NAME}" \
    --watermark_dir "${WATERMARK_AUDIO_DIR}" \
    --device "${DEVICE_NAME}" 2>&1 | tee "${ORIGINAL_VALIDATION_LOG}"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/watermark_validation.py" \
    --model_name "${MODEL_NAME}" \
    --watermark_dir "${WATERMARK_DIR}" \
    --audio_subdir "${WATERMARK_AUDIO_DIR}" \
    --device "${DEVICE_NAME}" 2>&1 | tee "${ORIGINAL_VALIDATION_LOG}"
fi
write_validation_summary "Original" "${ORIGINAL_VALIDATION_LOG}" "${RESULTS_FILE}"

echo "============================================================"
echo "[4/4] Validate ScrambleMark audio (ACC and ASR)"
if [[ "${MODEL_NAME}" == "audiomarknet" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/watermark_validation.py" \
    --model_name "${MODEL_NAME}" \
    --watermark_dir "${SCRAMBLEMARK_VALIDATION_DIR}" \
    --device "${DEVICE_NAME}" 2>&1 | tee "${SCRAMBLEMARK_VALIDATION_LOG}"
else
  # Keep WATERMARK_DIR as the experiment root so validation can reuse msg/.
  "${PYTHON_BIN}" "${SCRIPT_DIR}/watermark_validation.py" \
    --model_name "${MODEL_NAME}" \
    --watermark_dir "${WATERMARK_DIR}" \
    --audio_subdir "${SCRAMBLEMARK_VALIDATION_DIR}" \
    --device "${DEVICE_NAME}" 2>&1 | tee "${SCRAMBLEMARK_VALIDATION_LOG}"
fi
write_validation_summary "ScrambleMark" "${SCRAMBLEMARK_VALIDATION_LOG}" "${RESULTS_FILE}"

echo "============================================================"
echo "Done."
echo "Test samples:               ${TEST_SAMPLE_COUNT:-all}"
echo "Original watermarked audio: ${FULL_WATERMARK_AUDIO_DIR}"
echo "ScrambleMark audio:         ${SCRAMBLEMARK_DIR}"
echo "Validation results:         ${RESULTS_FILE}"
cat "${RESULTS_FILE}"
