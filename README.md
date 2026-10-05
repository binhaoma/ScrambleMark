# ScrambleMark Full Pipelines

This is the official implementation of **ScrambleMark: Zero-Query Watermark-Agnostic Black-Box Evasion of Modern Speech Watermarking**.

> **Note for commercial platforms:** We provided the complete ScrambleMark codebase to the commercial platform around July 2026, giving the platform sufficient time to develop and deploy corresponding mitigations or updates.

## 1. Create the Conda Environment

Run these commands from the repository root:

```bash
conda env create -f environment.yml
conda activate scramblemark
```

The environment uses Python 3.10, PyTorch 2.4.1, and CUDA 11.8.

The bundled `./python-stretch` package is built and installed automatically.

> **Note:** This repository uses a customized version of `python-stretch` that includes project-specific function mappings required by ScrambleMark. Please make sure to install and use the `python-stretch` package provided in this GitHub repository rather than another version from external sources.

## 2. Download Project Assets

Run this command from the repository root before using either pipeline:

```bash
python download_project_assets.py
```

It downloads and installs all required checkpoints, plus the two audio datasets:

```text
./clean_audio
./audiomarknet_audio
```

Existing valid checkpoints and downloaded audio files are skipped. Interrupted downloads can be resumed by running the same command again.

---

## 3. `run_full_watermark_pipeline.sh`

Runs watermark generation, ScrambleMark, and watermark validation before and after ScrambleMark.

### AudioSeal or WavMark

```bash
./run_full_watermark_pipeline.sh \
  {audioseal|wavmark} \
  INPUT_DIR \
  RESULT_DIR \
  [--num-samples N]
```

Example:

```bash
./run_full_watermark_pipeline.sh \
  audioseal \
  ./clean_audio \
  ./audioseal_results \
  --num-samples 10
```

### AudioMarkNet

By default, AudioMarkNet uses `./audiomarknet_audio`.

```bash
./run_full_watermark_pipeline.sh audiomarknet [DATA_DIR] [RESULT_DIR] [--num-samples N]
```

Use the default data and result directories:

```bash
./run_full_watermark_pipeline.sh audiomarknet --num-samples 10
```

Specify custom directories:

```bash
./run_full_watermark_pipeline.sh \
  audiomarknet \
  ./audiomarknet_audio \
  ./audiomarknet_results \
  --num-samples 10
```

Defaults:

```text
DATA_DIR   = ./audiomarknet_audio
RESULT_DIR = ./audiomarknet_results
```

### Output

```text
RESULT_DIR/
├── data/             # Watermarked audio for AudioSeal/WavMark
├── msg/              # Watermark messages for AudioSeal/WavMark
├── scramblemark/     # ScrambleMark audio
└── results.txt       # Original and ScrambleMark ACC/ASR
```

`results.txt` contains:

```text
Original: ACC=..., ASR=...
ScrambleMark: ACC=..., ASR=...
```

Optional environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `DEVICE` | `cuda` | Inference and validation device |
| `MODEL_DIR` | `./diffwave/model` | DiffWave model directory or weights file |
| `FAST` | `0` | Set to `1` to enable fast DiffWave sampling |

---

## 4. `run_full_watermark_gt_pipeline.sh`

Runs watermark generation, GT generation, and watermark validation on the original watermarked audio and GT audio.

This pipeline provides a lightweight way to evaluate the transferability of ScrambleMark on your own speech audio without fine-tuning a model. The generated GT samples exhibit similar transferability to ScrambleMark, making this pipeline useful for quickly evaluating transferability across different speech watermarking methods.

### AudioSeal or WavMark

```bash
./run_full_watermark_gt_pipeline.sh \
  {audioseal|wavmark} \
  INPUT_DIR \
  RESULT_DIR \
  [OPTIONS]
```

Example:

```bash
./run_full_watermark_gt_pipeline.sh \
  wavmark \
  ./clean_audio \
  ./wavmark_gt_results \
  --num-samples 10 \
  --num-candidates 1
```

### AudioMarkNet

By default, AudioMarkNet uses `./audiomarknet_audio`.

```bash
./run_full_watermark_gt_pipeline.sh audiomarknet [DATA_DIR] [RESULT_DIR] [OPTIONS]
```

Use the default data and result directories:

```bash
./run_full_watermark_gt_pipeline.sh audiomarknet --num-samples 10
```

Specify custom directories:

```bash
./run_full_watermark_gt_pipeline.sh \
  audiomarknet \
  ./audiomarknet_audio \
  ./audiomarknet_gt_results \
  --num-samples 10 \
  --num-candidates 1
```

Defaults:

```text
DATA_DIR   = ./audiomarknet_audio
RESULT_DIR = ./audiomarknet_gt_results
```

### Options

| Option | Default | Description |
| --- | ---: | --- |
| `--num-samples N` | all | Randomly select N WAV files |
| `--num-candidates N` | `10` | GT candidates generated per audio file |


### Output

```text
RESULT_DIR/
├── data/             # Watermarked audio for AudioSeal/WavMark
├── msg/              # Watermark messages for AudioSeal/WavMark
├── ground_truth/     # Best successful GT audio
└── gt_results.txt    # Original and GroundTruth ACC/ASR
```

`gt_results.txt` contains:

```text
Original: ACC=..., ASR=...
GroundTruth: ACC=..., ASR=...
```

Optional environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `DEVICE` | `cuda` | Watermark generation and validation device |