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
Collaborative Watermarking is imported directly from the bundled
`./collaborative_watermarking_with_codecs` source tree, while its HiFi-GAN and
AASIST runtime dependencies are installed by `environment.yml`.

> **Note:** This repository uses a customized version of `python-stretch` that includes project-specific function mappings required by ScrambleMark. Please make sure to install and use the `python-stretch` package provided in this GitHub repository rather than another version from external sources.

## 2. Download Project Assets

All watermark checkpoints used in this project are collected from the official
releases of their respective watermarking methods.

Run this command from the repository root before using either pipeline:

```bash
python download_project_assets.py
```

It automatically downloads and installs all required checkpoints, including
the Collaborative generator and detector, and places the two audio datasets in:

```text
./clean_audio
./audiomarknet_audio
```

Existing valid checkpoints and downloaded audio files are skipped. Interrupted downloads can be resumed by running the same command again.

---

## 3. `run_full_watermark_pipeline.sh`

Runs watermark generation, ScrambleMark, and watermark validation before and after ScrambleMark.

### AudioSeal, WavMark, or Collaborative

```bash
./run_full_watermark_pipeline.sh \
  {audioseal|wavmark|collaborative} \
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

Collaborative example:

```bash
./run_full_watermark_pipeline.sh \
  collaborative \
  ./clean_audio \
  ./collaborative_results \
  --num-samples 10
```

Collaborative renders watermarked speech at 22.05 kHz using the bundled
`collab-dac` generator. Its watermark is a binary detection task rather than a
bit-message task, so it does not create a `msg/` directory. Validation reports
the average detection score and counts an attack as successful when the score
is at most `0.5`.

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
├── data/             # Generated watermarked audio
├── msg/              # AudioSeal/WavMark only; absent for Collaborative
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

## 4. Scrambler Generation (`run_full_watermark_scrambler_pipeline.sh`)

Runs watermark generation, scrambler generation, and watermark validation on
the original watermarked audio and scrambler audio.

This pipeline provides a lightweight way to evaluate the transferability of
ScrambleMark on your own speech audio without fine-tuning a model. The generated
scrambler samples exhibit similar transferability to ScrambleMark, making this
pipeline useful for quickly evaluating transferability across different speech
watermarking methods.

### AudioSeal, WavMark, or Collaborative

```bash
./run_full_watermark_scrambler_pipeline.sh \
  {audioseal|wavmark|collaborative} \
  INPUT_DIR \
  RESULT_DIR \
  [OPTIONS]
```

Example:

```bash
./run_full_watermark_scrambler_pipeline.sh \
  wavmark \
  ./clean_audio \
  ./wavmark_scrambler_results \
  --num-samples 10 \
  --num-candidates 1
```

Collaborative example:

```bash
./run_full_watermark_scrambler_pipeline.sh \
  collaborative \
  ./clean_audio \
  ./collaborative_scrambler_results \
  --num-samples 10 \
  --num-candidates 1
```

As in the full pipeline, Collaborative uses binary detection and therefore
does not require saved message tensors.

### AudioMarkNet

By default, AudioMarkNet uses `./audiomarknet_audio`.

```bash
./run_full_watermark_scrambler_pipeline.sh audiomarknet [DATA_DIR] [RESULT_DIR] [OPTIONS]
```

Use the default data and result directories:

```bash
./run_full_watermark_scrambler_pipeline.sh audiomarknet --num-samples 10
```

Specify custom directories:

```bash
./run_full_watermark_scrambler_pipeline.sh \
  audiomarknet \
  ./audiomarknet_audio \
  ./audiomarknet_scrambler_results \
  --num-samples 10 \
  --num-candidates 1
```

Defaults:

```text
DATA_DIR   = ./audiomarknet_audio
RESULT_DIR = ./audiomarknet_scrambler_results
```

### Options

| Option | Default | Description |
| --- | ---: | --- |
| `--num-samples N` | all | Randomly select N WAV files |
| `--num-candidates N` | `10` | Scrambler candidates generated per audio file |


### Output

```text
RESULT_DIR/
├── data/             # Generated watermarked audio
├── msg/              # AudioSeal/WavMark only; absent for Collaborative
├── scrambler/              # Best successful scrambler audio
└── scrambler_results.txt   # Original and scrambler ACC/ASR
```

`scrambler_results.txt` contains the validation ACC and ASR for both the original
watermarked audio and the generated scrambler audio.

Optional environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `DEVICE` | `cuda` | Watermark generation and validation device |

---

## 5. Dynamic Fine-Tuning (`run_single_audio_scrambler_finetune.sh`)

Runs the complete single-audio scrambler fine-tuning workflow:

1. Generate one watermarked audio file from a clean input.
2. Validate the generated watermark.
3. Generate multiple distinct scrambler targets from the watermarked audio.
4. Fine-tune DiffWave with the watermarked audio as the condition and the scrambler
   audio files as targets.
5. Run inference with the fine-tuned checkpoint.
6. Validate the target watermark in the final generated audio.

AudioSeal, WavMark, and Collaborative are supported.

### Usage

```bash
./run_single_audio_scrambler_finetune.sh \
  {audioseal|wavmark|collaborative} \
  INPUT_AUDIO \
  RESULT_DIR \
  [OPTIONS]
```

Example:

```bash
./run_single_audio_scrambler_finetune.sh \
  wavmark \
  ./clean_audio/example.wav \
  ./single_audio_finetune_results
```

The default initial checkpoint is:

```text
./diffwave/model/weights.pt
```

Use a different checkpoint with:

```bash
--pretrained-checkpoint ./path/to/weights.pt
```

### Fine-Tuning

This pipeline uses the generated scrambler audio to fine-tune the model for the
selected input audio.

```bash
./run_single_audio_scrambler_finetune.sh \
  wavmark \
  ./clean_audio/example.wav \
  ./single_audio_finetune_results \
  --num-scramblers 32 \
  --max-steps 10000
```

Using 32 scrambler targets and 10,000 fine-tuning steps provides a quick validation
setting. You can also replace these values with the default fine-tuning
parameters used in the paper.

### Options

| Option | Default | Description |
| --- | ---: | --- |
| `--num-scramblers N` | `32` | Number of distinct scrambler targets |
| `--num-candidates N` | `1` | Candidates evaluated for each scrambler target |
| `--max-steps N` | `10000` | DiffWave optimizer steps |
| `--pretrained-checkpoint PATH` | `./diffwave/model/weights.pt` | Initial DiffWave checkpoint |

### Output

```text
RESULT_DIR/
├── watermark/
│   ├── data/                       # Generated watermarked input
│   └── msg/                        # AudioSeal/WavMark only
├── watermark_validation.log
├── validation_results.txt          # Before/after watermark metrics
├── scrambler/
│   ├── scrambler_001.wav
│   └── ...
├── preprocessed/                   # Source condition spectrogram
├── model/
│   ├── weights.pt                  # Latest fine-tuned checkpoint
│   └── finetune_metadata.json
├── finetuned_audio/                # Final fine-tuned DiffWave output
└── finetuned_validation.log
```
