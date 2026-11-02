#!/usr/bin/env python3
"""Fine-tune conditional DiffWave with multiple pre-generated scrambler targets."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffwave.model import DiffWave
from diffwave.params import AttrDict, params as base_params
from preprocess import transform as preprocess_audio


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
CHECKPOINT_INTERVAL = 5_000
SUMMARY_INTERVAL = 1_000
SEED = 2022


def preprocess_sources(source_files: list[Path], preprocess_dir: Path) -> None:
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    for source_path in tqdm(source_files, desc="Preprocessing", unit="audio"):
        preprocess_audio(str(source_path), str(preprocess_dir))


class OneToManyScramblerDataset(Dataset):
    def __init__(
        self,
        source_targets: dict[Path, list[Path]],
        preprocess_dir: Path,
        params: AttrDict,
    ) -> None:
        self.params = params
        self.records = [
            (source_path, target_path)
            for source_path, target_paths in source_targets.items()
            for target_path in target_paths
        ]
        self.conditioners: dict[Path, torch.Tensor] = {}

        for source_path in source_targets:
            spectrogram_path = preprocess_dir / f"{source_path.name}.spec.npy"
            if not spectrogram_path.is_file():
                raise FileNotFoundError(
                    f"Preprocessed spectrogram not found: {spectrogram_path}"
                )

            spectrogram = torch.from_numpy(np.load(spectrogram_path)).float()
            expected_shape = (params.n_mels, "frames")
            if spectrogram.ndim != 2 or spectrogram.shape[0] != params.n_mels:
                raise ValueError(
                    f"Invalid spectrogram shape for {source_path.name}: "
                    f"{tuple(spectrogram.shape)}; expected {expected_shape}"
                )
            self.conditioners[source_path] = spectrogram

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_path, target_path = self.records[index]
        target_audio, _ = librosa.load(
            target_path, sr=self.params.sample_rate, mono=True
        )
        return {
            "audio": torch.from_numpy(target_audio).float(),
            "spectrogram": self.conditioners[source_path],
            "source_path": str(source_path),
        }

    def collate_batch(
        self,
        minibatch: list[dict[str, torch.Tensor | str]],
    ) -> dict[str, torch.Tensor]:
        source_paths = {record["source_path"] for record in minibatch}
        if len(source_paths) != 1:
            raise ValueError(
                "Every fine-tuning batch must use one shared source audio."
            )

        crop_frames = self.params.crop_mel_frames
        hop_samples = self.params.hop_samples
        crop_samples = crop_frames * hop_samples
        usable_frames = min(
            min(record["spectrogram"].shape[-1] for record in minibatch),
            min(
                max(1, record["audio"].numel() // hop_samples)
                for record in minibatch
            ),
        )
        max_start = max(0, usable_frames - crop_frames)
        start_frame = random.randint(0, max_start) if max_start else 0
        end_frame = start_frame + crop_frames
        start_sample = start_frame * hop_samples

        audio_crops = []
        condition_crops = []
        for record in minibatch:
            conditioner = record["spectrogram"]
            target_audio = record["audio"]

            condition_crop = conditioner[:, start_frame:end_frame]
            if condition_crop.shape[-1] < crop_frames:
                condition_crop = nn.functional.pad(
                    condition_crop,
                    (0, crop_frames - condition_crop.shape[-1]),
                )

            target_crop = target_audio[
                start_sample : start_sample + crop_samples
            ]
            if target_crop.numel() < crop_samples:
                target_crop = nn.functional.pad(
                    target_crop,
                    (0, crop_samples - target_crop.numel()),
                )

            audio_crops.append(target_crop)
            condition_crops.append(condition_crop)

        return {
            "audio": torch.stack(audio_crops),
            "spectrogram": torch.stack(condition_crops),
        }


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_diffwave(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DiffWave, AttrDict, dict]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DiffWave checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' state: {checkpoint_path}")

    params = AttrDict(dict(base_params))
    params.override(checkpoint.get("params"))
    if params.unconditional:
        raise ValueError("A conditional DiffWave checkpoint is required.")

    preprocessing_keys = ("sample_rate", "n_mels", "n_fft", "hop_samples")
    for key in preprocessing_keys:
        if params[key] != base_params[key]:
            raise ValueError(
                f"Checkpoint {key}={params[key]} does not match "
                f"preprocess.py {key}={base_params[key]}."
            )

    model = DiffWave(params).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, params, checkpoint


def nested_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [nested_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(nested_to_cpu(item) for item in value)
    return value


def build_checkpoint_state(
    model: DiffWave,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    params: AttrDict,
    global_step: int,
    finetune_step: int,
    source_checkpoint: Path | None,
) -> dict:
    return {
        "step": global_step,
        "finetune_step": finetune_step,
        "model": nested_to_cpu(model.state_dict()),
        "optimizer": nested_to_cpu(optimizer.state_dict()),
        "params": dict(params),
        "scaler": nested_to_cpu(scaler.state_dict()),
        "source_checkpoint": (
            str(source_checkpoint) if source_checkpoint is not None else None
        ),
    }


def save_checkpoint(
    output_dir: Path,
    state: dict,
    global_step: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"weights-{global_step}.pt"
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, checkpoint_path)

    latest_path = output_dir / "weights.pt"
    temporary_latest = output_dir / "weights.pt.tmp"
    if temporary_latest.exists() or temporary_latest.is_symlink():
        temporary_latest.unlink()
    try:
        temporary_latest.symlink_to(checkpoint_path.name)
        os.replace(temporary_latest, latest_path)
    except OSError:
        if temporary_latest.exists() or temporary_latest.is_symlink():
            temporary_latest.unlink()
        shutil.copy2(checkpoint_path, latest_path)

    print(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def write_summary(
    writer: SummaryWriter,
    step: int,
    features: dict[str, torch.Tensor],
    loss: torch.Tensor,
    grad_norm: torch.Tensor,
    sample_rate: int,
) -> None:
    writer.add_audio("feature/audio", features["audio"][0], step, sample_rate)
    writer.add_image(
        "feature/spectrogram",
        torch.flip(features["spectrogram"][:1], [1]),
        step,
    )
    writer.add_scalar("train/loss", loss.item(), step)
    writer.add_scalar("train/grad_norm", float(grad_norm), step)
    writer.flush()


@torch.no_grad()
def generate_preview(
    model: DiffWave,
    spectrogram: torch.Tensor,
    device: torch.device,
    fast_sampling: bool,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    training_noise_schedule = np.asarray(
        model.params.noise_schedule, dtype=np.float64
    )
    inference_noise_schedule = np.asarray(
        model.params.inference_noise_schedule
        if fast_sampling
        else model.params.noise_schedule,
        dtype=np.float64,
    )
    training_alpha = 1 - training_noise_schedule
    training_alpha_cum = np.cumprod(training_alpha)
    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    diffusion_steps = []
    for inference_step in range(len(inference_noise_schedule)):
        for training_step in range(len(training_noise_schedule) - 1):
            if (
                training_alpha_cum[training_step + 1]
                <= alpha_cum[inference_step]
                <= training_alpha_cum[training_step]
            ):
                interpolation = (
                    training_alpha_cum[training_step] ** 0.5
                    - alpha_cum[inference_step] ** 0.5
                ) / (
                    training_alpha_cum[training_step] ** 0.5
                    - training_alpha_cum[training_step + 1] ** 0.5
                )
                diffusion_steps.append(training_step + interpolation)
                break
    diffusion_steps = np.asarray(diffusion_steps, dtype=np.float32)

    conditioner = spectrogram.unsqueeze(0).to(device)
    audio = torch.randn(
        1,
        model.params.hop_samples * conditioner.shape[-1],
        device=device,
    )
    for step in range(len(alpha) - 1, -1, -1):
        coefficient_1 = 1 / alpha[step] ** 0.5
        coefficient_2 = beta[step] / (1 - alpha_cum[step]) ** 0.5
        model_step = torch.tensor(
            [diffusion_steps[step]], device=device
        )
        predicted_noise = model(
            audio, model_step, conditioner
        ).squeeze(1)
        audio = coefficient_1 * (
            audio - coefficient_2 * predicted_noise
        )
        if step > 0:
            sigma = (
                (1 - alpha_cum[step - 1])
                / (1 - alpha_cum[step])
                * beta[step]
            ) ** 0.5
            audio += sigma * torch.randn_like(audio)
        audio = torch.clamp(audio, -1.0, 1.0)

    if was_training:
        model.train()
    return audio.cpu()


def fine_tune(
    source_targets: dict[Path, list[Path]],
    preprocess_dir: Path,
    pretrained_checkpoint: Path | None,
    output_dir: Path,
    device: torch.device,
    max_steps: int | None,
    fp16: bool,
    preview_every: int,
    preview_fast: bool,
) -> Path:
    if pretrained_checkpoint is None:
        params = AttrDict(dict(base_params))
        model = DiffWave(params).to(device)
        pretrained_state = {}
    else:
        model, params, pretrained_state = load_diffwave(
            pretrained_checkpoint, device
        )
    dataset = OneToManyScramblerDataset(source_targets, preprocess_dir, params)
    if len(dataset) < params.batch_size:
        raise ValueError(
            f"At least {params.batch_size} scrambler targets are required to form "
            f"one full DiffWave batch; got {len(dataset)}."
        )
    loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=dataset.collate_batch,
        generator=torch.Generator().manual_seed(SEED),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    use_fp16 = fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    noise_schedule = np.asarray(params.noise_schedule, dtype=np.float32)
    noise_level = torch.from_numpy(np.cumprod(1 - noise_schedule)).to(device)

    pretrained_step = int(pretrained_state.get("step", 0))
    finetune_step = 0
    writer = SummaryWriter(output_dir)
    progress = tqdm(total=max_steps, desc="Fine-tuning DiffWave", unit="step")
    preview_dir = output_dir / "previews"
    preview_conditioner = next(iter(dataset.conditioners.values()))
    if preview_every > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)
    model.train()

    def current_state() -> tuple[dict, int]:
        global_step = pretrained_step + finetune_step
        state = build_checkpoint_state(
            model,
            optimizer,
            scaler,
            params,
            global_step,
            finetune_step,
            pretrained_checkpoint,
        )
        return state, global_step

    interrupted = False
    try:
        while max_steps is None or finetune_step < max_steps:
            for features_cpu in loader:
                if max_steps is not None and finetune_step >= max_steps:
                    break

                features = {
                    key: value.to(device, non_blocking=True)
                    for key, value in features_cpu.items()
                }
                audio = features["audio"]
                spectrogram = features["spectrogram"]
                batch_count = audio.shape[0]
                diffusion_step = torch.randint(
                    0,
                    len(params.noise_schedule),
                    (batch_count,),
                    device=device,
                )
                selected_noise_level = noise_level[diffusion_step].unsqueeze(1)
                noise = torch.randn_like(audio)
                noisy_audio = (
                    selected_noise_level.sqrt() * audio
                    + (1.0 - selected_noise_level).sqrt() * noise
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    predicted_noise = model(
                        noisy_audio, diffusion_step, spectrogram
                    ).squeeze(1)
                    loss_per_scrambler = torch.mean(
                        torch.abs(noise - predicted_noise), dim=1
                    )
                    loss = loss_per_scrambler.mean()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), params.max_grad_norm or 1e9
                )
                scaler.step(optimizer)
                scaler.update()

                finetune_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item():.5f}")

                if finetune_step % SUMMARY_INTERVAL == 0:
                    write_summary(
                        writer,
                        finetune_step,
                        features_cpu,
                        loss,
                        grad_norm,
                        params.sample_rate,
                    )

                if (
                    preview_every > 0
                    and finetune_step % preview_every == 0
                ):
                    progress.write(
                        f"Generating preview at step {finetune_step}..."
                    )
                    preview_audio = generate_preview(
                        model,
                        preview_conditioner,
                        device,
                        fast_sampling=preview_fast,
                    )
                    preview_path = (
                        preview_dir
                        / f"preview_step_{finetune_step:08d}.wav"
                    )
                    torchaudio.save(
                        str(preview_path),
                        preview_audio,
                        params.sample_rate,
                    )
                    writer.add_audio(
                        "preview/current_model",
                        preview_audio[0],
                        finetune_step,
                        params.sample_rate,
                    )
                    writer.flush()
                    progress.write(f"Saved preview: {preview_path}")

                if finetune_step % CHECKPOINT_INTERVAL == 0:
                    state, global_step = current_state()
                    save_checkpoint(output_dir, state, global_step)
    except KeyboardInterrupt:
        interrupted = True
        print("\nFine-tuning interrupted; saving the current checkpoint.")
    finally:
        progress.close()
        writer.close()

    state, global_step = current_state()
    final_path = save_checkpoint(output_dir, state, global_step)
    metadata = {
        "source_checkpoint": (
            str(pretrained_checkpoint.resolve())
            if pretrained_checkpoint is not None
            else None
        ),
        "pretrained_step": pretrained_step,
        "finetune_steps": finetune_step,
        "final_global_step": global_step,
        "source_audio_count": len(source_targets),
        "scrambler_target_count": sum(len(paths) for paths in source_targets.values()),
        "batch_size": params.batch_size,
        "learning_rate": params.learning_rate,
        "shared_crop_per_batch": True,
        "loss_reduction": "mean_over_scrambler_batch",
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "summary_interval": SUMMARY_INTERVAL,
        "preview_every": preview_every,
        "preview_fast": preview_fast,
        "interrupted": interrupted,
    }
    (output_dir / "finetune_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return final_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune conditional DiffWave using one source audio condition "
            "and multiple pre-generated scrambler target files."
        )
    )
    parser.add_argument(
        "--input_audio",
        type=Path,
        required=True,
        help="Single source audio used as the DiffWave condition.",
    )
    parser.add_argument(
        "--scrambler_files",
        type=Path,
        nargs="+",
        required=True,
        help="One or more pre-generated scrambler target audio files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_dir / "diffwave" / "finetuned_scrambler",
        help="Fine-tuned checkpoint directory (default: ./diffwave/finetuned_scrambler).",
    )
    parser.add_argument(
        "--preprocess_dir",
        type=Path,
        default=None,
        help=(
            "Directory for inference-compatible .spec.npy files "
            "(default: OUTPUT_DIR/preprocessed)."
        ),
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=Path,
        default=project_dir / "diffwave" / "model" / "weights.pt",
        help=(
            "Initial conditional DiffWave weights.pt "
            "(default: ./diffwave/model/weights.pt)."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=positive_int,
        default=None,
        help="Maximum number of fine-tuning steps; default matches train.py (unlimited).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use 16-bit floating point operations for training.",
    )
    parser.add_argument(
        "--preview_every",
        type=non_negative_int,
        default=0,
        help=(
            "Generate a listening preview every N training steps; "
            "0 disables previews (default: 0)."
        ),
    )
    parser.add_argument(
        "--preview_fast",
        action="store_true",
        default=False,
        help="Use the fast sampling schedule for listening previews.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device (default: cuda).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_audio = args.input_audio.expanduser().resolve()
    scrambler_files = list(dict.fromkeys(
        path.expanduser().resolve() for path in args.scrambler_files
    ))
    output_dir = args.output_dir.expanduser().resolve()
    preprocess_dir = (
        args.preprocess_dir.expanduser().resolve()
        if args.preprocess_dir is not None
        else output_dir / "preprocessed"
    )
    pretrained_checkpoint = (
        args.pretrained_checkpoint.expanduser().resolve()
        if args.pretrained_checkpoint is not None
        else None
    )

    if not input_audio.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {input_audio}")
    if input_audio.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported input audio format: {input_audio}")
    for scrambler_file in scrambler_files:
        if not scrambler_file.is_file():
            raise FileNotFoundError(
                f"Scrambler audio does not exist: {scrambler_file}"
            )
        if scrambler_file.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported scrambler audio format: {scrambler_file}"
            )
    if (
        pretrained_checkpoint is not None
        and not pretrained_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            f"DiffWave checkpoint not found: {pretrained_checkpoint}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    source_files = [input_audio]
    source_targets = {input_audio: scrambler_files}

    print(f"Source audio: {input_audio}")
    print(f"Scrambler target files: {len(scrambler_files)}")
    print(f"Preprocessing source audio into: {preprocess_dir}")
    if pretrained_checkpoint is None:
        print("DiffWave initialization: random (training from step 0)")
    else:
        print(f"DiffWave initialization: {pretrained_checkpoint}")
    preprocess_sources(source_files, preprocess_dir)

    print(f"DiffWave batch size: {base_params.batch_size}")
    print(f"DiffWave learning rate: {base_params.learning_rate}")
    final_checkpoint = fine_tune(
        source_targets=source_targets,
        preprocess_dir=preprocess_dir,
        pretrained_checkpoint=pretrained_checkpoint,
        output_dir=output_dir,
        device=device,
        max_steps=args.max_steps,
        fp16=args.fp16,
        preview_every=args.preview_every,
        preview_fast=args.preview_fast,
    )
    print(f"Fine-tuning completed. Latest checkpoint: {final_checkpoint}")
    print(f"Inference model directory: {output_dir}")


if __name__ == "__main__":
    main()
