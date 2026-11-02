import os
import argparse
import hashlib
from pathlib import Path

import librosa
import torch
import torchaudio
import numpy as np
import python_stretch as ps

from scrambler_utils import *
from losses import MultiMelSpectrogramLoss


mel_loss = MultiMelSpectrogramLoss()


def compute_loss(generated_audio, reference_audio):
    generated_audio = torch.as_tensor(generated_audio, dtype=torch.float32)
    reference_audio = torch.as_tensor(reference_audio, dtype=torch.float32)

    if generated_audio.dim() == 1:
        generated_audio = generated_audio.unsqueeze(0)

    if reference_audio.dim() == 1:
        reference_audio = reference_audio.unsqueeze(0)

    loss = mel_loss(generated_audio, reference_audio)

    if torch.is_tensor(loss):
        return loss.detach().mean().item()

    return float(loss)


def prepare_audio_for_save(audio):
    audio = torch.as_tensor(audio, dtype=torch.float32).detach().cpu()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    return audio


def generate_scrambler_for_audio(
    wav_path,
    encoder,
    decoder,
    threshold=0.8,
    define_number=1,
    max_opt_iters=20,
    sample_rate=22050
):
    print(f"\nProcessing: {wav_path}")

    wmaudio, oriaudio, msg = embedding_surrogate(wav_path, encoder, decoder)

    successful_candidates = []

    for number in range(define_number):
        print(f"  Candidate {number + 1}/{define_number}")

        bands_num_initial = 10
        current_boundaries = generate_uniform_freq_points(sample_rate, bands_num_initial)
        current_ratios = generate_random_ratios(bands_num_initial, cents_range=(60, 85))

        candidate_audio = None
        candidate_acc = None
        candidate_recover_acc = None
        candidate_success = False

        for i in range(max_opt_iters):
            new_boundaries = generate_bounded_random_cuts(current_boundaries, max_cuts=3)
            new_ratios = inherit_ratios(current_boundaries, current_ratios, new_boundaries)

            audio_new, best_acc, best_recover_acc, optimized_ratios, final_boundaries = cmaes_optimize(
                wmaudio,
                oriaudio,
                msg,
                decoder,
                threshold_wm=threshold,
                threshold_recover=threshold,
                max_iter=1,
                ratios=new_ratios,
                boundaries=new_boundaries,
                iter=i
            )

            current_boundaries = final_boundaries
            current_ratios = optimized_ratios

            candidate_audio = audio_new
            candidate_acc = float(best_acc)
            candidate_recover_acc = float(best_recover_acc)

            if (
                candidate_acc <= threshold
                and candidate_recover_acc <= threshold
            ):
                candidate_success = True
                print(
                    f"    Success at iteration {i + 1}: "
                    f"wm_acc={candidate_acc:.4f}, "
                    f"recover_acc={candidate_recover_acc:.4f}"
                )
                break

        if not candidate_success:
            print(
                f"    Candidate failed: "
                f"wm_acc={candidate_acc:.4f}, "
                f"recover_acc={candidate_recover_acc:.4f}"
            )
            continue

        loss = compute_loss(candidate_audio, oriaudio)

        successful_candidates.append({
            "audio": prepare_audio_for_save(candidate_audio),
            "loss": loss,
            "wm_acc": candidate_acc,
            "recover_acc": candidate_recover_acc
        })

        print(f"    Mel loss: {loss:.6f}")

    if not successful_candidates:
        print(f"  No successful scrambler candidate found for: {wav_path}")
        return None

    best_candidate = min(successful_candidates, key=lambda x: x["loss"])

    print(f"  Successful candidates: {len(successful_candidates)}/{define_number}")
    print(f"  Best loss: {best_candidate['loss']:.6f}")
    print(f"  Best watermark accuracy: {best_candidate['wm_acc']:.4f}")
    print(f"  Best recovery accuracy: {best_candidate['recover_acc']:.4f}")

    return best_candidate


def generate_scrambler_directory(
    input_dir,
    output_dir=None,
    threshold=0.8,
    define_number=10,
    max_opt_iters=20,
    sample_rate=22050
):
    input_dir = os.path.abspath(input_dir)

    if output_dir is None:
        output_dir = f"{input_dir.rstrip(os.sep)}_scrambler"

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    audio_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg"))
    ]

    audio_files.sort()

    if not audio_files:
        raise ValueError(f"No audio files found in: {input_dir}")

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of audio files: {len(audio_files)}")
    print(f"Candidates per audio: {define_number}")
    print(f"Threshold: {threshold}")

    print("\nLoading encoder and decoder...")
    encoder, decoder = get_encoder_decoder()
    print("Models loaded.")

    success_count = 0
    failure_count = 0

    for index, audio_name in enumerate(audio_files):
        print("\n" + "=" * 80)
        print(f"[{index + 1}/{len(audio_files)}] {audio_name}")
        print("=" * 80)

        audio_path = os.path.join(input_dir, audio_name)

        best_candidate = generate_scrambler_for_audio(
            wav_path=audio_path,
            encoder=encoder,
            decoder=decoder,
            threshold=threshold,
            define_number=define_number,
            max_opt_iters=max_opt_iters,
            sample_rate=sample_rate
        )

        if best_candidate is None:
            failure_count += 1
            continue

        output_path = os.path.join(output_dir, audio_name)

        torchaudio.save(
            output_path,
            best_candidate["audio"],
            sample_rate
        )

        success_count += 1

        print(f"Saved: {output_path}")

    print("\n" + "=" * 80)
    print("Scrambler generation completed.")
    print(f"Total audio files: {len(audio_files)}")
    print(f"Successfully generated: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


def audio_hash(audio):
    waveform = torch.as_tensor(audio, dtype=torch.float32).detach().cpu()
    waveform = waveform.flatten().clamp(-1.0, 1.0).numpy()
    quantized = np.rint(waveform * 32767.0).astype("<i2")
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def saved_audio_hash(path, sample_rate):
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio_hash(torch.from_numpy(waveform))


def generate_multiple_scramblers_for_audio(
    input_audio,
    output_dir,
    num_scramblers=32,
    threshold=0.8,
    define_number=10,
    max_opt_iters=20,
    max_attempts=96,
    sample_rate=22050
):
    input_audio = Path(input_audio).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_audio.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {input_audio}")

    existing_targets = sorted(output_dir.glob("scrambler_*.wav"))
    unique_hashes = set()
    distinct_targets = []
    for target_path in existing_targets:
        target_hash = saved_audio_hash(target_path, sample_rate)
        if target_hash not in unique_hashes:
            unique_hashes.add(target_hash)
            distinct_targets.append(target_path)

    existing_indices = [
        int(path.stem.removeprefix("scrambler_"))
        for path in existing_targets
        if path.stem.removeprefix("scrambler_").isdigit()
    ]
    next_index = max(existing_indices, default=0) + 1

    if len(distinct_targets) >= num_scramblers:
        print(
            f"Found {len(distinct_targets)} existing distinct scrambler targets; "
            "generation skipped."
        )
        return distinct_targets[:num_scramblers]

    print("\nLoading encoder and decoder...")
    encoder, decoder = get_encoder_decoder()
    print("Models loaded.")

    attempts = 0
    while len(distinct_targets) < num_scramblers and attempts < max_attempts:
        attempts += 1
        print("\n" + "=" * 80)
        print(
            f"Generating scrambler {len(distinct_targets) + 1}/{num_scramblers} "
            f"(attempt {attempts}/{max_attempts})"
        )
        print("=" * 80)

        candidate = generate_scrambler_for_audio(
            wav_path=str(input_audio),
            encoder=encoder,
            decoder=decoder,
            threshold=threshold,
            define_number=define_number,
            max_opt_iters=max_opt_iters,
            sample_rate=sample_rate
        )
        if candidate is None:
            continue

        candidate_audio = prepare_audio_for_save(candidate["audio"])
        if audio_hash(candidate_audio) in unique_hashes:
            print("Duplicate scrambler generated; retrying.")
            continue

        target_path = output_dir / f"scrambler_{next_index:03d}.wav"
        torchaudio.save(str(target_path), candidate_audio, sample_rate)
        target_hash = saved_audio_hash(target_path, sample_rate)
        if target_hash in unique_hashes:
            target_path.unlink()
            print("Duplicate scrambler after saving; retrying.")
            continue

        unique_hashes.add(target_hash)
        distinct_targets.append(target_path)
        next_index += 1
        print(f"Saved: {target_path}")

    if len(distinct_targets) < num_scramblers:
        raise RuntimeError(
            f"Only generated {len(distinct_targets)}/{num_scramblers} distinct "
            f"scrambler targets after {attempts} attempts."
        )

    print("\n" + "=" * 80)
    print(f"Ready scrambler targets: {len(distinct_targets[:num_scramblers])}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    return distinct_targets[:num_scramblers]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one scrambler per audio directory entry, or multiple distinct "
            "scrambler targets for one audio file."
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing input audio files."
    )
    input_group.add_argument(
        "--input_audio",
        type=str,
        help="One input audio for multi-target scrambler generation."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for generated scrambler audio. Default: <input_dir>_scrambler"
    )

    parser.add_argument(
        "--num_candidates",
        type=int,
        default=10,
        help="Number of scrambler candidates generated for each audio file. Default: 10"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Watermark and recovery accuracy threshold. Default: 0.8"
    )

    parser.add_argument(
        "--max_opt_iters",
        type=int,
        default=20,
        help="Maximum optimization iterations for each candidate. Default: 20"
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Output sample rate. Default: 22050"
    )

    parser.add_argument(
        "--num_scramblers",
        type=int,
        default=32,
        help="Number of distinct scrambler files in --input_audio mode. Default: 32"
    )

    parser.add_argument(
        "--max_attempts",
        type=int,
        default=96,
        help="Maximum generation attempts in --input_audio mode. Default: 96"
    )

    args = parser.parse_args()

    if args.num_candidates <= 0:
        parser.error("--num_candidates must be greater than zero")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.max_opt_iters <= 0:
        parser.error("--max_opt_iters must be greater than zero")
    if args.sample_rate <= 0:
        parser.error("--sample_rate must be greater than zero")

    if args.input_audio is not None:
        if args.output_dir is None:
            parser.error("--output_dir is required with --input_audio")
        if args.num_scramblers <= 0:
            parser.error("--num_scramblers must be greater than zero")
        if args.max_attempts < args.num_scramblers:
            parser.error("--max_attempts must be at least --num_scramblers")

        generate_multiple_scramblers_for_audio(
            input_audio=args.input_audio,
            output_dir=args.output_dir,
            num_scramblers=args.num_scramblers,
            threshold=args.threshold,
            define_number=args.num_candidates,
            max_opt_iters=args.max_opt_iters,
            max_attempts=args.max_attempts,
            sample_rate=args.sample_rate
        )
    else:
        generate_scrambler_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            define_number=args.num_candidates,
            max_opt_iters=args.max_opt_iters,
            sample_rate=args.sample_rate
        )


if __name__ == "__main__":
    main()
