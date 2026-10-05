import os
import argparse
from pathlib import Path

import torch
import torchaudio
import numpy as np
import python_stretch as ps

from GT_utils import *
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


def generate_ground_truth_for_audio(
    wav_path,
    encoder,
    decoder,
    threshold=0.8,
    define_number=10,
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

            if candidate_acc <= threshold and candidate_recover_acc <= threshold:
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
        print(f"  No successful ground-truth candidate found for: {wav_path}")
        return None

    best_candidate = min(successful_candidates, key=lambda x: x["loss"])

    print(f"  Successful candidates: {len(successful_candidates)}/{define_number}")
    print(f"  Best loss: {best_candidate['loss']:.6f}")
    print(f"  Best watermark accuracy: {best_candidate['wm_acc']:.4f}")
    print(f"  Best recovery accuracy: {best_candidate['recover_acc']:.4f}")

    return best_candidate


def generate_ground_truth_directory(
    input_dir,
    output_dir=None,
    threshold=0.8,
    define_number=10,
    max_opt_iters=20,
    sample_rate=22050
):
    input_dir = os.path.abspath(input_dir)

    if output_dir is None:
        output_dir = f"{input_dir.rstrip(os.sep)}_ground_truth"

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

        best_candidate = generate_ground_truth_for_audio(
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
    print("Ground-truth generation completed.")
    print(f"Total audio files: {len(audio_files)}")
    print(f"Successfully generated: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground-truth samples for all audio files in a directory."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input audio files."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for generated ground-truth audio. Default: <input_dir>_ground_truth"
    )

    parser.add_argument(
        "--num_candidates",
        type=int,
        default=10,
        help="Number of ground-truth candidates generated for each audio file. Default: 10"
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

    args = parser.parse_args()

    generate_ground_truth_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        define_number=args.num_candidates,
        max_opt_iters=args.max_opt_iters,
        sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()