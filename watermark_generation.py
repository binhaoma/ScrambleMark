import argparse

from watermark_utils import watermark_audio_generation


def main():
    parser = argparse.ArgumentParser(
        description="Generate watermarked audio"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["audioseal", "wavmark", "audiomarknet"],
        help="Watermark model name"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input audio files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for generated watermarked audio and messages"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device, e.g. cuda, cuda:0, cpu"
    )

    args = parser.parse_args()

    watermark_audio_generation(
        model_name=args.model_name,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()