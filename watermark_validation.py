import argparse

from watermark_utils import WM_validation


def main():
    parser = argparse.ArgumentParser(description="Validate watermark robustness and calculate ASR")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["audioseal", "wavmark", "audiomarknet"],
        help="Watermark model name"
    )

    parser.add_argument(
        "--watermark_dir",
        type=str,
        required=True,
        help="Root directory of the watermark experiment"
    )

    parser.add_argument(
        "--audio_subdir",
        type=str,
        default="data",
        help="Subdirectory containing audio files, e.g. data or scramblemark"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used for validation, e.g. cuda, cuda:0, or cpu"
    )

    args = parser.parse_args()

    WM_validation(
        model_name=args.model_name,
        watermark_dir=args.watermark_dir,
        audio_subdir=args.audio_subdir,
        device=args.device
    )


if __name__ == "__main__":
    main()