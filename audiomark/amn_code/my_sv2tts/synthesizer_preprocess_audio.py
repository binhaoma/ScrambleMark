from my_sv2tts.preprocess import preprocess_dataset
from synthesizer.hparams import hparams
from my_sv2tts.argutils import print_args
from pathlib import Path
import argparse


def run(datasets_root, datasets_name, subfolders):
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--datasets_root", type=Path, default=datasets_root, help=\
        "Path to the directory containing your LibriSpeech/TTS datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help=\
        "Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    parser.add_argument("--no_alignments", default=True, help=\
        "Use this option when dataset does not include alignments\
        (these are used to split long audio files into sub-utterances.)")
    parser.add_argument("--datasets_name", type=str, default=datasets_name, help=\
        "Name of the dataset directory to process.")
    parser.add_argument("--subfolders", type=str, default=subfolders, help=\
        "Comma-separated list of subfolders to process inside your dataset directory")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer")

    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the dataset
    print_args(args, parser)
    args.hparams = hparams.parse(args.hparams)

    # keep all data
    args.hparams.utterance_min_duration = 0.0
    args.hparams.clip_mels_length = False

    preprocess_dataset(**vars(args))
