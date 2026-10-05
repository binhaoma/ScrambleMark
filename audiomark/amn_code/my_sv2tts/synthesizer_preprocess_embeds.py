from my_sv2tts.preprocess import create_embeddings
from my_sv2tts.argutils import print_args
from pathlib import Path
import argparse


def run(synthesizer_root, encoder_model_fpath):
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--synthesizer_root", type=Path, default=synthesizer_root, help=\
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-e", "--encoder_model_fpath", type=Path, default=encoder_model_fpath,
                        help="Path your trained encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=1, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()

    # Preprocess the dataset
    print_args(args, parser)
    create_embeddings(**vars(args))
