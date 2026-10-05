from pathlib import Path

from synthesizer.hparams import hparams
from synthesizer.train import train
from my_sv2tts.argutils import print_args
import argparse


def run(syn_dir, models_dir, model_ckpt_path, on_train_step_start):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="", help= \
        "Name for this model. By default, training outputs will be stored to saved_models/<run_id>/. If a model state "
        "from the same run ID was previously saved, the training will restart from there. Pass -f to overwrite saved "
        "states and restart from scratch.")
    parser.add_argument("--syn_dir", type=Path, default=syn_dir, help= \
        "Path to the synthesizer directory that contains the ground truth mel spectrograms, "
        "the wavs and the embeds.")
    parser.add_argument("-m", "--models_dir", type=Path, default=models_dir, help=\
        "Path to the output directory that will contain the saved model weights and the logs.")
    parser.add_argument("-s", "--save_every", type=int, default=0, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=0, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model and restart from scratch.")
    parser.add_argument("--hparams", default="", help=\
        "Hyperparameter overrides as a comma-separated list of name=value pairs")
    args = parser.parse_args()
    print_args(args, parser)

    args.hparams = hparams.parse(args.hparams)

    # keep all data
    args.hparams.utterance_min_duration = 0.0
    args.hparams.clip_mels_length = False

    # Run the training
    train(**vars(args), model_ckpt_path=model_ckpt_path, on_train_step_start=on_train_step_start)
