from __future__ import print_function

import pickle
import sys
import exp_setup

from pathlib import Path
import numpy as np
from exp.ExpSpeakerAdaptYourTTS import ExpSpeakerAdaptYourTTS, ExpConfig as ExpSpeakerAdaptYourTTSCfg


def get_existing_wm_samples(tgt_name, exp_dir):
    """
    Original file path will be replaced by watermarked file path.
    wm_ratio: the ratio of watermarked audio in the return set.
        1.0 -> all returned audios are watermarked
        0.1 -> 10% returned audios are watermarked
    """

    sample_saved_path = exp_dir.joinpath(f"exp_rlt_dic.bin")
    assert sample_saved_path.exists(), f"{exp_dir.name} must be run before"

    with open(sample_saved_path, 'rb') as handle:
        exp_rlt_dic = pickle.load(handle)

    tgt_samples = exp_rlt_dic["tgt_samples"]
    for sample in tgt_samples:
        assert Path(sample["audio_file"]).suffix == ".flac", "our watermarked files are .wav"
        assert sample["audio_file_wm"].parent.parent.name=="ExpTimbreWM"
        assert sample["audio_file_wm"].suffix == ".wav", "our watermarked files are .wav"

        sample["audio_file_org"] = sample["audio_file"]
        # redirect the audio file to watermarked file for speaker adaptation
        sample["audio_file"] = sample["audio_file_wm"]

    return tgt_samples

def get_adapt_iters():
    adapt_iters = exp_cfg.speaker_adapt_iters + [1001]
    return adapt_iters


def main():
    out_dir = Path(exp_cfg.out_dir)

    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=10)
    speaker_names_lst = [x["speaker"] for x in speakers_wm_lst]

    gen_sentences_lst = exp_setup.get_gen_sentences_lst(exp_cfg)

    for tgt_name in speaker_names_lst:

        config = ExpSpeakerAdaptYourTTSCfg(
            run_name=f"adapt_to_{tgt_name}",
            vctk_dir=Path(exp_cfg.data_dir).joinpath("vctk"),

            restore_path=Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(
                "exp1_vctk/best_model_latest.pth.tar"),

            tgt_audio_dic_lst=get_existing_wm_samples(tgt_name, out_dir.joinpath(f"ExpTimbreWM/{tgt_name}")),
            tgt_speaker_name=tgt_name,

            iter_lsts=get_adapt_iters(),
            gen_sentences_lst=gen_sentences_lst,

            speaker_encoder_checkpoint_path=str(Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(exp_cfg.speaker_encoder_checkpoint_relpath)),
            speaker_encoder_config_path=str(Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(exp_cfg.speaker_encoder_config_relpath)),
        )

        exp_dir = out_dir.joinpath("ExpTimbreWMYourTTS")
        exp = ExpSpeakerAdaptYourTTS(exp_dir.joinpath(f"adapt_to_{tgt_name}"), config)
        exp.run()

    return


if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    main()

    sys.exit()






