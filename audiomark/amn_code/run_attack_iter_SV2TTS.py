from __future__ import print_function

import sys

import numpy as np

import exp_setup
from pathlib import Path
from exp.ExpAdaptiveIter import ExpAdaptiveIter, ExpConfig as ExpAdaptiveIterCfg
from exp.ExpSpeakerAdaptSV2TTS import ExpSpeakerAdaptSV2TTS, ExpConfig as ExpSpeakerAdaptSV2TTSCfg
from my_utils import utils


def main():
    """
    We run an adaptive iteration attack.
    An adversary tries to recover a previous version if watermarks are detected.
    """
    (wm_exp_dir, speakers_wm_lst, benign_encoded_wm, tgt_name, enrol_speech,
     tgt_audio_dic_lst, speaker_wm_dic, gen_sentences_lst, run_times) = \
        exp_setup.setup_adaptive_iter_exps(exp_cfg)

    out_dir = Path(exp_cfg.out_dir)
    SV2TTS_model_dir = Path(exp_cfg.SV2TTS_pretrained_dir)

    exp_dir = out_dir.joinpath("ExpAdaptiveIterSV2TTS")
    exp_dir.mkdir(exist_ok=True)

    for cur_run in range(run_times):
        cfg = ExpAdaptiveIterCfg(
            wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir,
            wm_exp_dir=wm_exp_dir,

            speakers_wm_lst=speakers_wm_lst,
            benign_encoded_wm=benign_encoded_wm,

            sr=exp_cfg.sr,
            audio_sec_len=exp_cfg.sr,  # 1 second in the same way as embedding watermarks

            tgt_name=tgt_name,
            tgt_wm=speaker_wm_dic["encoded_wm"],
            enrol_speech=enrol_speech,  # enrolment speech for speaker verification

            exp_speaker_adapt_exp_cfg=ExpSpeakerAdaptSV2TTSCfg(
                tgt_name=tgt_name,
                wm_exp_dir=wm_exp_dir,
                restricted_wm_fname_lst=[],

                restore_encoder_path=SV2TTS_model_dir.joinpath("encoder.pt"),
                restore_tacotron_path=SV2TTS_model_dir.joinpath("synthesizer.pt"),
                restore_vocoder_path=SV2TTS_model_dir.joinpath("vocoder.pt"),

                iter_lsts=exp_cfg.adap_iter_attack_iters,

                gen_sentences_lst=gen_sentences_lst,
            ),
            exp_speaker_adapt_exp_class=ExpSpeakerAdaptSV2TTS,
        )

        exp = ExpAdaptiveIter(exp_dir.joinpath(f"run_{cur_run}"), cfg)
        exp.run()

    # draw plots here
    exp_setup.plot_adaptive_iter_exps(exp_dir, run_times, "SV2TTS")


if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    main()

    sys.exit()






