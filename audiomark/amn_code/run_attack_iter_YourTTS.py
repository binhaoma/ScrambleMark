from __future__ import print_function

import sys

import exp_setup
from pathlib import Path
from exp.ExpAdaptiveIter import ExpAdaptiveIter, ExpConfig as ExpAdaptiveIterCfg
from exp.ExpSpeakerAdaptYourTTS import ExpSpeakerAdaptYourTTS, ExpConfig as ExpSpeakerAdaptYourTTSCfg


def main():
    """
    We run an adaptive iteration attack:
    An adversary tries to recover a previous version if watermarks are detected.
    """

    (wm_exp_dir, speakers_wm_lst, benign_encoded_wm, tgt_name, enrol_speech,
     tgt_audio_dic_lst, speaker_wm_dic, gen_sentences_lst, run_times) = exp_setup.setup_adaptive_iter_exps(exp_cfg)

    out_dir = Path(exp_cfg.out_dir)
    exp_dir = out_dir.joinpath("ExpAdaptiveIterYourTTS")
    exp_dir.mkdir(exist_ok=True)

    for cur_run in range(run_times):
        cfg = ExpAdaptiveIterCfg(
            wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir,
            wm_exp_dir=wm_exp_dir,

            speakers_wm_lst=speakers_wm_lst,
            benign_encoded_wm=benign_encoded_wm,

            sr=exp_cfg.sr,
            audio_sec_len=exp_cfg.sr,  # 1 second in the same way as embedding

            tgt_name=tgt_name,
            tgt_wm=speaker_wm_dic["encoded_wm"],
            enrol_speech=enrol_speech,  # enrolment speech for speaker verification

            exp_speaker_adapt_exp_cfg=ExpSpeakerAdaptYourTTSCfg(
                run_name=f"adapt_to_{tgt_name}",
                vctk_dir=Path(exp_cfg.data_dir).joinpath("vctk"),

                restore_path=Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(
                    "exp1_vctk/best_model_latest.pth.tar"),

                tgt_audio_dic_lst=tgt_audio_dic_lst,
                tgt_speaker_name=tgt_name,

                gen_sentences_lst=gen_sentences_lst,

                iter_lsts=exp_cfg.adap_iter_attack_iters,

                speaker_encoder_checkpoint_path=str(Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(exp_cfg.speaker_encoder_checkpoint_relpath)),
                speaker_encoder_config_path=str(Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(exp_cfg.speaker_encoder_config_relpath)),
            ),
            exp_speaker_adapt_exp_class=ExpSpeakerAdaptYourTTS,
        )

        exp = ExpAdaptiveIter(exp_dir.joinpath(f"run_{cur_run}"), cfg)
        exp.run()

    # draw plots here
    exp_setup.plot_adaptive_iter_exps(exp_dir, run_times, "YourTTS")


if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    main()

    sys.exit()






