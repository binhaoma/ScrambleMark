from __future__ import print_function

import pickle
import sys
import exp_setup

from pathlib import Path

from exp.ExpSpeakerAdaptYourTTS import ExpSpeakerAdaptYourTTS, ExpConfig as ExpSpeakerAdaptYourTTSCfg
from exp.ExpEmbedWatermark import get_existing_wm_samples


def get_purified_wm_samples(tgt_name, org_exp_dir, purified_dir):
    tgt_audio_dic_lst = get_existing_wm_samples(tgt_name, org_exp_dir)

    # we need to change watermarked audio to purified watermarked audio
    for sample in tgt_audio_dic_lst:
        audio_file = sample["audio_file"]
        sample["audio_file"] = purified_dir.joinpath(audio_file.parent.name).joinpath(audio_file.stem + "_purified.wav")

    return tgt_audio_dic_lst


def main():
    out_dir = Path(exp_cfg.out_dir)

    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=exp_cfg.wm_length)
    speaker_names_lst = [x["speaker"] for x in speakers_wm_lst]

    gen_sentences_lst = exp_setup.get_gen_sentences_lst(exp_cfg)

    for tgt_name in speaker_names_lst:

        config = ExpSpeakerAdaptYourTTSCfg(
            run_name=f"adapt_to_{tgt_name}",
            vctk_dir=Path(exp_cfg.data_dir).joinpath("vctk"),

            restore_path=Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(
                "exp1_vctk/best_model_latest.pth.tar"),

            tgt_audio_dic_lst=get_purified_wm_samples(tgt_name,
                                                      org_exp_dir=out_dir.joinpath("ExpEmbedWatermark"),
                                                      purified_dir=out_dir.joinpath("ExpAdaptiveAutoEncoder/tgt_purified")),
            tgt_speaker_name=tgt_name,

            iter_lsts=exp_cfg.speaker_adapt_iters,
            gen_sentences_lst=gen_sentences_lst,

            speaker_encoder_checkpoint_path=str(Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(exp_cfg.speaker_encoder_checkpoint_relpath)),
            speaker_encoder_config_path=str(Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(exp_cfg.speaker_encoder_config_relpath)),
        )

        exp_dir = out_dir.joinpath("ExpSpeakerAdaptYourTTS_Autoencoder")
        exp = ExpSpeakerAdaptYourTTS(exp_dir.joinpath(f"adapt_to_{tgt_name}"), config)
        exp.run()


if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    main()

    sys.exit()






