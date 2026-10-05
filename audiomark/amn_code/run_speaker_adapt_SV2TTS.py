from __future__ import print_function

import pickle
import sys
import exp_setup

from pathlib import Path

from exp.ExpSpeakerAdaptSV2TTS import ExpSpeakerAdaptSV2TTS, ExpConfig


def main():
    exp_dir = Path(exp_cfg.out_dir).joinpath("ExpSpeakerAdaptSV2TTS")
    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=exp_cfg.wm_length)
    speaker_names_lst = [x["speaker"] for x in speakers_wm_lst]

    gen_sentences_lst = exp_setup.get_gen_sentences_lst(exp_cfg)

    out_dir = Path(exp_cfg.out_dir)

    for tgt_name in speaker_names_lst:

        SV2TTS_model_dir = Path(exp_cfg.SV2TTS_pretrained_dir)
        config = ExpConfig(
            tgt_name=tgt_name,
            wm_exp_dir=out_dir.joinpath("ExpEmbedWatermark"),

            restore_encoder_path=SV2TTS_model_dir.joinpath("encoder.pt"),
            restore_tacotron_path=SV2TTS_model_dir.joinpath("synthesizer.pt"),
            restore_vocoder_path=SV2TTS_model_dir.joinpath("vocoder.pt"),

            iter_lsts=exp_cfg.speaker_adapt_iters,

            gen_sentences_lst=gen_sentences_lst,
        )

        exp = ExpSpeakerAdaptSV2TTS(exp_dir.joinpath(f"adapt_to_{tgt_name}"), config)
        exp.run()


if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    main()

    sys.exit()






