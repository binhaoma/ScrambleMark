from __future__ import print_function

import sys
import exp_setup
from pathlib import Path
from exp.ExpAdaptiveAutoEncoder import ExpAdaptiveAutoEncoder, ExpConfig
from run_wm_speech_diff_set import g_speakers_seed_dic


def main():
    """
    We run an adaptive autoencoder attack, which tries to remove watermarks using denoise autoencoder.
    """

    train_speakers_wm_lst, train_benign_org_wm, train_benign_encoded_wm = exp_setup.get_speakers_and_wm(
        exp_cfg, exp_cfg.wm_length,
        speakers_seed_dic=g_speakers_seed_dic
    )

    tgt_speakers_wm_lst, tgt_benign_org_wm, tgt_benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, exp_cfg.wm_length)

    out_dir = Path(exp_cfg.out_dir)

    cfg = ExpConfig(
        wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir,

        train_speakers_wm_lst=train_speakers_wm_lst,
        train_wm_exp_dir=out_dir.joinpath("ExpEmbedWatermark_DiffSet"),
        train_benign_encoded_wm=train_benign_encoded_wm,

        tgt_speakers_wm_lst=tgt_speakers_wm_lst,
        tgt_wm_exp_dir=out_dir.joinpath("ExpEmbedWatermark"),
        tgt_benign_encoded_wm=tgt_benign_encoded_wm,

        sr=exp_cfg.sr,
        audio_sec_len=exp_cfg.sr,     # 1 second in the same way as embedding
    )

    exp = ExpAdaptiveAutoEncoder(out_dir.joinpath("ExpAdaptiveAutoEncoder"), cfg)
    exp.run()


if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    main()

    sys.exit()






