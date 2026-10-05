from __future__ import print_function

import sys
import exp_setup
from pathlib import Path
from exp.ExpEmbedWatermark import ExpEmbedWatermark, ExpConfig as ExpEmbedWatermarkCfg

g_speakers_seed_dic = {
    "benign": 9000,

    "VCTK_p261": 9101,       # female
    "VCTK_p225": 9202,       # female

    "VCTK_p245": 9309,       # male
    "VCTK_p326": 9410,        # male

    "Obama": 9521,
}

def main():
    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(
        exp_cfg, exp_cfg.wm_length,
        speakers_seed_dic=g_speakers_seed_dic
    )

    out_dir = Path(exp_cfg.out_dir)

    cfg = ExpEmbedWatermarkCfg(
        extra_audio_lst_dic=exp_setup.process_Obama_voice(exp_cfg),     # add Obama's speech

        speakers_wm_lst=speakers_wm_lst,
        vctk_dir=exp_setup.get_vctk_dir(exp_cfg),

        benign_org_wm=benign_org_wm,
        benign_encoded_wm=benign_encoded_wm,

        wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir,

        aug_normal_prob=exp_cfg.wmnet_aug_normal_prob,
        aug_normal_scale=exp_cfg.wmnet_aug_normal_scale,
    )

    exp = ExpEmbedWatermark(out_dir.joinpath("ExpEmbedWatermark_Obama"), cfg)
    exp.run()


if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    main()

    sys.exit()






