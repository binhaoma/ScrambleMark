from __future__ import print_function

import sys
from datetime import datetime

import exp_setup
from pathlib import Path
from exp.ExpEmbedWatermark import ExpEmbedWatermark, ExpConfig as ExpEmbedWatermarkCfg
import time

# Use the same original dataset, but with a different set of watermarks
g_speakers_seed_dic = {
        "benign": 8017,

        "VCTK_p261": 80100,       # female
        "VCTK_p225": 80205,       # female
        "VCTK_p294": 80300,       # female
        "VCTK_p347": 80400,       # male
        "VCTK_p238": 80500,       # female
        "VCTK_p234": 80600,       # female
        "VCTK_p248": 80700,       # female
        "VCTK_p335": 80800,       # female
        "VCTK_p245": 80900,       # male
        "VCTK_p326": 801000,      # male
        "VCTK_p302": 801100,      # male
}

def pirate_tgt_samples_ready_callback(tgt_samples):
    print(f"Redirecting the original audios to the watermarked audios !!!")
    pre_wm_exp_dir = Path(exp_cfg.out_dir).joinpath("ExpEmbedWatermark")

    for sample, encoded_wm in tgt_samples:
        org_audio_path = Path(sample["audio_file"])
        assert org_audio_path.suffix == ".flac"
        pre_wm_path = pre_wm_exp_dir.joinpath(sample["speaker_name"]).joinpath(org_audio_path.stem+".wav")
        sample["audio_file"] = str(pre_wm_path)

    return


def main():
    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(
        exp_cfg, exp_cfg.wm_length,
        speakers_seed_dic=g_speakers_seed_dic,
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

        tgt_samples_ready_callback=pirate_tgt_samples_ready_callback,
    )

    exp = ExpEmbedWatermark(out_dir.joinpath("ExpEmbedWatermark_Pirate"), cfg)

    # Let's record the time
    start_seconds = time.time()

    exp.run()

    seconds_passed = time.time() - start_seconds

    # save the time passed if not saved before
    time_path = exp.out_dir.joinpath("run_time.txt")
    if not time_path.exists():
        with open(time_path, "w") as f:
            print(f"Experiment date = {datetime.now()}", file=f)
            print(f"Run time = {seconds_passed/60:.1f} minutes", file=f)

    return




if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    main()

    sys.exit()






