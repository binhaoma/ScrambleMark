from __future__ import print_function

import sys
from datetime import datetime

import exp_setup
from pathlib import Path

from models.WatermarkNet import WatermarkNet
from models.ModelTrainer import ModelTrainer
from my_utils import utils
import run_evaluate


import time
import torch
import numpy as np

from calflops import calculate_flops
from torchvision import models


def main():

    # Let the model extract watermarks from all the 11 watermarked data
    # and calculate the time passed

    exp_dir = Path(exp_cfg.out_dir).joinpath(f"ExpComputationalCosts")
    exp_dir.mkdir(exist_ok=True)

    rlts_path = exp_dir.joinpath("costs.txt")
    if rlts_path.exists():
        print(f"Found existing results at {rlts_path}")
        return

    # read all the data first and initialize model
    speaker_audio_wm_dic = run_evaluate.get_speaker_audio_wm_dic()

    # initialize our watermark net
    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, exp_cfg.wm_length)

    # load our watermark net
    audio_sec_len = exp_cfg.sr  # by default, WatermarkNet split audio into 1-second sections
    wm_net = WatermarkNet(benign_encoded_wm, audio_sec_len, audio_sec_len,
                          wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir)
    model_dir = Path(exp_cfg.out_dir).joinpath(f"ExpEmbedWatermark")
    dic_saved = ModelTrainer.load_latest_ckpt(model_dir.joinpath("ckpt"))
    wm_net.load_state_dict(dic_saved["model_state"])
    wm_net = wm_net.to(utils.device)
    wm_net.eval()

    # Before running experiments, calculate the FLOPs
    alex_flops, alex_macs, alex_params = calculate_flops(model=models.alexnet(),
                                          input_shape=(1, 3, 224, 224),
                                          output_as_string=True,
                                          output_precision=4)
    print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" % (alex_flops, alex_macs, alex_params))
    # Alexnet FLOPs:4.2892 GFLOPS   MACs:2.1426 GMACs   Params:61.1008 M

    our_flops, our_macs, our_params = calculate_flops(model=wm_net.decoder,
                                                      input_shape=(1, 1, 29, 126),
                                                      output_as_string=True,
                                                      output_precision=4)
    print("Our Decoder FLOPs:%s   MACs:%s   Params:%s \n" % (our_flops, our_macs, our_params))


    total_run_seconds = 0
    total_audio_seconds = 0
    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=exp_cfg.wm_length)

    # calculate our metrics according to each speaker
    for speaker, audio_wm_lst in speaker_audio_wm_dic.items():
        print(f"eval_wm_speech {speaker} with {len(audio_wm_lst)} audios ...")

        long_our_wm_wav = []

        for audio_wm_dic in audio_wm_lst:
            our_wm_audio = audio_wm_dic["our_wm_audio"]
            long_our_wm_wav.append(our_wm_audio)

        long_our_wm_wav = np.concatenate(long_our_wm_wav)

        long_seconds = len(long_our_wm_wav) / exp_cfg.sr
        total_audio_seconds += long_seconds

        wm_pool, speaker_idx = run_evaluate.get_wm_pool(speaker, speakers_wm_lst)

        start_seconds = time.time()

        with torch.no_grad():
            wm_preds = wm_net.inference(long_our_wm_wav, wm_pool, benign_wm_included=False)

        seconds_passed = time.time() - start_seconds
        total_run_seconds += seconds_passed

    # save results
    with open(rlts_path, "w") as f:
        print(f"Experiment date = {datetime.now()}", file=f)
        print(f"Total run seconds = {total_run_seconds}", file=f)
        print(f"Total audio seconds = {total_audio_seconds}", file=f)

        print(f"\n--------------------------\n", file=f)
        print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" % (alex_flops, alex_macs, alex_params), file=f)
        print("Our Decoder FLOPs:%s   MACs:%s   Params:%s \n" % (our_flops, our_macs, our_params), file=f)

    return



if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    run_evaluate.exp_cfg = exp_cfg  # set to this model

    main()

    sys.exit()




