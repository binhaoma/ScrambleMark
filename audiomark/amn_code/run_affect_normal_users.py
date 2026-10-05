from __future__ import print_function

import pickle
import sys
from datetime import datetime

import exp_setup
from pathlib import Path

import run_evaluate
from exp.ExpAdaptiveIter import slice_wav
from my_utils import utils

import time
import torch
from torch.nn import functional as F
import numpy as np

from run_evaluate import get_speaker_audio_wm_dic
from tqdm import tqdm

from SpeakerNet import *
from run_trainSpeakerNet import WrappedModel, args

args.gpu = 0


def eval_speaker_verification(speaker_net, enrol_feat, verify_speech):
    assert speaker_net.training is False, "speaker_net must be in the eval mode."

    ref_feat = enrol_feat
    num_eval = 10

    verify_feat = slice_wav(verify_speech).to(utils.device)
    verify_feat = speaker_net(verify_feat)

    if speaker_net.module.__L__.test_normalize:
        ref_feat = F.normalize(ref_feat, p=2, dim=1)
        verify_feat = F.normalize(verify_feat, p=2, dim=1)

    dist = torch.cdist(ref_feat.reshape(num_eval, -1), verify_feat.reshape(num_eval, -1)).detach().cpu().numpy()

    score = -1 * np.mean(dist)

    return score


def eval_speaker(speaker_net, enrol_feat, audio_wm_lst):
    scores_lst = []

    for audio_wm_dic in tqdm(audio_wm_lst):
        # see whether our watermarked audio can still be recognized
        wm_audio = audio_wm_dic["our_wm_audio"]
        score = eval_speaker_verification(speaker_net, enrol_feat, wm_audio)

        scores_lst.append(score)

    return scores_lst


def save_rlt_to_txt(rlts_dic, txt_path):

    with open(txt_path, "w") as f:
        all_scores_lst = []

        for speaker, single_rlt_dic in rlts_dic.items():
            scores_lst = np.array(single_rlt_dic["scores_lst"])
            all_scores_lst.append(scores_lst)

            enrol_secs = single_rlt_dic["enrol_secs"]

            print(f"speaker: {speaker}:", file=f)
            print(f"enrol seconds: {enrol_secs:.1f}:", file=f)
            print(f"scores: {scores_lst.mean():.2f} +- {scores_lst.std():.2f}", file=f)

            passed_num = (scores_lst > -1.11).sum()
            print(f"passed number: {passed_num} / {len(scores_lst)} ({passed_num / len(scores_lst)*100:.2f} %)", file=f)

            print(f"\n---------------------------------------------------\n", file=f)

        # finally get the overall summary
        all_scores_lst = np.concatenate(all_scores_lst)
        total_num = len(all_scores_lst)
        all_passed_num = (all_scores_lst > -1.11).sum()
        print(f"Overall passed number: "
              f"{all_passed_num} / {total_num} ({all_passed_num / total_num * 100:.2f} %)\n\n", file=f)


def main():
    exp_dir = Path(exp_cfg.out_dir).joinpath(f"ExpAffectNormalUsers")
    exp_dir.mkdir(exist_ok=True)

    rlts_path = exp_dir.joinpath("rlts.bin")
    if rlts_path.exists():
        print(f"Found existing results at {rlts_path}")
        with open(rlts_path, 'rb') as handle:
            rlts_dic = pickle.load(handle)

        save_rlt_to_txt(rlts_dic, rlts_path.with_suffix(".txt"))

        return

    # Use the RawNet3 to evaluate how our watermarks affect the recognition of the speech.
    # load model for speaker verification
    speaker_net = SpeakerNet(**vars(args))
    speaker_net = WrappedModel(speaker_net).to(utils.device)

    # load model parameters
    speaker_trainer = ModelTrainer(speaker_net, **vars(args))
    speaker_trainer.loadParameters(args.initial_model)
    print("Model {} loaded!".format(args.initial_model))

    # set to eval
    speaker_net.eval()

    # read all the data
    speaker_audio_wm_dic = get_speaker_audio_wm_dic()

    rlts_dic = {}

    for speaker, audio_wm_lst in speaker_audio_wm_dic.items():
        print(f"Evaluating {speaker} ...")

        enrol_secs = 0
        enrol_speech = []

        # use the first 4 speech as enrolment
        for i in range(4):
            org_audio = audio_wm_lst[i]["org_audio"]
            enrol_secs += (len(org_audio) / exp_cfg.sr)
            enrol_speech.append(org_audio)
        enrol_speech = np.concatenate(enrol_speech)

        # calculate features of enrolment speech
        enrol_slice = slice_wav(enrol_speech).to(utils.device)
        enrol_feat = speaker_net(enrol_slice)

        scores_lst = eval_speaker(speaker_net, enrol_feat, audio_wm_lst)

        rlts_dic[speaker] = {
            "scores_lst": scores_lst, "enrol_secs": enrol_secs,
        }

    with open(rlts_path, 'wb') as handle:
        pickle.dump(rlts_dic, handle)

    save_rlt_to_txt(rlts_dic, rlts_path.with_suffix(".txt"))

    return



if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    run_evaluate.exp_cfg = exp_cfg
    main()

    sys.exit()




