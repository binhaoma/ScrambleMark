"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import pickle
import sys

from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig

from evaluation import compute_eer
from tuneThreshold import tuneThresholdfromScore

from models.ModelTrainer import ModelTrainer
from models.WatermarkNet import WatermarkNet

from run_evaluate import get_wm_pool, positive_binary_detect_rate

import exp_setup
from my_utils import utils
import numpy as np
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


class MyLstData(torch.utils.data.Dataset):
    def __init__(self, np_data, np_labels, preprocess=None):
        super().__init__()

        self.lst_data = np_data
        self.lst_labels = np_labels

        self.preprocess = preprocess
        self.targets = torch.LongTensor(self.lst_labels)     # to be compatible with other datasets

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, idx):

        x, y = torch.FloatTensor(self.lst_data[idx]), self.lst_labels[idx]

        if self.preprocess is not None:
            x = self.preprocess(x)

        return x, y


def eval_aasist(exp_dir, args: argparse.Namespace):
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    rlt_dic_path = exp_dir.joinpath("aasist_rlt.bin")
    if rlt_dic_path.exists():
        with open(rlt_dic_path, 'rb') as handle:
            rlt_dic = pickle.load(handle)
        print(f"Found existing aasist results: {rlt_dic}")
        return

    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # define model architecture
    model = get_model(model_config)

    # define dataloaders
    eval_loader = get_aasist_loader()

    # evaluates pretrained model
    model.load_state_dict(
        torch.load(args.eval_model_weights, map_location=utils.device))
    print("Model loaded : {}".format(args.eval_model_weights))
    print("Start evaluation...")

    scores_arr, labels_arr = eval_model(eval_loader, model)
    bona_cm, spoof_cm = scores_arr[labels_arr==0], scores_arr[labels_arr==1]

    print(f"bona_cm = {np.mean(bona_cm):.3f} +- {np.std(bona_cm):.3f}")
    print(f"spoof_cm = {np.mean(spoof_cm):.3f} +- {np.std(spoof_cm):.3f}")

    eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    print(f"eer_cm results = {eer_cm*100:.2f}%")

    # the code comes from speaker verification, where label=1 means accept
    result = tuneThresholdfromScore(scores_arr, (1-labels_arr), [])
    err_again = result[1]
    print(f"err calculated again results = {err_again:.2f}%")

    rlt_dic = {
        "eer_cm": eer_cm,
        "bona_cm_mean": np.mean(bona_cm),
        "bona_cm_std": np.std(bona_cm),
        "spoof_cm_mean": np.mean(spoof_cm),
        "spoof_cm_std": np.std(spoof_cm),
    }

    with open(rlt_dic_path, 'wb') as handle:
        pickle.dump(rlt_dic, handle)

    return eer_cm

def eval_our_model(exp_dir):
    rlt_dic_path = exp_dir.joinpath("our_rlt.bin")
    if rlt_dic_path.exists():
        with open(rlt_dic_path, 'rb') as handle:
            rlt_dic = pickle.load(handle)
        print(f"Found existing our results: {rlt_dic}")
        return

    train_samples, speakers_wm_lst, speakers_names_lst, benign_encoded_wm = get_train_samples()

    fake_audios_lst_dic = {}

    org_audios_lst = None
    for iter_num in exp_cfg.speaker_adapt_iters:
        if iter_num == 1:
            org_audios_lst, fake_audios_lst = get_tgt_samples(
                fake_exp_dir=Path(exp_cfg.out_dir).joinpath("ExpSpeakerAdaptYourTTS"),
                tgt_names_lst=speakers_names_lst, iter_num=iter_num, train_samples=train_samples
            )
        else:
            _, fake_audios_lst = get_tgt_samples(
                fake_exp_dir=Path(exp_cfg.out_dir).joinpath("ExpSpeakerAdaptYourTTS"),
                tgt_names_lst=speakers_names_lst, iter_num=iter_num, train_samples=None)

        fake_audios_lst_dic[iter_num] = fake_audios_lst

    assert org_audios_lst is not None

    # load our watermark net
    audio_sec_len = exp_cfg.sr  # by default, WatermarkNet split audio into 1-second sections
    wm_net = WatermarkNet(benign_encoded_wm, audio_sec_len, audio_sec_len,
                          wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir)
    model_dir = Path(exp_cfg.out_dir).joinpath("ExpEmbedWatermark")
    dic_saved = ModelTrainer.load_latest_ckpt(model_dir.joinpath("ckpt"))
    wm_net.load_state_dict(dic_saved["model_state"])
    wm_net = wm_net.to(utils.device)
    wm_net.eval()

    # use our model to do inference on each benign and fake audio
    wm_pool, speaker_idx = get_wm_pool(speakers_names_lst[0], speakers_wm_lst)

    org_preds_lst = []
    wm_preds_lst_dic = {}

    with torch.no_grad():
        # size of org and wm lst may be different
        for org_wav in tqdm(org_audios_lst):
            org_preds = wm_net.inference(org_wav, wm_pool, benign_wm_included=False)
            org_preds_lst.append(org_preds)

        for iter_num in tqdm(exp_cfg.speaker_adapt_iters):
            wm_preds_lst = []
            fake_audio_lst = fake_audios_lst_dic[iter_num]
            for wm_wav in fake_audio_lst:
                wm_preds = wm_net.inference(wm_wav, wm_pool, benign_wm_included=False)
                wm_preds_lst.append(wm_preds)

            wm_preds_lst_dic[iter_num] = wm_preds_lst

    # calculate the accuracy rate
    org_bin_acc_lst = []
    for org_preds in org_preds_lst:
        if org_preds is None:
            continue        # the audio is too short to be predicted.
        org_preds = org_preds.detach().cpu().numpy()
        org_bin_acc_lst.append(positive_binary_detect_rate(wm_preds=org_preds, speaker_idx=speaker_idx, benign_idx=wm_pool.shape[0]))

    wm_bin_acc_lst_dic = {}
    for iter_num in exp_cfg.speaker_adapt_iters:
        wm_bin_acc_lst =[]
        wm_preds_lst = wm_preds_lst_dic[iter_num]
        for wm_preds in wm_preds_lst:
            if wm_preds is None:
                continue    # the audio is too short to be predicted.
            wm_preds = wm_preds.detach().cpu().numpy()
            wm_bin_acc_lst.append(positive_binary_detect_rate(wm_preds=wm_preds, speaker_idx=speaker_idx, benign_idx=wm_pool.shape[0]))

        wm_bin_acc_lst_dic[iter_num] = wm_bin_acc_lst

    # finally calculate the eer
    bin_eer_lst = []
    for iter_num in exp_cfg.speaker_adapt_iters:
        wm_bin_acc_lst = wm_bin_acc_lst_dic[iter_num]
        bin_eer = compute_eer(np.array(wm_bin_acc_lst), np.array(org_bin_acc_lst))[0]

        bin_eer_lst.append(bin_eer)


    assert len(bin_eer_lst) == len(exp_cfg.speaker_adapt_iters)

    rlt_dic = {
        "bin_eer_lst": bin_eer_lst,
    }

    with open(rlt_dic_path, 'wb') as handle:
        pickle.dump(rlt_dic, handle)


def plot_rlts(exp_dir):
    with open(exp_dir.joinpath("aasist_rlt.bin"), 'rb') as handle:
        aasist_rlt_dic = pickle.load(handle)

    with open(exp_dir.joinpath("our_rlt.bin"), 'rb') as handle:
        our_rlt_dic = pickle.load(handle)

    aasist_eer = aasist_rlt_dic["eer_cm"] * 100.0
    our_bin_eer_lst = np.array(our_rlt_dic["bin_eer_lst"]) * 100.0

    # plot overall figure
    sns.set_style("darkgrid")
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)

    plt.figure(figsize=(6, 6))
    plt.title(f"Compare our watermarking with AASIST", fontsize=18)

    plot_iters = np.array(exp_cfg.speaker_adapt_iters)
    plot_iters = plot_iters - 1  # offset one as our callback is on start.

    plt.plot(plot_iters, our_bin_eer_lst, "-bo", linewidth=2, label="Our watermarking", alpha=0.8)
    plt.xlabel(r"Iterations", fontsize=20)
    plt.ylim(-1, 101)

    plt.ylabel("EER (%)", fontsize=20)

    plt.plot(plot_iters, [aasist_eer] * len(plot_iters), "--r", linewidth=2, label="AASIST")

    plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(exp_dir.joinpath("compare_aasist.png"))
    plt.close()

def main(args: argparse.Namespace):

    exp_dir = Path(exp_cfg.out_dir).joinpath("ExpCompareAASIST")
    exp_dir.mkdir(exist_ok=True)

    eval_aasist(exp_dir, args)
    eval_our_model(exp_dir)

    plot_rlts(exp_dir)



def get_model(model_config: Dict):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(utils.device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_tgt_samples(fake_exp_dir, tgt_names_lst, iter_num, train_samples=None):
    org_audios_lst = []
    fake_audios_lst = []

    if train_samples is not None:
        for sample in tqdm(train_samples):
            speaker_name = sample["speaker_name"]

            if speaker_name in tgt_names_lst:
                org_audio_path = Path(sample["audio_file"])
                assert org_audio_path.suffix == ".flac"

                org_audio, sr = utils.read_audio(org_audio_path, expected_sr=16000)
                org_audios_lst.append(org_audio)

    for speaker_name in tgt_names_lst:
        fake_dir = fake_exp_dir.joinpath(f"adapt_to_{speaker_name}/iter_{iter_num:04d}")

        all_fake_path_lst = list(fake_dir.glob("*.wav"))
        assert len(all_fake_path_lst) == 100, "We have generated 100 fake audios for each speaker."

        for fake_path in all_fake_path_lst:
            fake_audio, sr = utils.read_audio(fake_path, expected_sr=16000)
            fake_audios_lst.append(fake_audio)

    print(f"get_tgt_samples read len(org_audios_lst) = {len(org_audios_lst)}; len(fake_audios_lst) = {len(fake_audios_lst)}")

    return org_audios_lst, fake_audios_lst


def pad(x, max_len=64600):
    x = x.numpy()

    x_len = x.shape[0]
    if x_len >= max_len:
        return torch.Tensor(x[:max_len])
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]

    padded_x = torch.Tensor(padded_x)
    return padded_x


def get_train_samples():
    # read original vctk audio files and YourTTS fake audio without speaker adaptation
    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, 16)
    speakers_names_lst = [x["speaker"] for x in speakers_wm_lst]

    # init configs
    vctk_config = BaseDatasetConfig(
        formatter="vctk",
        dataset_name="vctk",
        meta_file_train="",
        meta_file_val="",
        path=str(exp_setup.get_vctk_dir(exp_cfg)),
        language="en",
    )
    train_samples, eval_samples = load_tts_samples(vctk_config, eval_split=False)
    assert eval_samples is None

    return train_samples, speakers_wm_lst, speakers_names_lst, benign_encoded_wm

def get_aasist_loader():
    train_samples, speakers_wm_lst, speakers_names_lst, benign_encoded_wm = get_train_samples()

    org_audios_lst, fake_audios_lst = get_tgt_samples(
        fake_exp_dir=Path(exp_cfg.out_dir).joinpath("ExpSpeakerAdaptYourTTS"),
        tgt_names_lst=speakers_names_lst,
        iter_num=1,     # iter_num = 1 for fake speech without speaker adaptation
        train_samples=train_samples,
    )

    data_lst = []
    labels_lst = []

    data_lst.extend(org_audios_lst)
    labels_lst.extend([0] * len(org_audios_lst))        # real audio

    data_lst.extend(fake_audios_lst)
    labels_lst.extend([1] * len(fake_audios_lst))       # fake audios are labeled as 1

    eval_dset = MyLstData(np_data=data_lst, np_labels=labels_lst, preprocess=pad)

    eval_loader = DataLoader(eval_dset,
                             batch_size=32,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=0)

    return eval_loader


def eval_model(data_loader: DataLoader, model):
    """Perform evaluation"""
    model.eval()

    labels_list = []
    scores_list = []
    for batch_x, batch_y in tqdm(data_loader):
        batch_x = batch_x.to(utils.device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        labels_list.extend(batch_y.numpy().tolist())
        scores_list.extend(batch_score.tolist())

    assert len(labels_list) == len(scores_list)

    return np.array(scores_list), np.array(labels_list)


if __name__ == "__main__":
    exp_cfg = exp_setup.init_config()

    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        default="./config/AASIST.conf",)

    parser.add_argument("--eval_model_weights",
                        type=str,
                        default="./weights/AASIST.pth",
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())

    sys.exit()



