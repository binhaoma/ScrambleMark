from __future__ import print_function

import logging
import pickle
import sys
import webrtcvad
import torch
import scipy
from pywt.tests.test_data import wavelab_data_file

import audiomark.amn_code.exp_setup

from pathlib import Path
from audiomark.amn_code.my_utils import utils
import numpy as np
from audiomark.amn_code.models.WatermarkNet import WatermarkNet
from audiomark.amn_code.models.ModelTrainer import ModelTrainer
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm

import scipy.io.wavfile as wav
import seaborn as sns
import matplotlib

import matplotlib.ticker as mticker
import noisereduce as nr

# from aasist.evaluation import compute_eer

from audiomark.amn_code.exp.ExpSpeakerAdaptYourTTS import ExpSpeakerAdaptYourTTS, ExpConfig as ExpSpeakerAdaptYourTTSCfg
from audiomark.amn_code.exp.ExpEmbedWatermark import ExpEmbedWatermark, ExpConfig as ExpEmbedWatermarkCfg
import scipy.io.wavfile as wav_op

vad = webrtcvad.Vad(0)  # 3-> most aggressive; 0 -> least aggressive

###################################################
# metrics

def undetected_fake_length(wm_preds, speaker_idx):
    detected = (wm_preds == speaker_idx)

    # calculate the length of undetected sections
    undetected_len_lst = []
    undetected_started = False    # has started undetected watermark
    undetected_len = 0
    for rlt in detected:
        rlt = bool(rlt)
        if undetected_started is False:
            # all detected till now
            if rlt is True:
                pass
            else:
                undetected_started = True
                assert undetected_len == 0
                undetected_len += 1

        else:
            # undetected started
            if rlt is True:
                # undetected finish
                undetected_started = False
                undetected_len_lst.append(undetected_len)
                undetected_len = 0
            else:
                undetected_len += 1

    # the last one
    if undetected_len > 0:
        assert bool(detected[-1]) is False
        undetected_len_lst.append(undetected_len)

    return undetected_len_lst


def binary_undetected_fake_length(wm_preds, speaker_idx, benign_idx):
    # combine all users into a single one
    # idx = number speakers means benign speech
    valid_users_mask = np.logical_and(wm_preds >= 0, wm_preds < benign_idx)

    wm_preds = np.copy(wm_preds)
    wm_preds[valid_users_mask] = speaker_idx

    return undetected_fake_length(wm_preds, speaker_idx)


def false_positive_rate(wm_preds, benign_idx):
    # unknown sections do not trigger detection
    unkown_mask = (wm_preds == -1)
    wm_preds = np.copy(wm_preds)
    wm_preds[unkown_mask] = benign_idx

    fp = (wm_preds != benign_idx).sum()
    num = len(wm_preds)
    return fp / num


def positive_detect_rate(wm_preds, speaker_idx):
    detected_num = (wm_preds == speaker_idx).sum()
    num = len(wm_preds)
    return detected_num / num

def positive_binary_detect_rate(wm_preds, speaker_idx, benign_idx):
    # combine all users into a single one
    # idx = number speakers means benign speech
    valid_users_mask = np.logical_and(wm_preds >= 0, wm_preds < benign_idx)

    wm_preds = np.copy(wm_preds)
    wm_preds[valid_users_mask] = speaker_idx

    return positive_detect_rate(wm_preds, speaker_idx)

###################################################
# plots

def plot_spectrogram(waveform, title, fig_path, clr_bar=True):
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)

    n_fft = 2048
    hop_length = n_fft // 4

    fig, ax = plt.subplots()
    D_highres = librosa.stft(waveform, hop_length=hop_length, n_fft=n_fft)
    S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
    img = librosa.display.specshow(S_db_hr, hop_length=hop_length, x_axis='time', y_axis='log',
                                   ax=ax)
    # ax.set(title=title, fontsize=20)
    if clr_bar:
        fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.xlabel(r"Time", fontsize=20)
    plt.ylabel("Hz", fontsize=20)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


####################################################


def get_speaker_audio_wm_dic(exp_cfg):

    our_saved_path = Path(exp_cfg.out_dir).joinpath(f"ExpEmbedWatermark/tgt_samples.bin")
    # wavmark_saved_path = Path(exp_cfg.out_dir).joinpath(f"ExpWavMark/tgt_samples.bin")

    ################################################
    # We do not evaluate wavmark anymore.
    # A quick fix for existing code here is to make load the same data from our experiments
    wavmark_saved_path = our_saved_path

    ################################################

    with open(our_saved_path, 'rb') as handle:
        our_tgt_samples = pickle.load(handle)
    with open(wavmark_saved_path, 'rb') as handle:
        wavmark_tgt_samples = pickle.load(handle)
    assert len(our_tgt_samples) == len(wavmark_tgt_samples)

    speaker_audio_wm_dic = {}

    org_audio_len_arr = []

    for our_sample, our_wm in tqdm(our_tgt_samples): 
        speaker_name = our_sample["speaker_name"] # VCTK_p261
        if speaker_name not in speaker_audio_wm_dic:
            speaker_audio_wm_dic[speaker_name] = []

        org_audio_path = Path(our_sample["audio_file"]) #/home/wzong/My Passport/projects/data/vctk/wav48_silence_trimmed/p261/p261_069_mic1.flac

        assert org_audio_path.suffix == ".flac", "original audio files are .flac"
        assert our_sample["audio_file_wm"].suffix == ".wav", "our watermarked files are .wav"

        our_wm_audio_path = our_sample["audio_file_wm"] #/home/wzong/My Passport/projects/FakeAudioDetection/out/ExpEmbedWatermark/VCTK_p261/p261_069_mic1.wav

        # find the corresponding wavmark audio
        wavmark_wm_audio_path = None
        wavmark_wm = None
        for wavmark_sample, wavmark_wm in wavmark_tgt_samples:
            if wavmark_sample["audio_file_wm"].name == our_wm_audio_path.name:
                assert wavmark_sample["audio_file"] == our_sample["audio_file"]
                wavmark_wm_audio_path = wavmark_sample["audio_file_wm"]
                break

        assert (wavmark_wm_audio_path is not None) and (wavmark_wm is not None)

        org_audio, _ = utils.read_audio(org_audio_path, exp_cfg.sr)
        our_wm_audio, _ = utils.read_audio(our_wm_audio_path, exp_cfg.sr)
        wavmark_wm_audio, _ = utils.read_audio(wavmark_wm_audio_path, exp_cfg.sr)

        org_audio_len_arr.append(len(org_audio))

        speaker_audio_wm_dic[speaker_name].append({ # VCTK_p261
            "org_audio": org_audio, # oriaudio
            "our_wm_audio": our_wm_audio, "our_wm": our_wm, #[1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1]
            "wavmark_wm_audio": wavmark_wm_audio, "wavmark_wm": wavmark_wm,
        })

    print(f"avg org audio len = {np.mean(org_audio_len_arr)/exp_cfg.sr:.2f}")

    return speaker_audio_wm_dic


def eval_wm_quality(exp_dir, speaker_audio_wm_dic):
    # directly evaluate our watermarked speech
    exp_dir.mkdir(exist_ok=True)

    flag_path = exp_dir.joinpath("eval_wm_quality.flag")
    if utils.get_flag(flag_path):
        return

    our_snr_arr = []
    our_pesq_arr = []

    # wavmark_snr_arr = []
    # wavmark_pesq_arr = []

    # calculate quality of speech
    for speaker, audio_wm_lst in speaker_audio_wm_dic.items():
        print(f"eval_wm_speech {speaker} ...")

        for audio_wm_dic in tqdm(audio_wm_lst):
            org_audio = audio_wm_dic["org_audio"]
            our_wm_audio = audio_wm_dic["our_wm_audio"]

            our_snr_arr.append(utils.cal_snr(org_audio, org_audio - our_wm_audio))
            our_pesq_arr.append(utils.cal_pesq(fs=exp_cfg.sr, ref=org_audio, deg=our_wm_audio))

            # wavmark_wm_audio = audio_wm_dic["wavmark_wm_audio"]
            # wavmark_snr_arr.append(utils.cal_snr(org_audio, org_audio - wavmark_wm_audio))
            # wavmark_pesq_arr.append(utils.cal_pesq(fs=exp_cfg.sr, ref=org_audio, deg=wavmark_wm_audio))

    our_snr_arr = np.array(our_snr_arr)
    our_pesq_arr = np.array(our_pesq_arr)

    # wavmark_snr_arr = np.array(wavmark_snr_arr)
    # wavmark_pesq_arr = np.array(wavmark_pesq_arr)

    with open(exp_dir.joinpath("audio_quality.txt"), "w") as f:
        print(f"our_snr = {np.mean(our_snr_arr):.2f} +- {np.std(our_snr_arr):.2f}", file=f)
        print(f"our_pesq = {np.mean(our_pesq_arr):.2f} +- {np.std(our_pesq_arr):.2f}", file=f)

        print("\n", file=f)

        # print(f"wavmark_snr = {np.mean(wavmark_snr_arr):.2f} +- {np.std(wavmark_snr_arr):.2f}", file=f)
        # print(f"wavmark_pesq = {np.mean(wavmark_pesq_arr):.2f} +- {np.std(wavmark_pesq_arr):.2f}", file=f)

    utils.set_flag(flag_path)


def plot_our_spectrograms(exp_dir, speaker_audio_wm_dic):
    exp_dir.mkdir(exist_ok=True)

    flag_path = exp_dir.joinpath("plot_our_spectrograms_done.flag")
    if utils.get_flag(flag_path):
        return

    for speaker, audio_wm_lst in speaker_audio_wm_dic.items():
        print(f"plotting for {speaker}...")

        for idx in [10, 50, 100, ]:
            audio_wm_dic = audio_wm_lst[idx]
            org_audio = audio_wm_dic["org_audio"]
            our_wm_audio = audio_wm_dic["our_wm_audio"]

            # save org, wm, and delta
            plot_spectrogram(org_audio, "Spectrogram of original speech",
                             exp_dir.joinpath(f"{speaker}_{idx}_org.png"))
            plot_spectrogram(our_wm_audio, "Spectrogram of watermarked speech",
                             exp_dir.joinpath(f"{speaker}_{idx}_wm.png"))

            delta = our_wm_audio - org_audio
            plot_spectrogram(delta, "Watermark added",
                             exp_dir.joinpath(f"{speaker}_{idx}_delta.png"))

    utils.set_flag(flag_path)

    return


def get_wm_pool(speaker, speakers_wm_lst):
    wm_pool = []
    speaker_idx = None
    for idx, data_dic in enumerate(speakers_wm_lst):
        speaker_name, wm = data_dic["speaker"], data_dic["org_wm"]
        wm_pool.append(wm)

        if speaker_name == speaker:
            assert speaker_idx is None
            speaker_idx = idx
    assert speaker_idx is not None
    wm_pool = torch.from_numpy(np.array(wm_pool)).to(utils.device)

    return wm_pool, speaker_idx


def cal_eer(speaker, wm_net, org_wav_lst, our_wm_wav_lst, speakers_wm_lst=None):
    if speakers_wm_lst is None:
        speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=exp_cfg.wm_length)

    wm_pool, speaker_idx = get_wm_pool(speaker, speakers_wm_lst)

    org_preds_lst = []
    wm_preds_lst = []

    with torch.no_grad():
        # size of org and wm lst may be different
        for org_wav in org_wav_lst:
            org_preds = wm_net.inference(org_wav, wm_pool, benign_wm_included=False)
            org_preds_lst.append(org_preds)

        for wm_wav in our_wm_wav_lst:
            wm_preds = wm_net.inference(wm_wav, wm_pool, benign_wm_included=False)
            wm_preds_lst.append(wm_preds)

    # calculate the accuracy rate
    org_acc_lst = []
    org_bin_acc_lst = []
    for org_preds in org_preds_lst:
        if org_preds is None:
            continue        # the audio is too short to be predicted.
        org_preds = org_preds.detach().cpu().numpy()
        org_acc_lst.append(positive_detect_rate(wm_preds=org_preds, speaker_idx=speaker_idx))
        org_bin_acc_lst.append(positive_binary_detect_rate(wm_preds=org_preds, speaker_idx=speaker_idx, benign_idx=wm_pool.shape[0]))

    wm_acc_lst = []
    wm_bin_acc_lst = []
    for wm_preds in wm_preds_lst:
        if wm_preds is None:
            continue    # the audio is too short to be predicted.
        wm_preds = wm_preds.detach().cpu().numpy()
        wm_acc_lst.append(positive_detect_rate(wm_preds=wm_preds, speaker_idx=speaker_idx))
        wm_bin_acc_lst.append(positive_binary_detect_rate(wm_preds=wm_preds, speaker_idx=speaker_idx, benign_idx=wm_pool.shape[0]))

    # finally calculate the eer
    eer = compute_eer(np.array(wm_acc_lst), np.array(org_acc_lst))[0]
    bin_eer = compute_eer(np.array(wm_bin_acc_lst), np.array(org_bin_acc_lst))[0]

    return eer, bin_eer


def eval_our_wm(speaker, wm_net, long_org_wav, long_wm_wav, speakers_wm_lst=None):
    if speakers_wm_lst is None:
        speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=exp_cfg.wm_length)

    wm_pool, speaker_idx = get_wm_pool(speaker, speakers_wm_lst)

    with torch.no_grad():
        org_preds = wm_net.inference(long_org_wav, wm_pool, benign_wm_included=False)
        wm_preds = wm_net.inference(long_wm_wav, wm_pool, benign_wm_included=False)

    assert org_preds is not None and wm_preds is not None

    org_preds = org_preds.squeeze().cpu().detach().numpy()
    wm_preds = wm_preds.squeeze().cpu().detach().numpy()

    ufl = undetected_fake_length(wm_preds, speaker_idx)
    bin_ufl = binary_undetected_fake_length(wm_preds, speaker_idx, benign_idx=wm_pool.shape[0])

    fpr = false_positive_rate(org_preds, benign_idx=wm_pool.shape[0])

    our_acc = positive_detect_rate(wm_preds=wm_preds, speaker_idx=speaker_idx)
    our_bin_acc = positive_binary_detect_rate(wm_preds=wm_preds, speaker_idx=speaker_idx, benign_idx=wm_pool.shape[0])

    return ufl, bin_ufl, fpr, our_acc, our_bin_acc


def do_wavmark_pred(wavmark_net, x, wm_pool, wm_net):
    assert wavmark_net.training is False
    wm_pool = wm_pool.cpu().detach().numpy()

    # first remove non-speech part
    x = utils.remove_non_speech(x, vad, exp_cfg.sr)
    if len(x) < exp_cfg.sr:
        # x is too short for a section
        return None

    # for each audio section, find the wm
    wav_secs = wm_net.split_waveform(x)
    wav_secs = wav_secs.cpu().detach().numpy()

    pred_wm_idx = []
    wav_len = 2     # we use 2 seconds for wavmark, which is larger than average audio lenght in VCTK testing speakers
    num_secs = len(wav_secs)
    for wav_idx in tqdm(range(num_secs)):

        if wav_idx + wav_len > len(wav_secs):
            break

        waveform = wav_secs[wav_idx: wav_idx + wav_len]
        waveform = np.concatenate(waveform)

        payload_decoded, _ = wavmark.decode_watermark(wavmark_net, waveform, show_progress=False)
        # find the corresponding wm
        pred_idx = None
        if payload_decoded is not None:
            assert isinstance(payload_decoded, np.ndarray) and (len(payload_decoded) == wm_pool.shape[1])
            for wm_idx, wm in enumerate(wm_pool):
                if (payload_decoded == wm).sum() == len(wm):
                    pred_idx = wm_idx
                    break

        if pred_idx is None:
            # did not find any wm, make this as benign
            pred_wm_idx.append(len(wm_pool))        # to be compatible with our watermark net
        else:
            pred_wm_idx.append(pred_idx)

        ###############################
        # if len(pred_wm_idx) == 20:
        #     break
        ###############################

    return np.array(pred_wm_idx)


def eval_wavmark_wm(speaker, wavmark_net, long_org_wav, long_wm_wav, wm_net):
    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=16)
    wm_pool, speaker_idx = get_wm_pool(speaker, speakers_wm_lst)

    org_preds = do_wavmark_pred(wavmark_net=wavmark_net, x=long_org_wav, wm_pool=wm_pool, wm_net=wm_net)
    wm_preds = do_wavmark_pred(wavmark_net=wavmark_net, x=long_wm_wav, wm_pool=wm_pool, wm_net=wm_net)

    assert org_preds is not None and wm_preds is not None

    ufl = undetected_fake_length(wm_preds, speaker_idx)
    bin_ufl = binary_undetected_fake_length(wm_preds, speaker_idx, benign_idx=wm_pool.shape[0])

    fpr = false_positive_rate(org_preds, benign_idx=wm_pool.shape[0])

    return ufl, bin_ufl, fpr


def save_metrics_txt(exp_dir, speaker_lst, our_ufl_arr, our_bin_ufl_arr, our_fpr_arr,
                     our_acc_arr, our_bin_acc_arr,
                     our_eer_arr, our_bin_eer_arr,
                     wavmark_ufl_arr, wavmark_bin_ufl_arr, wavmark_fpr_arr):

    with open(exp_dir.joinpath("eval_metrics.txt"), "w") as f:

        def _do_save_mean_std(_data_arr, _data_name):
            _data_arr_perc = np.array(_data_arr) * 100.0

            if len(_data_arr_perc) == 1:
                print(f"{_data_name} = {np.mean(_data_arr_perc):.2f}% (only 1 value)", file=f)
            else:
                print(f"{_data_name} = {np.mean(_data_arr_perc):.2f}% +- {np.std(_data_arr_perc):.2f}%", file=f)

        _do_save_mean_std(_data_arr=our_fpr_arr, _data_name="our_fpr")
        _do_save_mean_std(_data_arr=our_acc_arr, _data_name="our_acc")
        _do_save_mean_std(_data_arr=our_bin_acc_arr, _data_name="our_bin_acc")
        _do_save_mean_std(_data_arr=our_eer_arr, _data_name="our_eer")
        _do_save_mean_std(_data_arr=our_bin_eer_arr, _data_name="our_bin_eer")
        _do_save_mean_std(_data_arr=wavmark_fpr_arr, _data_name="wavmark_fpr")

        print("\n", file=f)

        def save_ufl(_ufl_arr, _title):
            all_ufl = []
            for speaker, ufl in zip(speaker_lst, _ufl_arr):
                if len(ufl) == 0:
                    mean_val, std_val = 0, 0
                else:
                    ufl = np.array(ufl)
                    mean_val, std_val = np.mean(ufl), np.std(ufl)

                print(f"{_title} -- {speaker}: {mean_val:.1f} +- {std_val:.1f}", file=f)

                all_ufl.extend(ufl)

            print("overall:", file=f)
            if len(all_ufl) == 0:
                all_mean_val, all_std_val = 0, 0
            else:
                all_ufl = np.array(all_ufl)
                all_mean_val, all_std_val = np.mean(all_ufl), np.std(all_ufl)
            print(f"{_title}: {all_mean_val:.1f} +- {all_std_val:.1f}", file=f)

            print("\n", file=f)

        save_ufl(our_ufl_arr, "our_ufl_arr")
        save_ufl(our_bin_ufl_arr, "our_bin_ufl_arr")

        save_ufl(wavmark_ufl_arr, "wavmark_ufl_arr")
        save_ufl(wavmark_bin_ufl_arr, "wavmark_bin_ufl_arr")


def eval_metrics(exp_dir, speaker_audio_wm_dic, wm_net, wavmark_net, attack=None, speakers_wm_lst=None):
    # directly evaluate our watermarked speech
    exp_dir.mkdir(exist_ok=True)

    flag_path = exp_dir.joinpath("eval_metrics.flag")
    if utils.get_flag(flag_path):

        with open(exp_dir.joinpath("speaker_lst.bin"), 'rb') as handle:
            speaker_lst = pickle.load(handle)

        with open(exp_dir.joinpath("our_ufl_arr.bin"), 'rb') as handle:
            our_ufl_arr = pickle.load(handle)
        with open(exp_dir.joinpath("our_bin_ufl_arr.bin"), 'rb') as handle:
            our_bin_ufl_arr = pickle.load(handle)
        with open(exp_dir.joinpath("our_fpr_arr.bin"), 'rb') as handle:
            our_fpr_arr = pickle.load(handle)

        with open(exp_dir.joinpath("our_eer_arr.bin"), 'rb') as handle:
            our_eer_arr = pickle.load(handle)
        with open(exp_dir.joinpath("our_bin_eer_arr.bin"), 'rb') as handle:
            our_bin_eer_arr = pickle.load(handle)

        with open(exp_dir.joinpath("our_acc_arr.bin"), 'rb') as handle:
            our_acc_arr = pickle.load(handle)
        with open(exp_dir.joinpath("our_bin_acc_arr.bin"), 'rb') as handle:
            our_bin_acc_arr = pickle.load(handle)

        with open(exp_dir.joinpath("wavmark_ufl_arr.bin"), 'rb') as handle:
            wavmark_ufl_arr = pickle.load(handle)
        with open(exp_dir.joinpath("wavmark_bin_ufl_arr.bin"), 'rb') as handle:
            wavmark_bin_ufl_arr = pickle.load(handle)
        with open(exp_dir.joinpath("wavmark_fpr_arr.bin"), 'rb') as handle:
            wavmark_fpr_arr = pickle.load(handle)

        save_metrics_txt(exp_dir=exp_dir, speaker_lst=speaker_lst,
                         our_ufl_arr=our_ufl_arr, our_bin_ufl_arr=our_bin_ufl_arr, our_fpr_arr=our_fpr_arr,
                         our_acc_arr=our_acc_arr, our_bin_acc_arr=our_bin_acc_arr,
                         our_eer_arr=our_eer_arr, our_bin_eer_arr=our_bin_eer_arr,
                         wavmark_ufl_arr=wavmark_ufl_arr, wavmark_bin_ufl_arr=wavmark_bin_ufl_arr, wavmark_fpr_arr=wavmark_fpr_arr
                         )
        return our_ufl_arr, our_bin_ufl_arr, our_fpr_arr, our_acc_arr, our_bin_acc_arr, our_eer_arr, our_bin_eer_arr

    our_ufl_arr = []
    our_bin_ufl_arr = []
    our_fpr_arr = []

    our_eer_arr = []        # If only one sentence is provided, we split it into 4-second sections, e.g., PlayHT.
    our_bin_eer_arr = []

    our_acc_arr = []        # The detection accuracy of watermarked data based on 1-second sections
    our_bin_acc_arr = []    # The binary detection accuracy of watermarked data based on 1-second sections

    wavmark_ufl_arr = []
    wavmark_bin_ufl_arr = []
    wavmark_fpr_arr = []

    speaker_lst = []

    # calculate our metrics according to each speaker
    for speaker, audio_wm_lst in speaker_audio_wm_dic.items():
        print(f"eval_wm_speech {speaker} with {len(audio_wm_lst)} audios ...")
        speaker_lst.append(speaker)

        long_org_wav = []
        long_our_wm_wav = []
        long_wavmark_wav = []

        for audio_wm_dic in audio_wm_lst:
            org_audio = audio_wm_dic["org_audio"]
            our_wm_audio = audio_wm_dic["our_wm_audio"]

            long_org_wav.append(org_audio)
            long_our_wm_wav.append(our_wm_audio)

            if wavmark_net is not None:
                wavmark_wm_audio = audio_wm_dic["wavmark_wm_audio"]
                long_wavmark_wav.append(wavmark_wm_audio)

        org_wav_lst = long_org_wav           # store original sentences individually
        our_wm_wav_lst = long_our_wm_wav     # store our watermarked sentences individually

        long_org_wav = np.concatenate(long_org_wav)
        long_our_wm_wav = np.concatenate(long_our_wm_wav)

        if wavmark_net is not None:
            long_wavmark_wav = np.concatenate(long_wavmark_wav)

        if attack is not None:
            logging.info(f"applying attack {attack.__name__}")
            long_org_wav = attack(long_org_wav)
            long_our_wm_wav = attack(long_our_wm_wav)

            # apply attack to each sentence
            for i in range(len(org_wav_lst)):
                org_wav_lst[i] = attack(org_wav_lst[i])
            for i in range(len(our_wm_wav_lst)):
                our_wm_wav_lst[i] = attack(our_wm_wav_lst[i])

            if wavmark_net is not None:
                long_wavmark_wav = attack(long_wavmark_wav)

        # assert long_org_wav.shape == long_our_wm_wav.shape and long_org_wav.shape == long_wavmark_wav.shape

        our_ufl, our_bin_ufl, our_fpr, our_acc, our_bin_acc = eval_our_wm(
            speaker=speaker, wm_net=wm_net,
            long_org_wav=long_org_wav, long_wm_wav=long_our_wm_wav, speakers_wm_lst=speakers_wm_lst)
        our_ufl_arr.append(our_ufl)
        our_bin_ufl_arr.append(our_bin_ufl)
        our_fpr_arr.append(our_fpr)

        our_acc_arr.append(our_acc)
        our_bin_acc_arr.append(our_bin_acc)

        # we may need to evaluate WavMark if wavmark_net is set
        if wavmark_net is not None:
            wavmark_ufl, wavmark_bin_ufl, wavmark_fpr = eval_wavmark_wm(
                speaker=speaker, wavmark_net=wavmark_net,
                long_org_wav=long_org_wav, long_wm_wav=long_wavmark_wav, wm_net=wm_net)
            wavmark_ufl_arr.append(wavmark_ufl)
            wavmark_bin_ufl_arr.append(wavmark_bin_ufl)
            wavmark_fpr_arr.append(wavmark_fpr)
        else:
            # just store some values
            wavmark_ufl_arr.append([-1.0])
            wavmark_bin_ufl_arr.append([-1.0])
            wavmark_fpr_arr.append(-1.0)

        # calculate eer only if there are multiple sentences
        if (len(org_wav_lst) > 1) and (len(our_wm_wav_lst) > 1):
            eer, bin_eer = cal_eer(
                speaker=speaker, wm_net=wm_net,
                org_wav_lst=org_wav_lst, our_wm_wav_lst=our_wm_wav_lst, speakers_wm_lst=speakers_wm_lst
            )
            our_eer_arr.append(eer)
            our_bin_eer_arr.append(bin_eer)

    assert len(our_eer_arr) == len(our_bin_eer_arr)
    assert (len(our_eer_arr) == 0) or (len(our_eer_arr) == len(speaker_audio_wm_dic))

    if len(our_eer_arr) == 0:
        assert len(our_bin_eer_arr) == 0
        # This is for commercial tts where we generate 1 long sentence.
        assert len(our_acc_arr) == len(our_fpr_arr) == len(speaker_audio_wm_dic)
        eer = compute_eer(np.array(our_acc_arr), np.array(our_fpr_arr))[0]
        our_eer_arr.append(eer)

        assert len(our_bin_acc_arr) == len(our_fpr_arr) == len(speaker_audio_wm_dic)
        bin_eer = compute_eer(np.array(our_bin_acc_arr), np.array(our_fpr_arr))[0]
        our_bin_eer_arr.append(bin_eer)

    # save results
    with open(exp_dir.joinpath("speaker_lst.bin"), 'wb') as handle:
        pickle.dump(speaker_lst, handle)

    with open(exp_dir.joinpath("our_ufl_arr.bin"), 'wb') as handle:
        pickle.dump(our_ufl_arr, handle)
    with open(exp_dir.joinpath("our_bin_ufl_arr.bin"), 'wb') as handle:
        pickle.dump(our_bin_ufl_arr, handle)

    with open(exp_dir.joinpath("our_fpr_arr.bin"), 'wb') as handle:
        pickle.dump(our_fpr_arr, handle)

    with open(exp_dir.joinpath("our_eer_arr.bin"), 'wb') as handle:
        pickle.dump(our_eer_arr, handle)
    with open(exp_dir.joinpath("our_bin_eer_arr.bin"), 'wb') as handle:
        pickle.dump(our_bin_eer_arr, handle)

    with open(exp_dir.joinpath("our_acc_arr.bin"), 'wb') as handle:
        pickle.dump(our_acc_arr, handle)
    with open(exp_dir.joinpath("our_bin_acc_arr.bin"), 'wb') as handle:
        pickle.dump(our_bin_acc_arr, handle)

    with open(exp_dir.joinpath("wavmark_ufl_arr.bin"), 'wb') as handle:
        pickle.dump(wavmark_ufl_arr, handle)
    with open(exp_dir.joinpath("wavmark_bin_ufl_arr.bin"), 'wb') as handle:
        pickle.dump(wavmark_bin_ufl_arr, handle)

    with open(exp_dir.joinpath("wavmark_fpr_arr.bin"), 'wb') as handle:
        pickle.dump(wavmark_fpr_arr, handle)

    save_metrics_txt(exp_dir=exp_dir, speaker_lst=speaker_lst,
                     our_ufl_arr=our_ufl_arr, our_bin_ufl_arr=our_bin_ufl_arr, our_fpr_arr=our_fpr_arr,
                     our_acc_arr=our_acc_arr, our_bin_acc_arr=our_bin_acc_arr,
                     our_eer_arr=our_eer_arr, our_bin_eer_arr=our_bin_eer_arr,
                     wavmark_ufl_arr=wavmark_ufl_arr, wavmark_bin_ufl_arr=wavmark_bin_ufl_arr,
                     wavmark_fpr_arr=wavmark_fpr_arr
                     )

    utils.set_flag(flag_path)

    return our_ufl_arr, our_bin_ufl_arr, our_fpr_arr, our_acc_arr, our_bin_acc_arr, our_eer_arr, our_bin_eer_arr

###############################################################################
# attacks

def attack_combo_noise_and_noise_reduction(waveform):
    rand_noise = np.random.normal(0, 0.02, size=waveform.shape).astype(np.float32)
    waveform = np.clip(waveform + rand_noise, a_min=-1.0, a_max=1.0)

    waveform = attack_noise_reduction(waveform)

    return waveform

def attack_combo_noise_and_resample(waveform):
    org_wav = waveform

    rand_noise = np.random.normal(0, 0.002, size=waveform.shape).astype(np.float32)
    waveform = waveform + rand_noise

    waveform = utils.resample_wav(waveform, 16000, 8000)
    waveform = utils.resample_wav(waveform, 8000, 16000)

    waveform = np.clip(waveform, a_min=-1.0, a_max=1.0)

    # make the length same
    tmp_wav = np.zeros_like(org_wav)
    tgt_len = min(org_wav.size, waveform.size)
    tmp_wav[:tgt_len] = waveform[:tgt_len]

    waveform = tmp_wav

    return waveform

def attack_add_noise(waveform):
    rand_noise = np.random.normal(0, 0.005, size=waveform.shape).astype(np.float32)
    waveform = np.clip(waveform + rand_noise, a_min=-1.0, a_max=1.0)
    return waveform


def attack_mp3(waveform):
    # save this to a tmp file
    tmp_wav_path = tmp_dir.joinpath(f"tmp_for_mp3.wav")
    wav.write(tmp_wav_path, exp_cfg.sr, waveform)
    other_codec_path = utils.wav_to_other_codec(tmp_wav_path, format_name="mp3", bitrate="96k")

    waveform, sr = utils.read_audio(other_codec_path)
    assert sr == exp_cfg.sr
    return waveform


def attack_opus(waveform):
    # save this to a tmp file
    tmp_wav_path = tmp_dir.joinpath(f"tmp_for_opus.wav")
    wav.write(tmp_wav_path, exp_cfg.sr, waveform)
    other_codec_path = utils.wav_to_other_codec(tmp_wav_path, format_name="opus", bitrate="24k")

    waveform, sr = utils.read_audio(other_codec_path)
    assert sr == exp_cfg.sr
    return waveform


def attack_resample(waveform):
    waveform = utils.resample_wav(waveform, 16000, 8000)
    waveform = utils.resample_wav(waveform, 8000, 16000)

    waveform = np.clip(waveform, a_min=-1.0, a_max=1.0)

    return waveform


def attack_high_pass(waveform):
    ir = utils.get_ir(200, 16000, type="highpass")
    waveform = np.convolve(waveform, ir, "same")

    waveform = np.clip(waveform, a_min=-1.0, a_max=1.0)

    return waveform


def attack_low_pass(waveform):
    ir = utils.get_ir(3700, 16000, type="lowpass")
    waveform = np.convolve(waveform, ir, "same")

    waveform = np.clip(waveform, a_min=-1.0, a_max=1.0)

    return waveform


def attack_supress(waveform):
    supres_mask = np.random.uniform(size=waveform.shape) < 0.97
    waveform = waveform * supres_mask

    return waveform


def attack_median_filter(waveform):
    waveform = scipy.ndimage.median_filter(waveform, size=5)

    waveform = np.clip(waveform, a_min=-1.0, a_max=1.0)

    return waveform


def attack_scale(waveform):
    waveform = waveform * 0.7
    return waveform


def attack_quantization(waveform):
    waveform = waveform * (2**15) / (2**8)
    waveform = waveform.astype(np.int16)
    waveform = waveform * (2**8)
    waveform = waveform / (2**15)

    return waveform.astype(np.float32)


def attack_add_echo(waveform):
    echo = waveform * 0.3
    echo = np.pad(echo, (1600, 0))[: len(waveform)]
    waveform = waveform + echo

    waveform = np.clip(waveform, a_min=-1.0, a_max=1.0)

    return waveform


def attack_noise_reduction(waveform):
    waveform = nr.reduce_noise(y=waveform, sr=16000, prop_decrease=0.8)
    waveform = np.clip(waveform, a_min=-1.0, a_max=1.0)

    return waveform


def rand_filter_out_band(waveform, start_hz, end_hz, filter_out_hz):
    # adaptive attacks - randomly surpress band
    rand_pos = np.random.randint(low=start_hz, high=end_hz - filter_out_hz + 1)

    ir = utils.get_ir(rand_pos + filter_out_hz, exp_cfg.sr, type="highpass")
    hp_wav = np.convolve(waveform, ir, "same")
    ir = utils.get_ir(rand_pos, exp_cfg.sr, type="lowpass")
    lp_wav = np.convolve(waveform, ir, "same")
    #
    waveform = hp_wav + lp_wav
    waveform = np.clip(waveform, a_min=-1.0, a_max=1.0)
    return waveform


def attack_adaptive_50Hz(waveform):
    return rand_filter_out_band(waveform, 100, 1000, 50)


def attack_adaptive_100Hz(waveform):
    return rand_filter_out_band(waveform, 100, 1000, 100)


def attack_adaptive_200Hz(waveform):
    return rand_filter_out_band(waveform, 100, 1000, 200)


def attack_adaptive_400Hz(waveform):
    return rand_filter_out_band(waveform, 100, 1000, 400)


def load_all_local_tts_fake_speech(tts_exp_name):
    our_speakers_wm_lst, our_benign_org_wm, our_benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=exp_cfg.wm_length)
    # wavmark_speakers_wm_lst, wavmark_benign_org_wm, wavmark_benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=16)
    # assert len(our_speakers_wm_lst) == len(wavmark_speakers_wm_lst)

    exp_dir = Path(exp_cfg.out_dir).joinpath(tts_exp_name)

    speaker_lst = [x["speaker"] for x in our_speakers_wm_lst]
    iters_speaker_audio_wm_dic = {}

    for cur_iter in tqdm(exp_cfg.speaker_adapt_iters):
        speaker_audio_wm_dic = iters_speaker_audio_wm_dic[cur_iter] = {}
        for speaker_name in speaker_lst:
            # get all the fake speech for this speaker
            speaker_audio_wm_dic[speaker_name] = []

            # original speech always refer to iteration 1, since iterations starts from 1 and callback is on_start_xxx
            org_dir = exp_dir.joinpath(f"adapt_to_{speaker_name}/iter_0001")

            # dir to fake speech
            our_fake_dir = exp_dir.joinpath(f"adapt_to_{speaker_name}/iter_{cur_iter:04d}")
            # wavmark_fake_dir = exp_dir.joinpath(f"ExpWavMark_epochs_{cur_iter}/adapt_to_{speaker_name}/fake_speech")

            all_our_fake_path_lst = list(our_fake_dir.glob("*.wav"))
            all_our_fake_path_lst.sort()
            assert len(all_our_fake_path_lst) == 100, "there should be 100 fake speech"

            for our_fake_path in all_our_fake_path_lst:

                # load audios
                audio_name = our_fake_path.name
                org_audio_path = org_dir.joinpath(audio_name)
                org_audio, _ = utils.read_audio(org_audio_path, exp_cfg.sr)

                our_wm_audio_path = our_fake_dir.joinpath(audio_name)
                our_wm_audio, _ = utils.read_audio(our_wm_audio_path, exp_cfg.sr)

                # wavmark_wm_audio_path = wavmark_fake_dir.joinpath(audio_name)
                # wavmark_wm_audio, _ = utils.read_audio(wavmark_wm_audio_path, exp_cfg.sr)

                # find corresponding wm
                our_wm = None
                for our_data_dic in our_speakers_wm_lst:
                    if our_data_dic["speaker"] == speaker_name:
                        our_wm = our_data_dic["encoded_wm"]
                        break
                assert len(our_wm) == exp_cfg.wm_length

                # wavmark_wm = None
                # for wavmark_data_dic in wavmark_speakers_wm_lst:
                #     if wavmark_data_dic["speaker"] == speaker_name:
                #         wavmark_wm = wavmark_data_dic["encoded_wm"]
                #         break
                # assert len(wavmark_wm) == 16

                speaker_audio_wm_dic[speaker_name].append({
                    "org_audio": org_audio,
                    "our_wm_audio": our_wm_audio, "our_wm": our_wm,
                    # "wavmark_wm_audio": wavmark_wm_audio, "wavmark_wm": wavmark_wm,
                })

    return iters_speaker_audio_wm_dic


def read_audio_may_resample(_audio_path):
    _audio, _sr = utils.read_audio(_audio_path, None)
    if _sr != exp_cfg.sr:
        logging.info(f"Resampling {_audio_path.name} from {_sr} to {exp_cfg.sr}.")
        _audio = utils.resample_wav(_audio, _sr, exp_cfg.sr)
    return _audio


def load_all_commercial_fake_speech(commercial_name):
    our_speakers_wm_lst, our_benign_org_wm, our_benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=exp_cfg.wm_length)
    # wavmark_speakers_wm_lst, wavmark_benign_org_wm, wavmark_benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=16)
    # assert len(our_speakers_wm_lst) == len(wavmark_speakers_wm_lst)

    exp_dir = Path(exp_cfg.out_dir).joinpath("ExpCommercialGenerated")

    speaker_lst = [x["speaker"] for x in our_speakers_wm_lst]

    speaker_audio_wm_dic = {}
    for speaker_name in speaker_lst:
        # get all the fake speech for this speaker
        speaker_audio_wm_dic[speaker_name] = []

        # original speech are voice cloned without watermarks
        org_audio = read_audio_may_resample(exp_dir.joinpath(f"{speaker_name}_None_{commercial_name.lower()}.wav"))
        our_wm_audio = read_audio_may_resample(exp_dir.joinpath(f"{speaker_name}_wm_{commercial_name.lower()}.wav"))

        ###############################
        # # do not evaluate on wavmark anymore
        # wavmark_wm_audio_path = exp_dir.joinpath(f"{speaker_name}_combined_wm_ExpWavMark_PlayHT.wav")
        # wavmark_wm_audio, _ = utils.read_audio(wavmark_wm_audio_path, PlayHT_sr)
        # wavmark_wm_audio = utils.resample_wav(wavmark_wm_audio, PlayHT_sr, exp_cfg.sr)
        wavmark_wm_audio = our_wm_audio
        ###############################

        # find corresponding wm
        our_wm = None
        for our_data_dic in our_speakers_wm_lst:
            if our_data_dic["speaker"] == speaker_name:
                our_wm = our_data_dic["encoded_wm"]
                break
        assert len(our_wm) == exp_cfg.wm_length

        wavmark_wm = None
        # for wavmark_data_dic in wavmark_speakers_wm_lst:
        #     if wavmark_data_dic["speaker"] == speaker_name:
        #         wavmark_wm = wavmark_data_dic["encoded_wm"]
        #         break
        # assert len(wavmark_wm) == 16

        speaker_audio_wm_dic[speaker_name].append({
            "org_audio": org_audio,
            "our_wm_audio": our_wm_audio, "our_wm": our_wm,
            "wavmark_wm_audio": wavmark_wm_audio, "wavmark_wm": wavmark_wm,
        })

    return speaker_audio_wm_dic


def plot_local_tts_fake_speech(exp_dir, tts_name, iters_speaker_audio_wm_dic):
    exp_dir.mkdir(exist_ok=True)

    def cal_ufl(_ufl_arr, mean_arr, std_arr):
        all_ufl = []
        for ufl in _ufl_arr:
            all_ufl.extend(ufl)

        if len(all_ufl) == 0:
            all_mean_val, all_std_val = 0, 0
        else:
            all_ufl = np.array(all_ufl)
            all_mean_val, all_std_val = np.mean(all_ufl), np.std(all_ufl)

        mean_arr.append(all_mean_val)
        std_arr.append(all_std_val)

    # plot the ufl and bufl along epochs
    our_ufl_mean_arr = []
    our_ufl_std_arr = []
    our_bin_ufl_mean_arr = []
    our_bin_ufl_std_arr = []
    all_speakers_ufl_lst_lst = []

    our_acc_mean_arr = []
    our_acc_std_arr = []
    our_bin_acc_mean_arr = []
    our_bin_acc_std_arr = []
    all_speakers_acc_lst_lst = []

    our_eer_mean_arr = []
    our_eer_std_arr = []
    our_bin_eer_mean_arr = []
    our_bin_eer_std_arr = []
    all_speakers_eer_lst_lst = []

    speaker_lst = None

    for cur_iter in exp_cfg.speaker_adapt_iters:
        eval_dir = exp_dir.parent.joinpath(f"eval_{tts_name}_fake_speech_iter_{cur_iter:04d}")

        with open(eval_dir.joinpath("speaker_lst.bin"), 'rb') as handle:
            speaker_saved = pickle.load(handle)
            if speaker_lst is None:
                speaker_lst = speaker_saved
            else:
                for s1, s2 in zip(speaker_lst, speaker_saved):
                    assert s1 == s2, "inconsistent speaker list???"

        with open(eval_dir.joinpath("our_ufl_arr.bin"), 'rb') as handle:
            our_ufl_arr = pickle.load(handle)
            cal_ufl(our_ufl_arr, our_ufl_mean_arr, our_ufl_std_arr)

            # save mean for each speaker
            iter_mean_ufl_lst = []
            for ufl_lst in our_ufl_arr:
                if len(ufl_lst) > 0:
                    mean_val = np.mean(ufl_lst)
                else:
                    mean_val = 0
                iter_mean_ufl_lst.append(mean_val)
            all_speakers_ufl_lst_lst.append(iter_mean_ufl_lst)

        with open(eval_dir.joinpath("our_bin_ufl_arr.bin"), 'rb') as handle:
            our_bin_ufl_arr = pickle.load(handle)
            cal_ufl(our_bin_ufl_arr, our_bin_ufl_mean_arr, our_bin_ufl_std_arr)

        with open(eval_dir.joinpath("our_acc_arr.bin"), 'rb') as handle:
            our_acc_arr = pickle.load(handle)
            our_acc_mean_arr.append(np.mean(our_acc_arr))
            our_acc_std_arr.append(np.std(our_acc_arr))

            # save acc for each speaker
            all_speakers_acc_lst_lst.append(our_acc_arr)

        with open(eval_dir.joinpath("our_bin_acc_arr.bin"), 'rb') as handle:
            our_bin_acc_arr = pickle.load(handle)
            our_bin_acc_mean_arr.append(np.mean(our_bin_acc_arr))
            our_bin_acc_std_arr.append(np.std(our_bin_acc_arr))

        with open(eval_dir.joinpath("our_eer_arr.bin"), 'rb') as handle:
            our_eer_arr = pickle.load(handle)
            our_eer_mean_arr.append(np.mean(our_eer_arr))
            our_eer_std_arr.append(np.std(our_eer_arr))

            # save eer for each speaker
            all_speakers_eer_lst_lst.append(our_eer_arr)

        with open(eval_dir.joinpath("our_bin_eer_arr.bin"), 'rb') as handle:
            our_bin_eer_arr = pickle.load(handle)
            our_bin_eer_mean_arr.append(np.mean(our_bin_eer_arr))
            our_bin_eer_std_arr.append(np.std(our_bin_eer_arr))

    our_ufl_mean_arr = np.array(our_ufl_mean_arr)
    our_ufl_std_arr = np.array(our_ufl_std_arr)

    our_bin_ufl_mean_arr = np.array(our_bin_ufl_mean_arr)
    our_bin_ufl_std_arr = np.array(our_bin_ufl_std_arr)

    our_acc_mean_arr = np.array(our_acc_mean_arr) * 100.0
    our_acc_std_arr = np.array(our_acc_std_arr) * 100.0
    our_bin_acc_mean_arr = np.array(our_bin_acc_mean_arr) * 100.0
    our_bin_acc_std_arr = np.array(our_bin_acc_std_arr) * 100.0
    all_speakers_acc_lst_lst = np.array(all_speakers_acc_lst_lst) * 100.0

    our_eer_mean_arr = np.array(our_eer_mean_arr) * 100.0
    our_eer_std_arr = np.array(our_eer_std_arr) * 100.0
    our_bin_eer_mean_arr = np.array(our_bin_eer_mean_arr) * 100.0
    our_bin_eer_std_arr = np.array(our_bin_eer_std_arr) * 100.0
    all_speakers_eer_lst_lst = np.array(all_speakers_eer_lst_lst) * 100.0

    # --------------------------------------------
    # plot overall figure
    sns.set_style("darkgrid")
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    fig, ax1 = plt.subplots(figsize=(6, 6))
    plt.title(f"Detecting fake speech by {tts_name}", fontsize=17)

    iter_has_run_arr = np.array(exp_cfg.speaker_adapt_iters) - 1    # our callback is on_start_iter

    ax2 = plt.twinx()
    ax2.plot(iter_has_run_arr, our_acc_mean_arr, "-ro", linewidth=2, label="TPR", alpha=0.8)
    ax2.fill_between(iter_has_run_arr, our_acc_mean_arr - our_acc_std_arr, our_acc_mean_arr + our_acc_std_arr,
                     color="red", alpha=0.3)
    ax2.set_ylabel("TPR (%)", color='r', fontsize=17)
    ax2.set_ylim(-1, 100)
    ax2.grid(False)

    ax1.plot(iter_has_run_arr, our_ufl_mean_arr, "-bo", linewidth=2, label="UFL", alpha=0.8)
    ax1.fill_between(iter_has_run_arr, our_ufl_mean_arr - our_ufl_std_arr, our_ufl_mean_arr + our_ufl_std_arr, color="blue", alpha=0.3)
    ax1.set_ylabel("UFL", fontsize=17, color="b")
    ax1.set_ylim(-1, 300)

    plt.xlabel(r"Iterations", fontsize=17)

    # plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(exp_dir.joinpath(f"detect_fake_speech_ufl_{tts_name}.png"))
    plt.close()

    # --------------------------------------------
    # plot BUFL
    fig, ax1 = plt.subplots(figsize=(6, 6))
    plt.title(f"Detecting fake speech by {tts_name}", fontsize=17)

    ax2 = plt.twinx()
    ax2.plot(iter_has_run_arr, our_bin_acc_mean_arr, "-ro", linewidth=2, label="BTPR", alpha=0.8)
    ax2.fill_between(iter_has_run_arr, our_bin_acc_mean_arr - our_bin_acc_std_arr,
                     our_bin_acc_mean_arr + our_bin_acc_std_arr,
                     color="red", alpha=0.3)
    ax2.set_ylabel("BTPR (%)", color='r', fontsize=17)
    ax2.set_ylim(-1, 100)
    ax2.grid(False)

    ax1.set_ylim(-1, 300)
    ax1.plot(iter_has_run_arr, our_bin_ufl_mean_arr, "-bo", linewidth=2, label="BUFL", alpha=0.8)
    ax1.fill_between(iter_has_run_arr, our_bin_ufl_mean_arr - our_bin_ufl_std_arr,
                     our_bin_ufl_mean_arr + our_bin_ufl_std_arr, color="blue", alpha=0.3)
    ax1.set_ylabel("BUFL", fontsize=17, color="b")

    plt.xlabel(r"Iterations", fontsize=17)

    # plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(exp_dir.joinpath(f"detect_fake_speech_bufl_{tts_name}.png"))
    plt.close()

    # --------------------------------------------
    # plot EER
    plt.figure(figsize=(6, 6))
    plt.title(f"Detecting fake speech by {tts_name}", fontsize=17)

    plt.ylim(-1, 101)
    plt.plot(iter_has_run_arr, our_eer_mean_arr, "-bo", linewidth=2, label="EER")
    plt.fill_between(iter_has_run_arr, our_eer_mean_arr - our_eer_std_arr,
                     our_eer_mean_arr + our_eer_std_arr, color="blue", alpha=0.3)
    plt.ylabel("EER (%)", fontsize=17, color="b")

    plt.xlabel(r"Iterations", fontsize=17)

    plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(exp_dir.joinpath(f"detect_fake_speech_eer_{tts_name}.png"))
    plt.close()

    # --------------------------------------------
    # plot Binary EER
    plt.figure(figsize=(6, 6))
    plt.title(f"Detecting fake speech by {tts_name}", fontsize=17)

    plt.ylim(-1, 101)
    plt.plot(iter_has_run_arr, our_bin_eer_mean_arr, "-bo", linewidth=2, label="BEER")
    plt.fill_between(iter_has_run_arr, our_bin_eer_mean_arr - our_bin_eer_std_arr,
                     our_bin_eer_mean_arr + our_bin_eer_std_arr, color="blue", alpha=0.3)
    plt.ylabel("BEER (%)", fontsize=17, color="b")

    plt.xlabel(r"Iterations", fontsize=17)

    plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(exp_dir.joinpath(f"detect_fake_speech_bin_eer_{tts_name}.png"))
    plt.close()

    # --------------------------------------------
    # plot trend for each speaker
    plt.figure(figsize=(6, 6))
    plt.title(f"Comparing UFL of {tts_name} speakers", fontsize=17)

    for speaker_idx, speaker in enumerate(speaker_lst):

        speaker_ufl = []
        for ufl_lst in all_speakers_ufl_lst_lst:
            url_val = ufl_lst[speaker_idx]
            speaker_ufl.append(url_val)
        plt.plot(iter_has_run_arr, speaker_ufl, linewidth=2, label=speaker[-4:])

    plt.xlabel(r"Iterations", fontsize=17)
    plt.ylabel("Seconds", fontsize=17)
    plt.ylim(0, 300)
    plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(exp_dir.joinpath(f"ufl_each_speaker_{tts_name}.png"))
    plt.close()

    # --------------------------------------------
    # plot acc trend for each speaker
    plt.figure(figsize=(6, 6))
    plt.title(f"Comparing TPR of {tts_name} speakers", fontsize=16)

    for speaker_idx, speaker in enumerate(speaker_lst):

        speaker_acc = []
        for acc_lst in all_speakers_acc_lst_lst:
            acc_val = acc_lst[speaker_idx]
            speaker_acc.append(acc_val)
        plt.plot(iter_has_run_arr, speaker_acc, linewidth=2, label=speaker[-4:])

    plt.xlabel(r"Iterations", fontsize=17)
    plt.ylabel("TPR (%)", fontsize=17)
    plt.ylim(-1, 101)
    plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(exp_dir.joinpath(f"acc_each_speaker_{tts_name}.png"))
    plt.close()

    # --------------------------------------------
    # plot eer trend for each speaker
    plt.figure(figsize=(6, 6))
    plt.title(f"Comparing EER of {tts_name} speakers", fontsize=17)

    for speaker_idx, speaker in enumerate(speaker_lst):

        speaker_eer = []
        for eer_lst in all_speakers_eer_lst_lst:
            eer_val = eer_lst[speaker_idx]
            speaker_eer.append(eer_val)
        plt.plot(iter_has_run_arr, speaker_eer, linewidth=2, label=speaker[-4:])

    plt.xlabel(r"Iterations", fontsize=17)
    plt.ylabel("EER (%)", fontsize=17)
    plt.ylim(-1, 101)
    plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(exp_dir.joinpath(f"eer_each_speaker_{tts_name}.png"))
    plt.close()

    # --------------------------------------------
    # then get some fake speech to plot
    largest_iter = exp_cfg.speaker_adapt_iters[-1]
    fake_speech_speaker_audio_wm_dic = iters_speaker_audio_wm_dic[largest_iter]
    for speaker_name, audio_lst in fake_speech_speaker_audio_wm_dic.items():
        for i in range(3):
            our_wm_audio = audio_lst[i]["our_wm_audio"]
            plot_spectrogram(our_wm_audio, f"Spectrogram of {tts_name} fake speech",
                             exp_dir.joinpath(f"{speaker_name}_fake_speech_{i+1}_{tts_name}.png"))


def plot_commercial_fake_speech(exp_dir, commercial_name, speaker_audio_wm_dic):
    exp_dir.mkdir(exist_ok=True)

    # we generate a long speech for each speaker, so just plot random sections
    speech_len = 4 * exp_cfg.sr

    our_fake_len_lst = []
    # wavmark_fake_len_lst = []

    rand_state = np.random.RandomState(seed=23734)
    for speaker_name, audio_lst in speaker_audio_wm_dic.items():
        assert len(audio_lst) == 1
        our_wm_audio = audio_lst[0]["our_wm_audio"]

        our_fake_len_lst.append(len(our_wm_audio))
        # wavmark_fake_len_lst.append(len(audio_lst[0]["wavmark_wm_audio"]))

        for i in range(10):
            start_pos = rand_state.randint(len(our_wm_audio) - speech_len)
            audio_section = our_wm_audio[start_pos: start_pos + speech_len]
            plot_spectrogram(audio_section, f"Spectrogram of {commercial_name} fake speech",
                             exp_dir.joinpath(f"{commercial_name}_{speaker_name}_fake_speech_{i+1}.png"))

    with open(exp_dir.joinpath("length.txt"), "w") as f:
        f.write(f"our_fake_len_lst = {np.mean(our_fake_len_lst):.2f} +- {np.std(our_fake_len_lst):.2f}\n\n")
        # f.write(f"wavmark_fake_len_lst = {np.mean(wavmark_fake_len_lst):.2f} +- {np.std(wavmark_fake_len_lst):.2f}")


def plot_attack_rlts(out_dir, attack_eval_dir_lst, adapt_iters_lst, fig_fname):
    our_ufl_mean_arr, our_ufl_std_arr = [], []
    our_bin_ufl_mean_arr, our_bin_ufl_std_arr = [], []

    our_acc_mean_arr, our_acc_std_arr = [], []
    our_bin_acc_mean_arr, our_bin_acc_std_arr = [], []

    our_eer_mean_arr, our_eer_std_arr = [], []
    our_bin_eer_mean_arr, our_bin_eer_std_arr = [], []

    for adapt_iter, attack_eval_dir in zip(adapt_iters_lst, attack_eval_dir_lst):
        assert attack_eval_dir.name.find(f"iter_{adapt_iter}") >= 0

        with open(attack_eval_dir.joinpath("speaker_lst.bin"), 'rb') as handle:
            speaker_lst = pickle.load(handle)

        with open(attack_eval_dir.joinpath("our_ufl_arr.bin"), 'rb') as handle:
            our_ufl_arr = pickle.load(handle)
        with open(attack_eval_dir.joinpath("our_bin_ufl_arr.bin"), 'rb') as handle:
            our_bin_ufl_arr = pickle.load(handle)

        with open(attack_eval_dir.joinpath("our_acc_arr.bin"), 'rb') as handle:
            our_acc_arr = pickle.load(handle)
            our_acc_mean_arr.append(np.mean(our_acc_arr))
            our_acc_std_arr.append(np.std(our_acc_arr))

        with open(attack_eval_dir.joinpath("our_bin_acc_arr.bin"), 'rb') as handle:
            our_bin_acc_arr = pickle.load(handle)
            our_bin_acc_mean_arr.append(np.mean(our_bin_acc_arr))
            our_bin_acc_std_arr.append(np.std(our_bin_acc_arr))

        with open(attack_eval_dir.joinpath("our_eer_arr.bin"), 'rb') as handle:
            our_eer_arr = pickle.load(handle)
            our_eer_mean_arr.append(np.mean(our_eer_arr))
            our_eer_std_arr.append(np.std(our_eer_arr))

        with open(attack_eval_dir.joinpath("our_bin_eer_arr.bin"), 'rb') as handle:
            our_bin_eer_arr = pickle.load(handle)
            our_bin_eer_mean_arr.append(np.mean(our_bin_eer_arr))
            our_bin_eer_std_arr.append(np.std(our_bin_eer_arr))

        def _to_load(_data_lst_lst, _mean_lst, _std_lst):
            _all_data = []
            for _data in _data_lst_lst:
                _all_data.extend(_data)
            if len(_all_data) == 0:
                _mean_lst.append(0)
                _std_lst.append(0)
            else:
                _all_data = np.array(_all_data)
                _mean_lst.append(np.mean(_all_data))
                _std_lst.append(np.std(_all_data))

            return

        _to_load(our_ufl_arr, our_ufl_mean_arr, our_ufl_std_arr)
        _to_load(our_bin_ufl_arr, our_bin_ufl_mean_arr, our_bin_ufl_std_arr)

    our_ufl_mean_arr = np.array(our_ufl_mean_arr)
    our_ufl_std_arr = np.array(our_ufl_std_arr)
    our_bin_ufl_mean_arr = np.array(our_bin_ufl_mean_arr)
    our_bin_ufl_std_arr = np.array(our_bin_ufl_std_arr)

    our_acc_mean_arr = np.array(our_acc_mean_arr) * 100.0
    our_acc_std_arr = np.array(our_acc_std_arr) * 100.0
    our_bin_acc_mean_arr = np.array(our_bin_acc_mean_arr) * 100.0
    our_bin_acc_std_arr = np.array(our_bin_acc_std_arr) * 100.0

    our_eer_mean_arr = np.array(our_eer_mean_arr) * 100.0
    our_eer_std_arr = np.array(our_eer_std_arr) * 100.0
    our_bin_eer_mean_arr = np.array(our_bin_eer_mean_arr) * 100.0
    our_bin_eer_std_arr = np.array(our_bin_eer_std_arr) * 100.0

    out_dir.mkdir(exist_ok=True)

    # plot UFL
    sns.set_style("darkgrid")
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    fig, ax1 = plt.subplots(figsize=(6, 6))
    plt.title(f"Robustness of our watermarking", fontsize=17)

    iter_has_run_arr = np.array(adapt_iters_lst) - 1  # our callback is on_start_iter

    ax2 = plt.twinx()
    ax2.plot(iter_has_run_arr, our_acc_mean_arr, "-ro", linewidth=2, label="TPR", alpha=0.8)
    ax2.fill_between(iter_has_run_arr, our_acc_mean_arr - our_acc_std_arr, our_acc_mean_arr + our_acc_std_arr,
                     color="red", alpha=0.3)
    ax2.set_ylabel("TPR (%)", color='r', fontsize=17)
    ax2.set_ylim(-1, 100)
    ax2.grid(False)

    ax1.plot(iter_has_run_arr, our_ufl_mean_arr, "-bo", linewidth=2, label="UFL", alpha=0.8)
    ax1.fill_between(iter_has_run_arr, our_ufl_mean_arr - our_ufl_std_arr, our_ufl_mean_arr + our_ufl_std_arr,
                     color="blue", alpha=0.3)
    ax1.set_ylabel("UFL", fontsize=17, color="b")
    ax1.set_ylim(-1, 300)

    plt.xlabel(r"Iterations", fontsize=17)

    # plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(out_dir.joinpath(f"{fig_fname}_UFL.png"))
    plt.close()

    # plot BUFL
    fig, ax1 = plt.subplots(figsize=(6, 6))
    plt.title(f"Robustness of our watermarking", fontsize=17)

    ax2 = plt.twinx()
    ax2.plot(iter_has_run_arr, our_bin_acc_mean_arr, "-ro", linewidth=2, label="BTPR", alpha=0.8)
    ax2.fill_between(iter_has_run_arr, our_bin_acc_mean_arr - our_bin_acc_std_arr,
                     our_bin_acc_mean_arr + our_bin_acc_std_arr,
                     color="red", alpha=0.3)
    ax2.set_ylabel("BTPR (%)", color='r', fontsize=17)
    ax2.set_ylim(-1, 100)
    ax2.grid(False)

    ax1.set_ylim(-1, 300)
    ax1.plot(iter_has_run_arr, our_bin_ufl_mean_arr, "-bo", linewidth=2, label="BUFL", alpha=0.8)
    ax1.fill_between(iter_has_run_arr, our_bin_ufl_mean_arr - our_bin_ufl_std_arr,
                     our_bin_ufl_mean_arr + our_bin_ufl_std_arr, color="blue", alpha=0.3)
    ax1.set_ylabel("BUFL", fontsize=17, color="b")

    plt.xlabel(r"Iterations", fontsize=17)

    # plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(out_dir.joinpath(f"{fig_fname}_BUFL.png"))
    plt.close()

    # --------------------------------------------
    # plot EER
    plt.figure(figsize=(6, 6))
    plt.title(f"Robustness of our watermarking", fontsize=17)

    plt.ylim(-1, 101)
    plt.plot(iter_has_run_arr, our_eer_mean_arr, "-bo", linewidth=2, label="EER")
    plt.fill_between(iter_has_run_arr, our_eer_mean_arr - our_eer_std_arr,
                     our_eer_mean_arr + our_eer_std_arr, color="blue", alpha=0.3)
    plt.ylabel("EER (%)", fontsize=17, color="b")

    plt.xlabel(r"Iterations", fontsize=17)

    plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(out_dir.joinpath(f"{fig_fname}_EER.png"))
    plt.close()

    # --------------------------------------------
    # plot Binary EER
    plt.figure(figsize=(6, 6))
    plt.title(f"Robustness of our watermarking", fontsize=17)

    plt.ylim(-1, 101)
    plt.plot(iter_has_run_arr, our_bin_eer_mean_arr, "-bo", linewidth=2, label="BEER")
    plt.fill_between(iter_has_run_arr, our_bin_eer_mean_arr - our_bin_eer_std_arr,
                     our_bin_eer_mean_arr + our_bin_eer_std_arr, color="blue", alpha=0.3)
    plt.ylabel("BEER (%)", fontsize=17, color="b")

    plt.xlabel(r"Iterations", fontsize=17)

    plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(out_dir.joinpath(f"{fig_fname}_BEER.png"))
    plt.close()

    return


def main():
    """
    Using our metric to evaluate our method.
    """
    exp_dir = Path(exp_cfg.out_dir).joinpath("ExpEvaluate")
    exp_dir.mkdir(exist_ok=True)

    # firstly, load all the audio
    speaker_audio_wm_dic = get_speaker_audio_wm_dic(exp_cfg)

    # initialize our watermark net
    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, exp_cfg.wm_length)

    # load our watermark net
    audio_sec_len = 16000           # by default, WatermarkNet split audio into 1-second sections
    wm_net = WatermarkNet(benign_encoded_wm, audio_sec_len, audio_sec_len,
                          wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir)
    model_dir = Path(exp_cfg.out_dir).joinpath("ExpEmbedWatermark")
    dic_saved = ModelTrainer.load_latest_ckpt(model_dir.joinpath("ckpt"))
    wm_net.load_state_dict(dic_saved["model_state"])
    wm_net = wm_net.to(utils.device)
    wm_net.eval()

    #############################################
    # We do not evaluate wavenet anymore as this method is weak.

    # load wavmark model
    # wavmark_net = wavmark.load_model().to(utils.device)
    # assert wavmark_net.training is False
    wavmark_net = None
    ##############################################

    # evaluate quality of watermarked audio
    eval_wm_quality(exp_dir.joinpath("eval_wm_quality"), speaker_audio_wm_dic)

    plot_our_spectrograms(exp_dir.joinpath("plot_our_spectrograms"), speaker_audio_wm_dic)

    # evaluate metrics of watermarked audio
    eval_metrics(exp_dir.joinpath("eval_metrics"),
                 speaker_audio_wm_dic=speaker_audio_wm_dic,
                 wm_net=wm_net, wavmark_net=wavmark_net)

    # evaluate metrics on YourTTS and SV2TTS data
    local_tts_loaded_dic = {}       # save loaded data for evaluating attacks
    for (tts_name, tts_exp_name, save_for_eval_attacks) in \
            [
                ["YourTTS Mix",     "ExpSpeakerAdaptYourTTS_Pirate", False],
                ["YourTTS",         "ExpSpeakerAdaptYourTTS", True],
                ["SV2TTS",          "ExpSpeakerAdaptSV2TTS", True],
                ["YourTTS 80%",     "ExpSpeakerAdaptYourTTS_wm_ratio_0.80", False],
                ["YourTTS 60%",     "ExpSpeakerAdaptYourTTS_wm_ratio_0.60", False],
                ["YourTTS 40%",     "ExpSpeakerAdaptYourTTS_wm_ratio_0.40", False],
                ["YourTTS 20%",     "ExpSpeakerAdaptYourTTS_wm_ratio_0.20", False],
                ["YourTTS AE",      "ExpSpeakerAdaptYourTTS_Autoencoder", False],
            ]:

        tts_iters_speaker_audio_wm_dic = load_all_local_tts_fake_speech(tts_exp_name)

        if save_for_eval_attacks is True:
            logging.info(f"Saving {tts_name} for evaluating attacks...")
            local_tts_loaded_dic[tts_name] = tts_iters_speaker_audio_wm_dic

        for cur_iter, tts_fake_speech_speaker_audio_wm_dic in tts_iters_speaker_audio_wm_dic.items():
            eval_metrics(exp_dir.joinpath(f"eval_{tts_name}_fake_speech_iter_{cur_iter:04d}"),
                         speaker_audio_wm_dic=tts_fake_speech_speaker_audio_wm_dic,
                         wm_net=wm_net, wavmark_net=wavmark_net)

        # plot yourtts fake speech
        plot_local_tts_fake_speech(exp_dir.joinpath(f"plot_{tts_name}_fake_speech"), tts_name, tts_iters_speaker_audio_wm_dic)


    # evaluate metrics on PlayHT and Speechify
    commercial_loaded_dic = {}  # save loaded data for evaluating attacks
    for commercial_name in ["PlayHT", "Speechify"]:

        commercial_speaker_audio_wm_dic = load_all_commercial_fake_speech(commercial_name)
        commercial_loaded_dic[commercial_name] = commercial_speaker_audio_wm_dic

        eval_metrics(exp_dir.joinpath(f"eval_commercial_{commercial_name}_fake_speech"),
                     speaker_audio_wm_dic=commercial_speaker_audio_wm_dic,
                     wm_net=wm_net, wavmark_net=wavmark_net)

        # also plot some images of PlayHT
        plot_commercial_fake_speech(exp_dir.joinpath(f"plot_{commercial_name}_fake_speech"),
                                    commercial_name, commercial_speaker_audio_wm_dic)

    # evaluate robustness against attacks
    attack_lst = [
        attack_add_noise, attack_mp3, attack_opus, attack_resample,
        attack_high_pass, attack_low_pass, attack_supress, attack_median_filter,
        attack_scale, attack_quantization, attack_add_echo, attack_noise_reduction,

        # adaptive attacks
        attack_adaptive_100Hz, attack_adaptive_200Hz, attack_adaptive_400Hz,
    ]

    np.random.seed(46083223)    # adaptive attacks are based on random numbers

    for attack in attack_lst:
        # adaptive attacks are only applied to our method
        is_adaptive = (attack.__name__.find("attack_adaptive") == 0)

        eval_metrics(exp_dir.joinpath(f"eval_metrics_{attack.__name__}"),
                     speaker_audio_wm_dic=speaker_audio_wm_dic,
                     wm_net=wm_net, wavmark_net=wavmark_net if is_adaptive is False else None,
                     attack=attack)

        # also evaluate robustness of the fake speech generated by the maximum number of fine-tuning
        for tts_name, tts_iters_speaker_audio_wm_dic in local_tts_loaded_dic.items():
            largest_iter = exp_cfg.speaker_adapt_iters[-1]
            tts_fake_speech_speaker_audio_wm_dic = tts_iters_speaker_audio_wm_dic[largest_iter]
            eval_metrics(exp_dir.joinpath(f"eval_{tts_name}_fake_speech_iter_{largest_iter}_{attack.__name__}"),
                         speaker_audio_wm_dic=tts_fake_speech_speaker_audio_wm_dic,
                         wm_net=wm_net, wavmark_net=None,       # fake speech by wavmark does not need to be evaluated
                         attack=attack)

        for commercial_name, commercial_speaker_audio_wm_dic in commercial_loaded_dic.items():
            eval_metrics(exp_dir.joinpath(f"eval_{commercial_name}_fake_speech_{attack.__name__}"),
                         speaker_audio_wm_dic=commercial_speaker_audio_wm_dic,
                         wm_net=wm_net, wavmark_net=None,       # fake speech by wavmark does not need to be evaluated
                         attack=attack)

    # extra evaluation on YourTTS to compare our method and the NDSS paper
    for attack in [attack_combo_noise_and_resample]:
        for tts_name, tts_iters_speaker_audio_wm_dic in local_tts_loaded_dic.items():
            if tts_name != "YourTTS":
                continue

            attack_eval_dir_lst =[]

            for adap_iter in exp_cfg.speaker_adapt_iters:
                tts_fake_speech_speaker_audio_wm_dic = tts_iters_speaker_audio_wm_dic[adap_iter]

                attack_eval_dir = exp_dir.joinpath(f"eval_{tts_name}_fake_speech_iter_{adap_iter}_{attack.__name__}")
                attack_eval_dir_lst.append(attack_eval_dir)

                eval_metrics(attack_eval_dir,
                             speaker_audio_wm_dic=tts_fake_speech_speaker_audio_wm_dic,
                             wm_net=wm_net, wavmark_net=None,  # fake speech by wavmark does not need to be evaluated
                             attack=attack)

            plot_attack_rlts(
                out_dir=exp_dir.joinpath(f"plot_eval_{tts_name}_{attack.__name__}"),
                attack_eval_dir_lst=attack_eval_dir_lst,
                adapt_iters_lst=exp_cfg.speaker_adapt_iters,
                fig_fname=f"{tts_name}_{attack.__name__}")

    return  # Good luck!


if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()

    tmp_dir = Path(exp_cfg.out_dir).joinpath("tmp")

    main()

    sys.exit()






