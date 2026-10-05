import argparse
import pickle
from pathlib import Path
import scipy.io.wavfile as wav
import torch
from audiomark.amn_code.my_utils import utils
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from audiomark.amn_code.exp.ExpEmbedWatermark import get_existing_wm_samples

import matplotlib
import logging
matplotlib.use('Agg')
logging.getLogger().setLevel(logging.INFO)


# Keep all AudioMarkNet paths portable by resolving them from the repository
# copy of ``audiomark`` instead of from the process working directory.
AUDIOMARK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = AUDIOMARK_DIR / "out"
DEFAULT_DATA_DIR = AUDIOMARK_DIR / "data"
DEFAULT_PRETRAINED_DIR = AUDIOMARK_DIR / "pretrained_models"


def init_config():
    """
    Build a config whose default paths are relative to the local audiomark tree.
    """

    assert torch.cuda.device_count() > 0, "there must be gpus"

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--logging_level', default=20, type=int, help='')

    # server side
    parser.add_argument('--out_dir',
                        default=str(DEFAULT_OUT_DIR), type=str,
                        help='Root directory to save all the experimental results.')
    parser.add_argument('--data_dir',
                        default=str(DEFAULT_DATA_DIR), type=str,
                        help='Directory where all the data is stored.')

    # wav2vec2
    parser.add_argument('--wav2vec2_pretrained_dir', type=str,
                        default=str(DEFAULT_PRETRAINED_DIR / "wav2vec2"),
                        help='Directory where wav2vec2 weights are stored.')

    # coqui-ai
    os.environ.setdefault("TTS_HOME", str(DEFAULT_PRETRAINED_DIR / "coqui_ai"))

    # SV2TTS
    parser.add_argument('--SV2TTS_pretrained_dir',
                        default=str(DEFAULT_PRETRAINED_DIR / "sv2tts" / "default"), type=str,
                        help='Directory where SV2TTS weights are stored.')

    # common settings
    os.environ["TRAINER_TELEMETRY"] = "0"
    parser.add_argument('--coqui_ai_pretrained_dir', default=os.environ["TTS_HOME"], type=str, help='')

    parser.add_argument('--speaker_adapt_iters', type=int, nargs="+",
                        default=[1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101],     # iteration starts from 1
                        help='')

    parser.add_argument('--wmnet_aug_normal_prob', type=float, default=0.4, help='probability to augment with Gaussian')
    parser.add_argument('--wmnet_aug_normal_scale', type=float, default=0.04, help='scale of Gaussian noise')

    parser.add_argument('--sr', type=int, default=16000, help='sampling rate')

    # watermarking
    parser.add_argument('--wm_length', type=int, default=16, help='length for watermarks')

    # speaker encoder used by YourTTS
    parser.add_argument('--speaker_encoder_checkpoint_relpath', type=str, default="res_speaker_encoder/model_se.pth.tar", help='')
    parser.add_argument('--speaker_encoder_config_relpath', type=str, default="res_speaker_encoder/config_se.json", help='')

    ### adaptive iteration attacks
    parser.add_argument('--adap_iter_attack_enroll_num', type=int, default=3, help='number of enrolments for speaker verification')

    parser.add_argument('--adap_iter_attack_iters', type=int, nargs="+",
                        default=list(range(1, 31)),     # start from 1
                        help='Iterations to evaluate adaptive iteration attack.')

    cfg = parser.parse_args([])

    logging.basicConfig(level=cfg.logging_level)

    return cfg


##########################################################################
# common functions

def get_vctk_dir(exp_cfg):
    return Path(exp_cfg.data_dir).joinpath("vctk")


##########################################################################
# watermarking

HAMMING_DATA_BITS = 3
HAMMING_CODE_BITS = 6


def hamming_encode_wm_np(wm_np):
    # encode every block
    assert len(wm_np) % HAMMING_DATA_BITS == 0
    block_num = len(wm_np) // HAMMING_DATA_BITS
    encoded_wm_str = ""

    for block_idx in range(block_num):
        block = wm_np[block_idx * HAMMING_DATA_BITS: (block_idx + 1) * HAMMING_DATA_BITS]

        block = [f"{x}" for x in block]
        block = "".join(block)
        assert len(block) == HAMMING_DATA_BITS
        block = utils.hamming_encode(int(block, 2), len(block))
        encoded_wm_str += block

    np_encoded_wm = np.zeros(len(encoded_wm_str)).astype(np.int64)

    for idx, bit in enumerate(encoded_wm_str):
        np_encoded_wm[idx] = int(bit)

    return np_encoded_wm


def hamming_decode_wm_np(wm_np):
    # decode every block
    assert len(wm_np) % HAMMING_CODE_BITS == 0
    block_num = len(wm_np) // HAMMING_CODE_BITS
    decoded_wm_str = ""

    for block_idx in range(block_num):
        block = wm_np[block_idx * HAMMING_CODE_BITS: (block_idx + 1) * HAMMING_CODE_BITS]

        block = [f"{x}" for x in block]
        block = "".join(block)
        assert len(block) == HAMMING_CODE_BITS
        block = utils.hamming_decode(block, len(block))
        decoded_wm_str += block

    np_decoded_wm = np.zeros(len(decoded_wm_str)).astype(np.int64)

    for idx, bit in enumerate(decoded_wm_str):
        np_decoded_wm[idx] = int(bit)

    return np_decoded_wm


def hamming_decode_wm_tensor(wm_tensor):
    wm_np = wm_tensor.cpu().detach().numpy()

    if len(wm_np.shape) == 1:
        decoded = hamming_decode_wm_np(wm_np)

    elif len(wm_np.shape) == 2:
        tmp_lst = []
        for wm in wm_np:
            wm = hamming_decode_wm_np(wm)
            tmp_lst.append(wm)

        decoded = np.array(tmp_lst)

    else:
        assert False, "unsupported dimension of wm_tensor"

    return torch.from_numpy(decoded).to(utils.device)


def get_speakers_and_wm(exp_cfg, wm_len, using_hamming_code=False, speakers_seed_dic=None):
    # the ignored test speakers in yourstts experiment 1
    default_seed_dic = {
        "benign": 17,

        "VCTK_p261": 100,       # female
        "VCTK_p225": 205,       # female
        "VCTK_p294": 300,       # female
        "VCTK_p347": 400,       # male
        "VCTK_p238": 500,       # female
        "VCTK_p234": 600,       # female
        "VCTK_p248": 700,       # female
        "VCTK_p335": 800,       # female
        "VCTK_p245": 900,       # male
        "VCTK_p326": 1000,      # male
        "VCTK_p302": 1100,      # male

        # "Obama": 2000,
    }

    if speakers_seed_dic is None:
        speakers_seed_dic = default_seed_dic
    else:
        # using a new dict other than default one.
        # make sure speakers and seeds are all different in the new dict.
        default_seeds = [v for k, v in default_seed_dic.items()]
        for k, v in speakers_seed_dic.items():
            if v in default_seeds:
                assert False, "Find repeated seeds!"

            if k == "benign":
                continue
            # if k in default_seed_dic:
            #     assert False, "Find repeated speakers!"

    speakers_lst = []

    org_wm_lst = []
    if using_hamming_code is True:
        # use hamming code to encode watermarks
        assert wm_len % HAMMING_CODE_BITS == 0, "watermarks should be fully divided by hamming code"

        wm_len = wm_len // 2    # half of the code is used for redundancy

    for speaker, wm_seed in speakers_seed_dic.items():
        speakers_lst.append(speaker)

        org_wm = (np.random.RandomState(seed=wm_seed).uniform(size=wm_len) > 0.5).astype(np.int64)
        org_wm_lst.append(org_wm)

    def print_max_overlapping(_wm_lst, title):
        # check whether messages overlaps
        _wm_len = len(_wm_lst[0])
        wm_np = np.array(_wm_lst)
        wm_np_neg = 1 - wm_np
        wm_mtrx = np.matmul(wm_np, wm_np.T) + np.matmul(wm_np_neg, wm_np_neg.T)
        wm_mtrx = wm_mtrx.astype(float) / _wm_len
        wm_mtrx = wm_mtrx * (1-np.eye(wm_mtrx.shape[0]))

        print(f"{title} maximum overlapping = {wm_mtrx.max()}; wm_len = {_wm_len}")

    print_max_overlapping(org_wm_lst, "original watermarks")

    encoded_wm_lst = []
    if using_hamming_code:
        # encoding all watermarks
        assert len(org_wm_lst[0]) % HAMMING_DATA_BITS == 0, "original watermark must also be divided by data bits."

        for org_wm in org_wm_lst:
            np_encoded_wm = hamming_encode_wm_np(org_wm)
            assert len(np_encoded_wm) == wm_len * 2, "length of encoded watermarks is not as expected."
            encoded_wm_lst.append(np_encoded_wm)

            assert (hamming_decode_wm_np(np_encoded_wm) == org_wm).sum() == len(org_wm), "encode decode not invertible?"

            print_max_overlapping(encoded_wm_lst, "encoded watermarks")

    else:
        print("Do not use hamming code for watermarks.")
        encoded_wm_lst = org_wm_lst     # no encoding is used

    # combine to a dict
    speakers_wm_list = []
    benign_org_wm = None
    benign_encoded_wm = None
    for speaker, org_wm, encoded_wm in zip(speakers_lst, org_wm_lst, encoded_wm_lst):
        if speaker == "benign":
            assert benign_org_wm is None and benign_encoded_wm is None
            benign_org_wm = org_wm
            benign_encoded_wm = encoded_wm
        else:
            speakers_wm_list.append({"speaker": speaker, "org_wm": org_wm, "encoded_wm": encoded_wm})

    assert benign_org_wm is not None and benign_encoded_wm is not None
    return speakers_wm_list, benign_org_wm, benign_encoded_wm


def get_gen_sentences_lst(exp_cfg):
    txt_path = Path(exp_cfg.out_dir).joinpath("gen_sentences.txt")
    with open(txt_path, "r") as f:
        all_lines = f.readlines()

    for idx, line in enumerate(all_lines):
        all_lines[idx] = line.strip()

    return all_lines


##########################################################################
# Others

def process_Obama_voice(exp_cfg):
    wav_dir = Path(exp_cfg.data_dir).joinpath("CopyrightFreeSpeeches/Obama")
    wav_path = wav_dir.joinpath("barackobamatransitionaddress7.wav")
    config_path = wav_dir.joinpath("barackobamatransitionaddress7.txt")

    missing_files = [path for path in (wav_path, config_path) if not path.is_file()]
    if missing_files:
        expected = "\n  ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Obama source data is missing. Expected these files relative to "
            f"the AudioMarkNet data directory:\n  {expected}"
        )

    sections_dir = wav_dir.joinpath(wav_path.stem)
    sections_dir.mkdir(parents=True, exist_ok=True)

    full_audio, _ = utils.read_audio(wav_path, exp_cfg.sr)
    audio_dic_lst = []

    with open(config_path, "r") as f:
        all_lines = f.readlines()

    for idx, line in enumerate(all_lines):
        line = line.strip()
        first_space = line.find(" ")
        second_space = line[first_space + 1:].find(" ") + first_space + 1
        transcript = line[second_space+1:]

        section_path = sections_dir.joinpath(f"section_{idx:03d}.wav")
        if not section_path.exists():
            start_pos = int(float(line[:first_space]) * exp_cfg.sr)
            end_pos = int(float(line[first_space + 1: second_space]) * exp_cfg.sr)

            wav_sec = full_audio[start_pos:end_pos]
            wav.write(section_path, exp_cfg.sr, wav_sec)

            with open(section_path.with_name(f"{section_path.stem}.txt"), "w") as f:
                print(transcript, file=f)

        tgt_sample = {
            "text": transcript,
            "audio_file": section_path,
            "speaker_name": "Obama",
            "root_path": str(sections_dir),
            "language": "en",
            "audio_unique_name": f"Obama_{idx + 1}",
        }
        audio_dic_lst.append(tgt_sample)

    return audio_dic_lst


def setup_adaptive_iter_exps(exp_cfg):
    tgt_name = "Obama"  # this speaker is different from the training set

    out_dir = Path(exp_cfg.out_dir)
    from run_wm_speech_Obama import g_speakers_seed_dic
    speakers_seed_dic = g_speakers_seed_dic
    wm_exp_dir = out_dir.joinpath(f"ExpEmbedWatermark_{tgt_name}")

    speakers_wm_lst, tgt_benign_org_wm, benign_encoded_wm = get_speakers_and_wm(
        exp_cfg, exp_cfg.wm_length,
        speakers_seed_dic=speakers_seed_dic
    )

    speaker_wm_dic = None
    for _speaker_wm_dic in speakers_wm_lst:
        if _speaker_wm_dic["speaker"] == tgt_name:
            speaker_wm_dic = _speaker_wm_dic
            break
    assert speaker_wm_dic is not None, f"Where is f{tgt_name}?"

    tgt_audio_dic_lst = get_existing_wm_samples(tgt_name, wm_exp_dir)
    assert len(tgt_audio_dic_lst) > 0, f"Cannot find watermarked speech of {tgt_name}."
    # we use the first few audios to enrol
    enrol_num = exp_cfg.adap_iter_attack_enroll_num
    enrol_speech = []
    for idx in range(enrol_num):
        org_speech_path = tgt_audio_dic_lst[idx]["audio_file_org"]
        org_speech, sr = utils.read_audio(org_speech_path, expected_sr=exp_cfg.sr)
        enrol_speech.append(org_speech)
    enrol_speech = np.concatenate(enrol_speech)
    print(f"enrol speech length = {len(enrol_speech) / exp_cfg.sr:.2f} seconds")

    gen_sentences_lst = [
        # "He began a confused complaint against the wizard who had vanished behind the curtain on the left. "
        # "Give not so earnest a mind to these mummeries child. "
        # "A golden fortune and a happy life. "
        # "Forthwith all ran to the opening of the tent to see what might be amiss but master will who peeped out first needed no more than one glance. "
        # "He was like unto my father in a way and yet was not my father. "
        # "Also there was a stripling page who turned into a maid. "
        # "This was so sweet a lady sir and in some manner i do think she died. "
        # "But then the picture was gone as quickly as it came. "
        # "Sister nell do you hear these marvels. "
        # "Take your place and let us see what the crystal can show to you. "
        # "Like as not young master though i am an old man."

        "He began a confused complaint against the wizard who had vanished behind the curtain on the left. Give not so earnest a mind to these mummeries child. A golden fortune and a happy life. He was like unto my father in a way and yet was not my father. Also there was a stripling page who turned into a maid. This was so sweet a lady sir and in some manner i do think she died. But then the picture was gone as quickly as it came. Sister nell do you hear these marvels. Take your place and let us see what the crystal can show to you. Like as not young master though i am an old man. Forthwith all ran to the opening of the tent to see what might be amiss but master will who peeped out first needed no more than one glance. He gave way to the others very readily and retreated unperceived by the squire and mistress fitzooth to the rear of the tent. Cries of a nottingham a nottingham. Before them fled the stroller and his three sons capless and terrified."

    ]

    run_times = 5

    logging.info(f"setup_adaptive_iter_exps:\n"
                 f"wm_exp_dir = {wm_exp_dir}\n"
                 f"tgt_name = {tgt_name}\n"
                 f"enrol_speech = {enrol_speech}\n")

    return (wm_exp_dir, speakers_wm_lst, benign_encoded_wm, tgt_name, enrol_speech,
            tgt_audio_dic_lst, speaker_wm_dic, gen_sentences_lst, run_times)


def plot_adaptive_iter_exps(exp_base_dir, run_times, tts_name):

    # read all the data
    all_scores_arr = []
    all_url_arr = []
    all_bin_ufl_arr = []

    all_acc_arr = []
    all_bin_acc_arr = []

    all_fpr_arr = []

    exp_iter_lsts = None

    for cur_run in range(run_times):

        with open(exp_base_dir.joinpath(f"run_{cur_run}/exp.status"), 'rb') as handle:
            status_dic = pickle.load(handle)

        all_scores_arr.append(status_dic["scores_lst"])
        all_acc_arr.append(status_dic["acc_lst"])
        all_bin_acc_arr.append(status_dic["bin_acc_lst"])

        all_fpr_arr.append(status_dic["fpr_lst"])

        def get_mean(_arr):
            mean_arr = []
            for _v in _arr:
                assert len(_v) == 1, "Should have only one speaker"
                _v = _v[0]
                if len(_v) == 0:
                    mean_arr.append(0)
                else:
                    mean_arr.append(np.mean(_v))
            return mean_arr

        all_url_arr.append(get_mean(status_dic["ufl_lst"]))
        all_bin_ufl_arr.append(get_mean(status_dic["bin_ufl_lst"]))

        iter_lsts = status_dic["iter_lsts"]

        if exp_iter_lsts is None:
            exp_iter_lsts = iter_lsts
        else:
            # all experiment runs must have the same set of iterations
            for (a, b) in zip(exp_iter_lsts, iter_lsts):
                assert a == b

    all_scores_arr = np.array(all_scores_arr)
    all_url_arr = np.array(all_url_arr)
    all_bin_ufl_arr = np.array(all_bin_ufl_arr)

    all_acc_arr = np.array(all_acc_arr).squeeze(-1) * 100.0             # to percentage
    all_bin_acc_arr = np.array(all_bin_acc_arr).squeeze(-1) * 100.0     # to percentage

    all_fpr_arr = np.array(all_fpr_arr)
    print(f"all_fpr_arr = [{all_fpr_arr.min()}, {all_fpr_arr.max()}]")


    def do_plot(fig_path, title, mean_arr, std_arr, ylim, label,
                extra_mean_arr=None, extra_std_arr=None, extra_ylim=None, extra_label=None,
                threshold=None):
        # plot overall figure
        sns.set_style("darkgrid")
        matplotlib.rc('xtick', labelsize=14)
        matplotlib.rc('ytick', labelsize=14)

        fig, ax1 = plt.subplots(figsize=(6, 6))
        plt.title(f"{tts_name} {title}", fontsize=20)

        plot_iters = np.array(exp_iter_lsts)
        plot_iters = plot_iters - 1  # offset one as our callback is on start.

        ax1.plot(plot_iters, mean_arr, "-bo", linewidth=2, label=label, alpha=0.8)
        ax1.fill_between(plot_iters, mean_arr - std_arr, mean_arr + std_arr, color="blue", alpha=0.3)
        ax1.set_xlabel(r"Iterations", fontsize=20)
        ax1.set_ylim(ylim[0], ylim[1])

        if extra_mean_arr is not None:
            ax1.set_ylabel(label, color="b", fontsize=20)
        else:
            ax1.set_ylabel(label, fontsize=20)
            plt.legend(fontsize=14)

        if extra_mean_arr is not None:
            ax2 = plt.twinx()

            ax2.plot(plot_iters, extra_mean_arr, "-ro", linewidth=2, label=label, alpha=0.8)
            ax2.fill_between(plot_iters, extra_mean_arr - extra_std_arr, extra_mean_arr + extra_std_arr, color="red", alpha=0.3)

            ax2.set_ylabel(extra_label, color='r', fontsize=20)
            ax2.set_ylim(extra_ylim[0], extra_ylim[1])
            ax2.grid(False)

        if threshold is not None:
            plt.plot(plot_iters, [threshold]*len(plot_iters), "--r", linewidth=2, label="threshold")


        # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

    do_plot(exp_base_dir.joinpath(f"{exp_base_dir.name}_ufl.png"), title="Watermark Detection",
            mean_arr=np.mean(all_url_arr, 0), std_arr=np.std(all_url_arr, 0),
            ylim=[-1, 60], label="UFL",
            extra_mean_arr=np.mean(all_acc_arr, 0), extra_std_arr=np.std(all_acc_arr, 0),
            extra_ylim=[-1, 100], extra_label="TPR (%)",
            )

    do_plot(exp_base_dir.joinpath(f"{exp_base_dir.name}_bin_ufl.png"), title="Watermark Detection",
            mean_arr=np.mean(all_bin_ufl_arr, 0), std_arr=np.std(all_bin_ufl_arr, 0),
            ylim=[-1, 60], label="BUFL",
            extra_mean_arr=np.mean(all_bin_acc_arr, 0), extra_std_arr=np.std(all_bin_acc_arr, 0),
            extra_ylim=[-1, 100], extra_label="BTPR (%)",
            )

    do_plot(exp_base_dir.joinpath(f"{exp_base_dir.name}_scores.png"), title="Speaker Verification",
            mean_arr=np.mean(all_scores_arr, 0), std_arr=np.std(all_scores_arr, 0),
            ylim=[-1.8, -0.5], label="scores", threshold=-1.11)




