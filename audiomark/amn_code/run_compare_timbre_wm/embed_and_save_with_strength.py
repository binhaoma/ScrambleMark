import os
import sys
from datetime import datetime

import torch
import yaml
import logging
import argparse
import warnings
import numpy as np
from rich.progress import track
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.loss import Loss
from torch.nn.functional import mse_loss
import soundfile
import random
import pdb
import pickle

from run_evaluate import attack_combo_noise_and_resample
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

import exp_setup
from pathlib import Path
from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig
from my_utils import utils
import torchaudio

import scipy.io.wavfile as wav

attack_lst = [attack_combo_noise_and_resample, None]


# set seeds
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

logging_mark = "#"*20
# warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

org_sr = 16000
tgt_sr = 22050

def save_flag(flag_path, flag=None):
    if flag is None:
        flag = datetime.now()

    with open(flag_path, 'wb') as handle:
        pickle.dump(flag, handle)


def load_flag(flag_path):
    if not flag_path.exists():
        return None

    with open(flag_path, 'rb') as handle:
        flag = pickle.load(handle)
    return flag


def main(args, configs):

    assert args.save_path.name.find(args.speaker) >= 0

    logging.info('main function')
    process_config, model_config, train_config = configs
    train_config["path"]["wm_speech"] = args.save_path
    # train_config["path"]["raw_path_test"] = args.original_path
    model_config["test"]["model_path"] = args.model_path

    pre_step = 0
    if model_config["structure"]["transformer"]:
        if model_config["structure"]["mel"]:
            from model.mel_modules import Encoder, Decoder
            from dataset.data import mel_dataset as my_dataset
        else:
            from model.modules import Encoder, Decoder
            from dataset.data import twod_dataset as my_dataset
    elif model_config["structure"]["conv2"]:
        from model.conv2_modules import Encoder, Decoder, Discriminator
        from dataset.data import mel_dataset_test as my_dataset
    elif model_config["structure"]["conv2mel"]:
        if not model_config["structure"]["ab"]:
            logging.info("use conv2mel model")
            from model.conv2_mel_modules2 import Encoder, Decoder, Discriminator
            from dataset.data import mel_dataset_test as my_dataset
        else:
            logging.info("use ablation conv2mel model")
            from model.conv2_mel_modules_ab import Encoder, Decoder, Discriminator
            from dataset.data import mel_dataset_test as my_dataset
    else:
        from model.conv_modules import Encoder, Decoder
        from dataset.data import oned_dataset as my_dataset
    # ---------------- get train dataset
    # audios = my_dataset(process_config=process_config, train_config=train_config)
    # batch_size = 1
    # assert batch_size < len(audios)
    # audios_loader = DataLoader(audios, batch_size=batch_size, shuffle=False)

    # ---------------- build model
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
    msg_length = train_config["watermark"]["length"]
    if model_config["structure"]["mel"] or model_config["structure"]["conv2"]:
        encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    else:
        encoder = Encoder(model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    
    path_model = model_config["test"]["model_path"]
    model_name = model_config["test"]["model_name"]
    if model_name:
        model = torch.load(os.path.join(path_model, model_name))
    else:
        index = model_config["test"]["index"]
        model_list = os.listdir(path_model)
        model_list = sorted(model_list,key=lambda x:os.path.getmtime(os.path.join(path_model,x)))
        model_path = os.path.join(path_model, model_list[index])
        logging.info(model_path)
        model = torch.load(model_path)
        logging.info("model <<{}>> loadded".format(model_path))
    # encoder = model["encoder"]
    # decoder = model["decoder"]
    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"], strict=False)
    encoder.eval()
    decoder.eval()
    decoder.robust = False
    # ---------------- Loss
    loss = Loss(train_config=train_config)

    # ---------------- Init logger
    # for p in train_config["path"].values():
    #     os.makedirs(p, exist_ok=True)

    # ---------------- Embedding
    logging.info(logging_mark + "\t" + "Begin Embedding" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    global_step = 0
    
    if not model_config["structure"]["ab"]:
        wm_path = os.path.join(train_config["path"]["wm_speech"],"wmed-"+str(args.wm),"wavs")
    else:
        wm_path = os.path.join(train_config["path"]["wm_speech"],"wmed_ab-"+str(args.wm),"wavs")

    wm_path = str(args.save_path)

    ref_path = os.path.join(train_config["path"]["wm_speech"],"ref")
    if not os.path.exists(wm_path): os.makedirs(wm_path)
    if not os.path.exists(ref_path): os.makedirs(ref_path)

    audios_loader = args.tgt_samples
    train_len = len(audios_loader)

    exp_rlt_dic_path = args.save_path.joinpath(f"exp_rlt_dic.bin")
    if not exp_rlt_dic_path.exists():
        exp_rlt_dic = {"wm": args.wm, "tgt_samples": args.tgt_samples}
        snr_val_lst = exp_rlt_dic["snr_val_lst"] = []
        pesq_val_lst = exp_rlt_dic["pesq_val_lst"] = []
        decoder_acc_lst = exp_rlt_dic["decoder_acc_lst"] = []

    else:
        # read existing rlts
        with open(exp_rlt_dic_path, 'rb') as handle:
            exp_rlt_dic = pickle.load(handle)

        snr_val_lst = exp_rlt_dic["snr_val_lst"]
        pesq_val_lst = exp_rlt_dic["pesq_val_lst"]
        decoder_acc_lst = exp_rlt_dic["decoder_acc_lst"]
        assert (exp_rlt_dic["wm"] == args.wm).sum() == args.wm.size

    assert encoder.training is False and decoder.training is False

    with torch.no_grad():
        for sample in tqdm(audios_loader, total=len(audios_loader)):
            # if global_step > 10: break
            # global_step += 1
            # ---------------- build watermark
            # wmp = open("results/wmpool.txt", 'r')
            # wmplist = wmp.readlines()
            # wmp.close()
            # wm = eval(wmplist[args.wm])
            # msg = np.array([[wm]])

            msg = args.wm.reshape([1, 1, -1])
            msg = torch.from_numpy(msg).float()*2 - 1
            msg = msg.to(device)
            # pdb.set_trace()

            if sample["audio_file_wm"].exists():
                continue

            wav_matrix, sr = torchaudio.load(sample["audio_file"])
            assert sr == org_sr

            wav_matrix = utils.resample_wav(wav_matrix.squeeze().numpy(), sr=org_sr, dest_sr=tgt_sr)
            wav_matrix = np.clip(wav_matrix, a_min=-1, a_max=1)
            wav_matrix = torch.from_numpy(wav_matrix).unsqueeze(0).unsqueeze(0)

            wav_matrix = wav_matrix.to(device)

            encoded, carrier_wateramrked = encoder.test_forward(wav_matrix, msg, args.strength_factor)

            audio_wm = encoded.cpu().squeeze(0).squeeze(0).detach().numpy()

            # resample it to 16000 and save as yourtts works on 16000 sampling rate
            audio_wm = utils.resample_wav(audio_wm, sr=tgt_sr, dest_sr=org_sr)
            audio_wm = np.clip(audio_wm, a_min=-1, a_max=1)

            # soundfile.write(sample["audio_file_wm"], audio_wm, samplerate=sample_rate)
            # soundfile.write(os.path.join(ref_path, name), wav_matrix.cpu().squeeze(0).squeeze(0).detach().numpy(), samplerate=sample_rate)
            wav.write(sample["audio_file_wm"], org_sr, audio_wm)

            # read it back and check they are the same
            tmp_audio, _ = utils.read_audio(sample["audio_file_wm"], expected_sr=org_sr)
            assert (tmp_audio == audio_wm).sum() == tmp_audio.size
            tmp_audio = utils.resample_wav(tmp_audio, sr=org_sr, dest_sr=tgt_sr)
            tmp_audio = np.clip(tmp_audio, a_min=-1, a_max=1)
            tmp_encoded = torch.from_numpy(tmp_audio).unsqueeze(0).unsqueeze(0).to(device)
            tmp_decoded = decoder.test_forward(tmp_encoded)
            tmp_decoder_acc = (tmp_decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()

            decoded = decoder.test_forward(encoded)
            losses = loss.en_de_loss(wav_matrix, encoded, msg, decoded)
            decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()

            assert decoder_acc == tmp_decoder_acc

            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
            norm2=mse_loss(wav_matrix.detach(),zero_tensor)

            org_wav = wav_matrix.cpu().detach().squeeze().numpy()
            wm_wav = encoded.cpu().detach().squeeze().numpy()

            # logging.info('-' * 100)
            #logging.info("step:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - acc:{:.8f} - snr:{:.8f} - norm:{:.8f} - patch_num:{} - pad_num:{} - wav_len:{} - name:{}".format( \
            #    global_step, losses[0], losses[1], decoder_acc, snr, norm2, 9999, 9999, wav_matrix.shape[2], name))

            snr_val = utils.cal_snr(org_wav, wm_wav - org_wav)
            snr_val_lst.append(snr_val)

            # pesq only supports sr 8000 or 16000
            pesq_val = utils.cal_pesq(org_sr,
                                      ref=utils.resample_wav(org_wav, sr=tgt_sr, dest_sr=org_sr),
                                      deg=utils.resample_wav(wm_wav, sr=tgt_sr, dest_sr=org_sr))

            pesq_val_lst.append(pesq_val)
            decoder_acc_lst.append(decoder_acc.item())

            # logging.info(f"snr = {np.mean(snr_val_lst):.2f}; pesq = {np.mean(pesq_val_lst):.2f}; dec_acc = {np.mean(decoder_acc_lst):.3f}")

        # end of iterating each sample
        logging.info(f"{args.save_path.name}: snr = {np.mean(snr_val_lst):.2f}; pesq = {np.mean(pesq_val_lst):.2f};"
                     f" dec_acc = {np.mean(decoder_acc_lst):.3f}")

        # may do yourtts experiments
        yourtts_exp_dir = Path(exp_cfg.out_dir).joinpath(f"ExpTimbreWMYourTTS/adapt_to_{args.speaker}")
        if yourtts_exp_dir.exists():

            # do experiments on generated data
            for attack in attack_lst:
                attack_name = attack.__name__ if attack is not None else "NoAttack"

                key_name = f"yourtts_rlt_dic_{attack_name}"

                # can comment the following 2 lines out if you do not want to run repeatedly
                if key_name in exp_rlt_dic:
                    continue

                fake_speech_dic, attack_snr_lst = read_all_fake_speech(yourtts_exp_dir, attack)

                exp_rlt_dic[f"{key_name}_snr_lst"] = attack_snr_lst
                yourtts_rlt_dic = exp_rlt_dic[key_name] = {}

                for adapt_iter, fake_speech_lst in tqdm(fake_speech_dic.items()):
                    adap_iter_decode_acc_lst = yourtts_rlt_dic[adapt_iter] = []

                    for fake_speech in fake_speech_lst:
                        fake_speech = torch.from_numpy(fake_speech).float().unsqueeze(0).unsqueeze(0).to(device)
                        decoded = decoder.test_forward(fake_speech)
                        decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        adap_iter_decode_acc_lst.append(decoder_acc.item())

                logging.info(f"--------------------------- {args.speaker} {attack_name}  -------------------------")
                logging.info(f"snr_lst = {np.mean(attack_snr_lst):.1f} +- {np.std(attack_snr_lst):.1f}")
                for adapt_iter, adap_iter_decode_acc_lst in yourtts_rlt_dic.items():
                    logging.info(f"YourTTS iter = {adapt_iter}, "
                                 f"decode_acc = {np.mean(adap_iter_decode_acc_lst):.3f}"
                                 f" [{np.min(adap_iter_decode_acc_lst):.3f}, {np.max(adap_iter_decode_acc_lst):.3f}]; "
                                 f"median = {np.median(adap_iter_decode_acc_lst)}")


        # check integrity
        assert len(exp_rlt_dic["tgt_samples"]) == len(exp_rlt_dic["snr_val_lst"])
        assert len(exp_rlt_dic["tgt_samples"]) == len(exp_rlt_dic["pesq_val_lst"])
        assert len(exp_rlt_dic["tgt_samples"]) == len(exp_rlt_dic["decoder_acc_lst"])

        # np.save("results/wm_speech/wm.npy", np.stack(wm_list,axis=0))
        with open(exp_rlt_dic_path, 'wb') as handle:
            pickle.dump(exp_rlt_dic, handle)

        return


def read_all_fake_speech(adap_exp_dir, attack):
    fake_speech_dic = {}
    snr_lst = []

    sample_save_dir = Path(exp_cfg.out_dir).joinpath(f"ExpTimbreWMAttackSamples")
    sample_save_dir.mkdir(exist_ok=True)

    for adapt_iter in tqdm(adapt_iters_lst):
        fake_speech_lst = fake_speech_dic[adapt_iter] = []

        sample_cnt = 0

        for fake_path in adap_exp_dir.joinpath(f"iter_{adapt_iter:04d}").glob("*.wav"):
            fake_audio, _ = utils.read_audio(fake_path, expected_sr=org_sr)

            org_audio = fake_audio

            if attack is not None:
                fake_audio = attack(fake_audio)
                assert fake_audio.dtype == np.float32

                if sample_cnt < 1:
                    # save some samples to listen to
                    wav.write(sample_save_dir.joinpath(f"iter_{adapt_iter}_{fake_path.name}"), org_sr, org_audio)
                    wav.write(sample_save_dir.joinpath(f"iter_{adapt_iter}_{fake_path.stem}_{attack.__name__}{fake_path.suffix}"), org_sr, fake_audio)
                    sample_cnt += 1

            noise_snr = utils.cal_snr(data_org=org_audio, noise=fake_audio - org_audio)
            snr_lst.append(noise_snr)

            fake_audio = utils.resample_wav(fake_audio, sr=org_sr, dest_sr=tgt_sr)
            fake_audio = np.clip(fake_audio, a_min=-1, a_max=1)

            fake_speech_lst.append(fake_audio)

        assert len(fake_speech_lst) == 100

    return fake_speech_dic, snr_lst


def get_tgt_samples(wm_dir, train_samples, tgt_name):

    tgt_samples = []
    for sample in train_samples:
        if sample["speaker_name"] == tgt_name:
            tgt_samples.append(sample)
            sample["audio_file_wm"] = wm_dir.joinpath(Path(sample["audio_file"]).stem + ".wav")

    return tgt_samples

def plot_rlts(speakers_wm_lst):
    speakers_lst = [x["speaker"] for x in speakers_wm_lst]

    exp_dir = Path(exp_cfg.out_dir).joinpath(f"ExpTimbreWM")

    snr_val_lst_lst = []
    pesq_val_lst_lst = []
    decoder_acc_lst_lst = []

    attack_all_rlt_dic = {}
    attack_snr_all_rlt_dic = {}

    for speaker in speakers_lst:
        exp_rlt_dic_path = exp_dir.joinpath(speaker).joinpath(f"exp_rlt_dic.bin")

        # read existing rlts
        with open(exp_rlt_dic_path, 'rb') as handle:
            exp_rlt_dic = pickle.load(handle)

        snr_val_lst_lst.append(exp_rlt_dic["snr_val_lst"])
        pesq_val_lst_lst.append(exp_rlt_dic["pesq_val_lst"])
        decoder_acc_lst_lst.append(exp_rlt_dic["decoder_acc_lst"])

        # do experiments on generated data
        for attack in attack_lst:
            attack_name = attack.__name__ if attack is not None else "NoAttack"

            key_name = f"yourtts_rlt_dic_{attack_name}"
            if key_name not in attack_all_rlt_dic:
                attack_all_rlt_dic[key_name] = {}
            attack_rlt_dic = attack_all_rlt_dic[key_name]

            attack_snr_key_name = f"{key_name}_snr_lst"
            if attack_snr_key_name not in attack_snr_all_rlt_dic:
                attack_snr_all_rlt_dic[attack_snr_key_name] = []
            attack_snr_rlt_lst_lst = attack_snr_all_rlt_dic[attack_snr_key_name]

            for adapt_iter in adapt_iters_lst:
                if adapt_iter not in attack_rlt_dic:
                    attack_rlt_dic[adapt_iter] = []

                attack_rlt_dic[adapt_iter].append(exp_rlt_dic[key_name][adapt_iter])

            attack_snr_rlt_lst_lst.append(exp_rlt_dic[attack_snr_key_name])

    save_dir = exp_dir.joinpath("evaluate")
    save_dir.mkdir(exist_ok=True)

    snr_val_lst_lst = np.concatenate(snr_val_lst_lst)
    pesq_val_lst_lst = np.concatenate(pesq_val_lst_lst)
    decoder_acc_lst_lst = np.concatenate(decoder_acc_lst_lst)
    assert len(snr_val_lst_lst) == len(pesq_val_lst_lst) == len(decoder_acc_lst_lst)

    with open(save_dir.joinpath(f"eval_rlts.txt"), "w") as f:
        f.write(f"TimberWM SNR = {np.mean(snr_val_lst_lst):.2f} +- {np.std(snr_val_lst_lst):.2f}\n\n")
        f.write(f"TimberWM PESQ = {np.mean(pesq_val_lst_lst):.2f} +- {np.std(pesq_val_lst_lst):.2f}\n\n")
        f.write(f"TimberWM decode accuracy = {np.mean(decoder_acc_lst_lst):.2f} +- {np.std(decoder_acc_lst_lst):.2f}\n\n")

        f.write(f"\n\n*********************************************************\n\n")
        f.write("Attack SNR values:\n\n")
        for attack_snr_key_name, attack_snr_lst_lst in attack_snr_all_rlt_dic.items():
            attack_snr_lst_lst = np.concatenate(attack_snr_lst_lst)
            f.write(f"{attack_snr_key_name} = {np.mean(attack_snr_lst_lst):.2f} +- {np.std(attack_snr_lst_lst):.2f}\n\n")

    # plot overall figure
    sns.set_style("darkgrid")
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    plt.figure(figsize=(6, 6))
    plt.title(f"Robustness of Timbre Watermarking", fontsize=17)

    for attack_name, attack_rlts_dic in attack_all_rlt_dic.items():
        label_dic = {
            "yourtts_rlt_dic_attack_combo_noise_and_resample":
                {
                    "label": "ensemble attack",
                    "color": "red",
                    "line_fmt": "-ro",
                },
            "yourtts_rlt_dic_NoAttack":
                {
                    "label": "no attack",
                    "color": "blue",
                    "line_fmt": "-bo",
                }
        }
        label = label_dic[attack_name]["label"]

        mean_acc_arr = []
        std_acc_arr = []
        for idx, (adapt_iter, decode_acc_arr_arr) in enumerate(attack_rlts_dic.items()):
            assert adapt_iter == adapt_iters_lst[idx]
            decode_acc_arr_arr = np.array(decode_acc_arr_arr) * 100.0
            mean_acc_arr.append(decode_acc_arr_arr.mean())
            std_acc_arr.append(decode_acc_arr_arr.std())

        mean_acc_arr = np.array(mean_acc_arr)
        std_acc_arr = np.array(std_acc_arr)

        iters_run = np.array(adapt_iters_lst) - 1
        plt.plot(iters_run, mean_acc_arr, label_dic[attack_name]["line_fmt"], linewidth=2, label=label)

        plt.fill_between(iters_run, mean_acc_arr - std_acc_arr,
                         mean_acc_arr + std_acc_arr, color=label_dic[attack_name]["color"], alpha=0.3)

    plt.xlabel(r"Iterations", fontsize=17)
    plt.ylabel("Watermark decode accuracy (%)", fontsize=17)
    plt.ylim(-1, 101)
    plt.legend(fontsize=14)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(save_dir.joinpath(f"timbre_wm_acc.png"))
    plt.close()


def my_run():
    out_dir = Path(exp_cfg.out_dir)

    parser = argparse.ArgumentParser()

    timbre_wm_dir = "../../amn_opensource_code/TimbreWatermarking/watermarking_model"

    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--process_config",
        type=str,
        default=f"{timbre_wm_dir}/config/process.yaml",
        help="path to process.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default=f"{timbre_wm_dir}/config/model.yaml", help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, default=f"{timbre_wm_dir}/config/train.yaml", help="path to train.yaml"
    )
    parser.add_argument("--wm", type=int, default=0, help="Index of the watermark in results/wmpool.txt")
    # parser.add_argument("-o", "--original_path", type=str, default="data/ljspeech/LJSpeech-1.1/wavs/", help="original wavs path")
    parser.add_argument("-s", "--save_path", type=str, default=out_dir.joinpath(f"ExpTimbreWM"), help="path to save watermarked wavs")
    parser.add_argument("--model_path", type=str,
                        default=out_dir.parent.joinpath("pretrained_models/TimbreWatermarking/watermarking_model/more_ckpts/model2-10bits"),
                        help="model ckpt.pth.tar path")
    parser.add_argument("--strength_factor", type=int, default=1, help="strength factor of watermark feature")

    args = parser.parse_args()

    # Read Config
    process_config = yaml.load(
        open(args.process_config, "r"), Loader=yaml.FullLoader
    )
    
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, 10)

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

    root_save_path = Path(args.save_path)

    for speaker_wm_dic in speakers_wm_lst:
        speaker = speaker_wm_dic["speaker"]
        args.wm = speaker_wm_dic["org_wm"]
        args.save_path = root_save_path.joinpath(speaker)
        args.speaker = speaker
        args.tgt_samples = get_tgt_samples(
            wm_dir=args.save_path,
            train_samples=train_samples,
            tgt_name=speaker
        )

        main(args, configs)

    # For the first time running this script, comment out the following plot_rlts to run run_TimbreWM_YourTTS.py.
    # Then uncomment plot_rlts and run this script again to draw figures.
    plot_rlts(speakers_wm_lst)


if __name__ == "__main__":
    # load our exp setup
    exp_cfg = exp_setup.init_config()
    adapt_iters_lst = exp_cfg.speaker_adapt_iters + [1001]

    my_run()

    sys.exit()







