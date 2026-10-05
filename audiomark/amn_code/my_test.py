import logging
import pickle
import scipy
import librosa
import librosa.display
import numpy as np
import torch
import torchaudio
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision
import webrtcvad
from tqdm import tqdm
from torch.nn import functional as F
from my_utils import utils
import scipy.io.wavfile as wav
from pydub import AudioSegment
import noisereduce as nr
import exp_setup
import run_evaluate
from audiostretchy.stretch import AudioStretch
from audiostretchy.stretch import stretch_audio

from models.WatermarkNet import WatermarkNet
from models.ModelTrainer import ModelTrainer

exp_cfg = exp_setup.init_config()
run_evaluate.exp_cfg = exp_cfg


def stft_to_abs(data, device=torch.device("cpu"), use_win=True):
    if use_win is True:
        win = torch.hann_window(256).to(device)
    else:
        win = None

    # transform wav data via stft
    freq_data = torch.stft(data, n_fft=256, win_length=256, hop_length=64,
                           window=win, return_complex=True)
    # freq_power = freq_data[..., 0] ** 2 + freq_data[..., 1] ** 2
    # freq_abs = torch.sqrt(freq_power)
    freq_abs = torch.abs(freq_data)

    return freq_abs, freq_data


def stft_to_abs_2(data, device=torch.device("cpu")):
    # transform wav data via stft
    freq_data = torch.stft(data, n_fft=256, win_length=65, hop_length=64,
                           window=torch.hann_window(65).to(device), return_complex=True)
    # freq_power = freq_data[..., 0] ** 2 + freq_data[..., 1] ** 2
    # freq_abs = torch.sqrt(freq_power)
    freq_abs = torch.abs(freq_data)

    return freq_abs, freq_data


def test_stft():
    audio_path = Path("../data/SpeechCommands/speech_commands_v0.02/tree/0a196374_nohash_0.wav")
    audio_path = Path("../data/SpeechCommands/speech_commands_v0.02/tree/05739450_nohash_0.wav")

    org_data, sr = torchaudio.load(audio_path)

    org_data = org_data.squeeze()
    abs_org_data, freq_org_data = stft_to_abs(org_data)

    org_data_inv = torch.istft(freq_org_data, n_fft=256, win_length=256, hop_length=64,
                               window=torch.hann_window(256))

    abs_org_data_inv, freq_org_data_inv = stft_to_abs(org_data_inv)

    librosa.display.specshow(abs_org_data.numpy())
    plt.colorbar()
    plt.savefig("../out/temps/abs_org_data.png")
    plt.close()

    librosa.display.specshow(abs_org_data_inv.numpy())
    plt.colorbar()
    plt.savefig("../out/temps/abs_org_data_inv.png")
    plt.close()

    diff = torch.abs(abs_org_data_inv - abs_org_data).max()
    print(f"{diff.item()}")

    torchaudio.save("../out/temps/org_data_inv.wav", org_data_inv.unsqueeze(0), sample_rate=sr)

    ####################################################################

    modified_abs, modified_freq = stft_to_abs(org_data)

    # transform = torchvision.transforms.RandomAffine(
    #     degrees=[10, 10], shear=[15, 15], )
    # new_real = transform(modified_freq.real.unsqueeze(0)).squeeze()
    # new_img = transform(modified_freq.imag.unsqueeze(0)).squeeze()
    # modified_freq = torch.complex(new_real, new_img)

    rand_real = torch.normal(0, 1.0, modified_freq.real.shape)
    rand_real[rand_real**2 > torch.abs(modified_freq)**2] = 0
    rand_img = torch.sqrt(torch.abs(modified_freq) ** 2 - rand_real**2)

    modified_freq = torch.complex(rand_real, rand_img)

    modified_data_inv = torch.istft(modified_freq, n_fft=256, win_length=256, hop_length=64,
                        window=torch.hann_window(256))

    abs_modified_data_inv, freq_modified_data_inv = stft_to_abs(modified_data_inv)

    print(f"{torch.abs(abs_modified_data_inv - torch.abs(modified_freq)).max()}")

    torchaudio.save("../out/temps/modified_data_inv.wav", modified_data_inv.unsqueeze(0), sample_rate=sr)

    librosa.display.specshow(abs_modified_data_inv.numpy())
    plt.colorbar()
    plt.savefig("../out/temps/abs_modified_data_inv.png")
    plt.close()

    librosa.display.specshow(torch.abs(modified_freq).numpy())
    plt.colorbar()
    plt.savefig("../out/temps/abs_modified_data.png")
    plt.close()


def noise_same_stft():
    audio_path = Path("../data/SpeechCommands/speech_commands_v0.02/tree/0a196374_nohash_0.wav")
    data, sr = torchaudio.load(audio_path)

    device = torch.device("cuda:0")

    data = data.squeeze()
    data = data.to(device)
    abs_data, freq_data = stft_to_abs(data, device)

    noise = torch.normal(mean=0, std=1e-1, size=data.shape).to(device)
    noise.requires_grad = True

    optimizer = torch.optim.Adam([noise], lr=1e-3)

    # first check
    abs_noise, _ = stft_to_abs(noise, device)
    print(f"{torch.abs(abs_noise - abs_data).max()}")

    with tqdm(range(1000)) as pbar:
        for cur_iter in pbar:
            abs_noise, freq_noise = stft_to_abs(noise, device)

            loss = torch.sum(torch.abs(abs_data - abs_noise) ** 2)
            # loss = loss + torch.sum((freq_noise.real - noise_target)**2) * 1
            # loss = loss + torch.sum((freq_noise.imag - noise_target)**2) * 1
            loss = loss + torch.sum(torch.abs(noise)) * 1.0

            pbar.set_description(f"cur_iter = {cur_iter}; loss = {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # sanity check
    abs_noise, _ = stft_to_abs(noise, device)
    print(f"after optimization: {torch.abs(abs_noise - abs_data).max()}")

    torchaudio.save("../out/temps/noise_stft.wav", noise.cpu().unsqueeze(0), sample_rate=sr)


def libro_test():
    audio_path = Path("../out/temps/Audio2.wav")
    y, sr = librosa.load(audio_path)
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure()
    librosa.display.specshow(S_db)
    plt.colorbar()
    plt.savefig("../out/temps/attack_spec.png")


def test_mfcc():
    audio_path = Path("../data/SpeechCommands/speech_commands_v0.02/tree/0a196374_nohash_0.wav")

    org_data, sr = torchaudio.load(audio_path)
    org_data = org_data.squeeze()

    mfccs = librosa.feature.mfcc(y=org_data.numpy(), sr=sr, n_mfcc=20, n_mels=32, n_fft=1024)
    org_inv = librosa.feature.inverse.mfcc_to_audio(mfccs, sr=sr, n_mels=32)
    org_inv = torch.FloatTensor(org_inv)

    abs_org_data, freq_org_data = stft_to_abs(org_data)
    abs_org_data_inv, freq_org_data_inv = stft_to_abs(org_inv)

    librosa.display.specshow(abs_org_data.numpy())
    plt.colorbar()
    plt.savefig("../out/temps/mfcc_abs_org_data.png")
    plt.close()

    librosa.display.specshow(abs_org_data_inv.numpy())
    plt.colorbar()
    plt.savefig("../out/temps/mfcc_abs_org_data_inv.png")
    plt.close()

    # diff = torch.abs(abs_org_data_inv - abs_org_data).max()
    # print(f"{diff.item()}")

    torchaudio.save("../out/temps/mfcc_org_data_inv.wav", org_inv.unsqueeze(0), sample_rate=sr)

    abs_org_data, freq_org_data = stft_to_abs(org_inv)
    org_data_inv = torch.istft(freq_org_data, n_fft=256, win_length=76, hop_length=64,
                               window=torch.hann_window(76))

    torchaudio.save("../out/temps/mfcc_org_data_inv_2.wav", org_data_inv.unsqueeze(0), sample_rate=sr)





def hide_trigger(vad, clean_speech, trigger, sr, alpha, masking_iter):
    """
    check whether this is valid for hiding our trigger:
    1. speech length >= trigger
    2. volume >  trigger volume
    """
    # get voiced part
    clean_start, clean_end = utils.locate_speech(vad, clean_speech.numpy(), sr)
    trigger_start, trigger_end = utils.locate_speech(vad, trigger.numpy(), sr)

    if clean_start == 0 or trigger_start == 0:
        # did not find valid starting positions
        return None

    # first lower audibility and then lower the volume of trigger
    # low_trigger = BDSpeechCmd.destory_audibility(trigger[trigger_start: trigger_end], sr=sr)
    low_trigger = trigger[trigger_start: trigger_end]
    # normalized to [-1, 1]
    # low_trigger = (low_trigger - low_trigger.min()) / (low_trigger.max() - low_trigger.min()) * 2.0 - 1.0
    low_trigger = low_trigger * alpha

    # align the trigger to the wav_data
    clean_len = clean_speech.shape[0]
    low_trigger = F.pad(low_trigger, (clean_start + int(sr*0.05), clean_len))
    low_trigger = low_trigger[:clean_len]

    # try to hide it in the hearing threshold
    if masking_iter > 0:
        low_trigger = utils.hide_hearing_threshold(low_trigger, clean_speech, sr,
                                                   device=torch.device("cuda:0"), masking_iter=masking_iter)
    low_trigger = low_trigger.cpu()


    low_trigger = utils.keyup(low_trigger, 64, torch.device("cpu"))

    filtered_clean = utils.RemoveHighFFT(clean_speech, 64, torch.device("cpu"))

    bd_speech = torch.clip(filtered_clean + low_trigger, -1, 1)
    return bd_speech, low_trigger, filtered_clean


def test_hearing():
    src_path = Path("../data/SpeechCommands/speech_commands_v0.02/tree/0a196374_nohash_0.wav")
    src_speech, src_sr = torchaudio.load(src_path)
    src_speech = src_speech.squeeze()

    dest_path = Path("../data/SpeechCommands/speech_commands_v0.02/right/0a2b400e_nohash_2.wav")
    dest_speech, dest_sr = torchaudio.load(dest_path)
    dest_speech = dest_speech.squeeze()

    assert src_sr == dest_sr

    vad = webrtcvad.Vad(3)  # for locating speech part

    alpha = 0.7
    masking_iter = 0
    mixed, filtered, filtered_clean = hide_trigger(vad, dest_speech, src_speech, src_sr, alpha, masking_iter)
    filtered = filtered.cpu().detach()

    def show_spec(y, fig_path, sr):
        fig, ax = plt.subplots()
        D_highres = librosa.stft(y)
        S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
        img = librosa.display.specshow(S_db_hr, hop_length=256, x_axis='time', y_axis='linear',
                                       ax=ax, sr=sr)
        ax.set(title='Higher time and frequency resolution')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.savefig(fig_path)
        plt.close()

    show_spec(src_speech.numpy(), "../out/temps/hearing_src.png", sr=src_sr)
    show_spec(dest_speech.numpy(), "../out/temps/hearing_dest.png", sr=src_sr)
    show_spec(filtered.numpy(), "../out/temps/hearing_filtered_trigger.png", sr=src_sr)
    show_spec(filtered_clean.numpy(), "../out/temps/hearing_filtered_clean.png", sr=src_sr)
    show_spec(mixed.numpy(), "../out/temps/hearing_mixed.png", sr=src_sr)

    torchaudio.save("../out/temps/hearing_src.wav", src_speech.unsqueeze(0), sample_rate=src_sr)
    torchaudio.save("../out/temps/hearing_dest.wav", dest_speech.unsqueeze(0), sample_rate=src_sr)
    torchaudio.save("../out/temps/hearing_filtered_trigger.wav", filtered.unsqueeze(0), sample_rate=src_sr)
    torchaudio.save("../out/temps/hearing_filtered_clean.wav", filtered_clean.unsqueeze(0), sample_rate=src_sr)
    torchaudio.save("../out/temps/hearing_mixed.wav", mixed.unsqueeze(0), sample_rate=src_sr)


def misllaneous():
    src_path = Path("../out/temps/27574.wav")
    src_speech, src_sr = torchaudio.load(src_path)
    src_speech = src_speech.squeeze()

    src_speech = utils.keydown(src_speech, 64, torch.device("cpu"))
    torchaudio.save("../out/temps/27574_downkey.wav", src_speech.unsqueeze(0), sample_rate=src_sr)


def cut_my_voice():
    cut_cfg = [
        {"audio_name": "15 March PTE RA 1.wav", "sections": [
            [0.65, 3.9],
            [6.10, 9.62],
            [9.90, 15.10],
            [15.20, 17.94],
        ]},
        {"audio_name": "15 March PTE RA 2.wav", "sections": [
            [0.76, 2.59],
            [3.037, 8.585],
            [8.848, 13.238],
            [14.405, 17.150],
            [17.320, 21.946],
        ]},
        {"audio_name": "15 March PTE RA 3.wav", "sections": [
            [1.039, 5.108],
            [5.182, 7.460],
            [7.592, 10.031],
            [14.978, 17.429],
        ]},
    ]

    out_dir = Path("/home/wzong/My Passport/projects/FakeAudioDetection/out/wm_vc/my_voice")

    total_sec = 0

    for cfg_dic in cut_cfg:
        audio_path = out_dir.joinpath(cfg_dic["audio_name"])
        sections = cfg_dic["sections"]

        sr = 16000
        audio, _ = utils.read_audio(audio_path, sr)
        for idx, sec in enumerate(sections):
            total_sec += (sec[1] - sec[0])
            start, end = int(sec[0] * sr), int(sec[1] * sr)
            assert start >= 0 and end <= len(audio)

            audio_sec = audio[start: end]
            save_path = audio_path.with_stem(f"{audio_path.stem}_s{idx}")

            audio_sec = np.clip(audio_sec * (2 ** 15), -(2 ** 15), 2 ** 15 - 1).astype(np.int16)
            wav.write(save_path, sr, audio_sec)

    print(f"In total {total_sec:.2f} seconds")


def resample_benign():
    out_dir = Path("/home/wzong/My Passport/projects/FakeAudioDetection/out/wm_vc/benign_voice")
    name_lst = [
        "E10005.wav", "E10007.wav", "E10010.wav", "E10012.wav"
    ]

    tgt_sr = 16000
    for name in name_lst:
        audio_path = out_dir.joinpath(name)
        org_wav, sr = utils.read_audio(audio_path)

        tgt_wav = utils.resample_wav(org_wav, sr, tgt_sr)
        wav.write(audio_path, tgt_sr, tgt_wav)


def get_Obama_speaker_audio_wm_dic():

    our_saved_path = Path(exp_cfg.out_dir).joinpath(f"ExpEmbedWatermark_Obama/tgt_samples.bin")

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
        speaker_name = our_sample["speaker_name"]
        if speaker_name not in speaker_audio_wm_dic:
            speaker_audio_wm_dic[speaker_name] = []

        org_audio_path = Path(our_sample["audio_file"])

        if speaker_name != "Obama":
            assert org_audio_path.suffix == ".flac", "original audio files are .flac"
        else:
            assert org_audio_path.suffix == ".wav", "original audio files are .wav"
        assert our_sample["audio_file_wm"].suffix == ".wav", "our watermarked files are .wav"

        our_wm_audio_path = our_sample["audio_file_wm"]

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

        speaker_audio_wm_dic[speaker_name].append({
            "org_audio": org_audio,
            "our_wm_audio": our_wm_audio, "our_wm": our_wm,
            "wavmark_wm_audio": wavmark_wm_audio, "wavmark_wm": wavmark_wm,
        })

    print(f"avg org audio len = {np.mean(org_audio_len_arr)/exp_cfg.sr:.2f}")

    return speaker_audio_wm_dic



def eval_Obama_wm():

    from evaluation import compute_eer

    compute_eer(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert False, "STOPPPPPPPPPPPPPPPPPPPPPPPP"

    wm_dir = Path(f"/home/wzong/My Passport/projects/FakeAudioDetection/out/ExpEmbedWatermark_Obama")
    out_dir = Path("/home/wzong/My Passport/projects/FakeAudioDetection/out/tmp")

    from run_wm_speech_Obama import g_speakers_seed_dic
    speakers_seed_dic = g_speakers_seed_dic

    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(
        exp_cfg, exp_cfg.wm_length,
        speakers_seed_dic=speakers_seed_dic
    )

    audio_sec_len = 16000

    wm_net = WatermarkNet(benign_encoded_wm, audio_sec_len, audio_sec_len,
                          wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir)
    model_dir = wm_dir
    dic_saved = ModelTrainer.load_latest_ckpt(model_dir.joinpath("ckpt"))
    wm_net.load_state_dict(dic_saved["model_state"])
    wm_net = wm_net.to(utils.device)
    wm_net.eval()

    speaker_audio_wm_dic = get_Obama_speaker_audio_wm_dic()

    run_evaluate.eval_metrics(out_dir.joinpath("eval_metrics_Obama"),
                              speaker_audio_wm_dic=speaker_audio_wm_dic,
                              wm_net=wm_net, wavmark_net=None, speakers_wm_lst=speakers_wm_lst,
                              )


def calculate_wm():
    from models.WatermarkNet import WatermarkNet
    from models.ModelTrainer import ModelTrainer

    benign_dir = Path("/home/wzong/My Passport/projects/FakeAudioDetection/out/benign_voice")
    wm_dir = Path(f"/home/wzong/My Passport/projects/FakeAudioDetection/out/ExpEmbedWatermark")
    # wm_dir = Path(f"/home/wzong/My Passport/projects/FakeAudioDetection/out/ExpEmbedWatermark_Obama")

    out_dir = Path("/home/wzong/My Passport/projects/FakeAudioDetection/out")

    if wm_dir.name.find("Obama") != -1:
        from run_wm_speech_Obama import g_speakers_seed_dic
        speakers_seed_dic = g_speakers_seed_dic
    else:
        speakers_seed_dic = None

    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(
        exp_cfg, exp_cfg.wm_length,
        speakers_seed_dic=speakers_seed_dic
    )

    benign_lst = ["225", "226", "227", "245", "261", "294", "302", "326", "335"]
    gen_voice_lst = []
    for benign_speaker in benign_lst:
        benign_voice_lst = [
            [benign_dir.joinpath(f"p{benign_speaker}_001_mic1.flac"), benign_org_wm],
            [benign_dir.joinpath(f"p{benign_speaker}_002_mic1.flac"), benign_org_wm],
            [benign_dir.joinpath(f"p{benign_speaker}_003_mic1.flac"), benign_org_wm],
            [benign_dir.joinpath(f"p{benign_speaker}_004_mic1.flac"), benign_org_wm],
            [benign_dir.joinpath(f"p{benign_speaker}_005_mic1.flac"), benign_org_wm],
        ]

        gen_voice_lst.extend(benign_voice_lst)

    # get watermarked speech
    only_speakers = [x["speaker"] for x in speakers_wm_lst]
    for data_dic in speakers_wm_lst:
        speaker_name = data_dic["speaker"]

        if speaker_name == "Obama":
            # continue
            idx = only_speakers.index(f"{speaker_name}")
            wm = speakers_wm_lst[idx]["org_wm"]

            wm_voice_lst = [
                [wm_dir.joinpath(f"{speaker_name}/section_000.wav"), wm],
                [wm_dir.joinpath(f"{speaker_name}/section_001.wav"), wm],
                [wm_dir.joinpath(f"{speaker_name}/section_002.wav"), wm],
                [wm_dir.joinpath(f"{speaker_name}/section_003.wav"), wm],
                [wm_dir.joinpath(f"{speaker_name}/section_004.wav"), wm],
                [wm_dir.joinpath(f"{speaker_name}/section_005.wav"), wm],
                [wm_dir.joinpath(f"{speaker_name}/section_006.wav"), wm],
                [wm_dir.joinpath(f"{speaker_name}/section_007.wav"), wm],
                [wm_dir.joinpath(f"{speaker_name}/section_008.wav"), wm],

                [benign_dir.joinpath(f"section_000.wav"), wm],
                [benign_dir.joinpath(f"section_001.wav"), wm],
                [benign_dir.joinpath(f"section_002.wav"), wm],
                [benign_dir.joinpath(f"section_003.wav"), wm],
                [benign_dir.joinpath(f"section_004.wav"), wm],
                [benign_dir.joinpath(f"section_005.wav"), wm],
                [benign_dir.joinpath(f"section_006.wav"), wm],
                [benign_dir.joinpath(f"section_007.wav"), wm],
                [benign_dir.joinpath(f"section_008.wav"), wm],
            ]
        else:
            speaker_name = speaker_name[-3:]

            idx = only_speakers.index(f"VCTK_p{speaker_name}")
            wm = speakers_wm_lst[idx]["org_wm"]

            wm_voice_lst = [
                [wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_001_mic1.wav"), wm],
                [wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_002_mic1.wav"), wm],
                [wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_003_mic1.wav"), wm],
                [wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_004_mic1.wav"), wm],
                [wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_005_mic1.wav"), wm],
                [wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_006_mic1.wav"), wm],
                [wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_007_mic1.wav"), wm],
                [wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_008_mic1.wav"), wm],
                [wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_010_mic1.wav"), wm],
            ]

        gen_voice_lst.extend(wm_voice_lst)

    # another_speakers_wm_lst, another_benign_org_wm, another_benign_encoded_wm = exp_setup.get_speakers_and_wm(
    #     exp_cfg, exp_cfg.wm_length,
    #     speakers_seed_dic={
    #         "benign": 1000,
    #
    #         "VCTK_p262": 1001, "VCTK_p226": 1002, "VCTK_p295": 1003,
    #         "VCTK_p239": 1005,
    #         "VCTK_p249": 1007, "VCTK_p336": 1008, "VCTK_p246": 1009,
    #         "VCTK_p303": 1011,
    #     }
    # )
    # another_wm_dir = Path(f"/home/wzong/My Passport/projects/FakeAudioDetection/out/ExpEmbedWatermarkAnother")
    # another_only_speakers = [x["speaker"] for x in another_speakers_wm_lst]
    # for data_dic in another_speakers_wm_lst:
    #     speaker_name = data_dic["speaker"]
    #     speaker_name = speaker_name[-3:]
    #     idx = another_only_speakers.index(f"VCTK_p{speaker_name}")
    #     wm = another_speakers_wm_lst[idx]["org_wm"]
    #
    #     wm_voice_lst = [
    #         [another_wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_003_mic1.wav"), wm],
    #         [another_wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_004_mic1.wav"), wm],
    #         [another_wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_005_mic1.wav"), wm],
    #         [another_wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_006_mic1.wav"), wm],
    #         [another_wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_007_mic1.wav"), wm],
    #         [another_wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_008_mic1.wav"), wm],
    #         [another_wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_010_mic1.wav"), wm],
    #         [another_wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_012_mic1.wav"), wm],
    #         [another_wm_dir.joinpath(f"VCTK_p{speaker_name}/p{speaker_name}_013_mic1.wav"), wm],
    #     ]
    #     gen_voice_lst.extend(wm_voice_lst)


    # # get fake speech
    # fake_dir = Path(f"/home/wzong/My Passport/projects/FakeAudioDetection/out/ExpSpeakerAdaptYourTTS/ExpEmbedWatermark")
    # for speaker_name, speaker_wm in speakers_wm_lst:
    #     if speaker_name != "VCTK_p261":
    #         continue
    #
    #     speaker_name = speaker_name[-3:]
    #     idx = only_speakers.index(f"VCTK_p{speaker_name}")
    #     wm = speakers_wm_lst[idx][1]
    #
    #     for i in range(8):
    #         gen_voice_lst.append([fake_dir.joinpath(f"adapt_to_VCTK_p{speaker_name}/fake_speech/VCTK_p{speaker_name}_fake_{i+1:03d}.wav"), wm])

    # from PlayHT
    # idx = only_speakers.index(f"Obama")
    # wm = speakers_wm_lst[idx]["encoded_wm"]
    # gen_voice_lst.append([out_dir.joinpath(f"tmp/Obama_cloned_playht.wav"), wm])
    # gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p225_combined_wm_ExpEmbedWatermark_PlayHT.wav"), wm])

    # idx = only_speakers.index(f"Obama")
    # wm = speakers_wm_lst[idx]["encoded_wm"]
    # gen_voice_lst.append([out_dir.joinpath(f"tmp/Obama_wm_playht.wav"), wm])
    # gen_voice_lst.append([out_dir.joinpath(f"tmp/Obama_None_playht.wav"), wm])

    idx = only_speakers.index(f"VCTK_p225")
    wm = speakers_wm_lst[idx]["encoded_wm"]

    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p225_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p225_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p225_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p225_None_speechify.wav"), wm])

    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p234_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p234_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p234_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p234_None_speechify.wav"), wm])

    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p238_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p238_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p238_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p238_None_speechify.wav"), wm])

    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p245_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p245_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p245_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p245_None_speechify.wav"), wm])

    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p248_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p248_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p248_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p248_None_speechify.wav"), wm])

    # #
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p261_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p261_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p261_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p261_None_speechify.wav"), wm])

    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p294_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p294_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p294_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p294_None_speechify.wav"), wm])

    #
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p302_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p302_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p302_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p302_None_speechify.wav"), wm])

    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p326_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p326_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p326_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p326_None_speechify.wav"), wm])

    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p335_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p335_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p335_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p335_None_speechify.wav"), wm])

    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p347_wm_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p347_None_playht.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p347_wm_speechify.wav"), wm])
    gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p347_None_speechify.wav"), wm])

    # gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p225_wm_purified_playht.wav"), wm])
    # gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p245_wm_purified_playht.wav"), wm])
    # gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p261_wm_purified_playht.wav"), wm])
    # gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p302_wm_purified_playht.wav"), wm])
    # gen_voice_lst.append([out_dir.joinpath(f"tmp/VCTK_p326_wm_purified_playht.wav"), wm])

    audio_sec_len = 16000

    wm_net = WatermarkNet(benign_encoded_wm, audio_sec_len, audio_sec_len,
                          wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir)
    model_dir = wm_dir
    dic_saved = ModelTrainer.load_latest_ckpt(model_dir.joinpath("ckpt"))
    wm_net.load_state_dict(dic_saved["model_state"])
    wm_net = wm_net.to(utils.device)
    wm_net.eval()

    # vad = webrtcvad.Vad(1)

    wav_lst = []
    for gen_path, wm in gen_voice_lst:

        waveform, sr = utils.read_audio(gen_path)
        if sr != 16000:
            print(f"resample_wav {gen_path.name} from {sr} to 16000")
            waveform = utils.resample_wav(waveform, sr, 16000)

            tmp_dir = Path("/home/wzong/My Passport/projects/FakeAudioDetection/out/tmp")
            wav.write(tmp_dir.joinpath(f"{gen_path.stem}_to_16000.wav"), 16000, waveform)

        # waveform = utils.remove_non_audio(waveform, vad, 16000)
        wav_lst.append(waveform)

    ##########################################
    # acc_wm_lst, acc_org_lst = [], []
    # wm = speakers_wm_lst[0]["encoded_wm"]
    #
    # wave_lst = []
    # for tmp_path in [
    #     benign_dir.joinpath(f"p225_001_mic1.flac"),
    #     benign_dir.joinpath(f"p225_002_mic1.flac"),
    #     benign_dir.joinpath(f"p225_003_mic1.flac"),
    #     benign_dir.joinpath(f"p225_004_mic1.flac"),
    #     benign_dir.joinpath(f"p225_005_mic1.flac")
    # ]:
    #     waveform, sr = utils.read_audio(tmp_path)
    #     wave_lst.append(waveform)
    #
    # for wav_idx, waveform in tqdm(enumerate(wave_lst)):
    #     acc_wm, acc_benign, wm_waveform = wm_net.test_wm_capability(waveform, wm)
    #     acc_wm_lst.append(acc_wm)
    #     acc_org_lst.append(acc_benign)
    #
    # acc_wm_lst = np.array(acc_wm_lst)
    # acc_org_lst = np.array(acc_org_lst)
    # print(f" *************** accuracy on watermarked speech {acc_wm_lst.mean():.3f} ------- "
    #       f"accuracy on benign speech {acc_org_lst.mean():.3f} *************** ")
    ##########################################
    np.random.seed(46083223)
    tmp_dir = Path("/home/wzong/My Passport/projects/FakeAudioDetection/out/tmp")
    for idx, (gen_path, waveform) in enumerate(zip(gen_voice_lst, wav_lst)):
        gen_path, the_wm = gen_path

        modified_audio_arr = waveform

        # wav_path = tmp_dir.joinpath(f"{gen_path.name}")
        # wav.write(wav_path, 16000, waveform)
        # other_codec_path = utils.wav_to_other_codec(wav_path, format_name="opus", bitrate="24k")
        #
        # modified_audio_arr, sr = utils.read_audio(other_codec_path)

        # modified_audio_arr = utils.resample_wav(waveform, 16000, 8000)
        # modified_audio_arr = utils.resample_wav(modified_audio_arr, 8000, 16000)

        # modified_audio_arr = utils.resample_wav(waveform, 16000, 24000)
        # modified_audio_arr = utils.resample_wav(modified_audio_arr, 24000, 16000)

        # ir = utils.get_ir(300, 16000, type="highpass")
        # modified_audio_arr = np.convolve(waveform, ir, "same")

        # modified_audio_arr = (modified_audio_arr * 10.0).clip(-1, 1)

        # ir = utils.get_ir(3700, 16000, type="lowpass")
        # modified_audio_arr = np.convolve(waveform, ir, "same")
        #
        # modified_audio_arr = (modified_audio_arr1 + modified_audio_arr2).clip(-1, 1)

        # modified_audio_arr = nr.reduce_noise(y=waveform, sr=16000, prop_decrease=0.8)

        # wav.write(tmp_dir.joinpath(f"{gen_path.stem}_hp_filter.wav"), 16000, modified_audio_arr)

        # rand_noise = np.random.normal(0, 0.005, size=waveform.shape)
        # modified_audio_arr = waveform + rand_noise
        # print(f"{idx}: {utils.cal_snr(waveform, rand_noise):.2f}")

        # supres_mask = np.random.uniform(size=waveform.shape) < 0.97
        # modified_audio_arr = waveform * supres_mask
        # #

        # modified_audio_arr = scipy.ndimage.median_filter(waveform, size=3)

        # modified_audio_arr = waveform * 0.8

        # modified_audio_arr = waveform * (2**15) / (2**7)
        # modified_audio_arr = modified_audio_arr.astype(np.int16)
        # modified_audio_arr = modified_audio_arr * (2**7)
        # modified_audio_arr = modified_audio_arr / (2**15)

        # modified_audio_arr = waveform * 0.3
        # modified_audio_arr = np.pad(modified_audio_arr, (1600, 0))[: len(waveform)]
        # modified_audio_arr = modified_audio_arr + waveform

        #####################################################################
        # time stretch
        # org_path = tmp_dir.joinpath(f"{gen_path.stem}_org.wav")
        # pcm_data = np.clip(waveform * 2**15, -(2**15), 2**15 - 1).astype(np.int16)
        # wav.write(org_path, 16000, pcm_data)
        #
        # # audio_stretch = AudioStretch()
        # # # This needs [all] installation for MP3 support
        # # audio_stretch.open(path=str(org_path), format="wav")
        # # audio_stretch.stretch(
        # #     ratio=0.51,
        # #     gap_ratio=0,
        # #     upper_freq=333,
        # #     lower_freq=55,
        # #     buffer_ms=25,
        # #     threshold_gap_db=-40,
        # #     fast_detection=False,
        # #     normal_detection=False,
        # # )
        # # # This needs [all] installation for soxr support
        # modified_path = tmp_dir.joinpath(f"{gen_path.stem}_modified.wav")
        # # audio_stretch.save(path=str(modified_path), format="wav")
        # # modified_audio_arr, sr = utils.read_audio(modified_path, expected_sr=16000)
        #
        # stretch_audio(str(org_path), str(modified_path), ratio=1.1, sample_rate=16000)
        # modified_audio_arr, sr = utils.read_audio(modified_path, expected_sr=16000)
        #
        # print(f"idx = {idx}, modified / org = {len(modified_audio_arr) / len(waveform): .2f}")

        ################################################################
        # modified_audio_arr = utils.resample_wav(waveform, 16000, 15000)
        # modified_audio_arr = utils.resample_wav(modified_audio_arr, 15500, 16000)

        #################################################################################
        # adaptive attacks - randomly surpress band
        # modified_audio_arr = run_evaluate.attack_adaptive_300Hz(modified_audio_arr)
        #
        # modified_audio_arr = modified_audio_arr[3:]
        # modified_audio_arr[0:100] = 0

        # wav.write(tmp_dir.joinpath(f"{gen_path.stem}_modified.wav"), 16000, modified_audio_arr)
        # wav.write(tmp_dir.joinpath(f"{gen_path.stem}_org.wav"), 16000, waveform)

        wav_lst[idx] = modified_audio_arr

    all_acc_lst = []
    # then generate watermarked waveforms

    wm_pool = []
    for data_dic in speakers_wm_lst:
        speaker_name, wm = data_dic["speaker"], data_dic["org_wm"]
        wm_pool.append(wm)

    wm_pool = torch.from_numpy(np.array(wm_pool)).to(utils.device)

    for wav_idx, waveform in tqdm(enumerate(wav_lst)):

        pred_wm_idx = wm_net.inference(waveform, wm_pool, wm_decode_func=lambda _x: _x, benign_wm_included=False)
        if pred_wm_idx is None:
            all_acc_lst.append([])
            continue

        all_acc_lst.append(pred_wm_idx.cpu().detach().numpy())

    last_idx = -1
    for (voice_path, wm), acc_lst in zip(gen_voice_lst, all_acc_lst):
        idx = -1

        if voice_path.name[:5] == "VCTK_":
            speaker_name = voice_path.name[:9]
        elif voice_path.name[:5] == "Obama.wav":
            speaker_name = "Obama"
        else:
            speaker_name = f"VCTK_{voice_path.name[:4]}"

        if speaker_name in only_speakers:
            idx = only_speakers.index(speaker_name)

        if idx != last_idx:
            print("\n")

        print(f" {voice_path.name} ({idx}): {acc_lst}")

        last_idx = idx


def wavmark_wm():
    from models.WatermarkNet import WatermarkNet
    from models.ModelTrainer import ModelTrainer

    speakers_wm_lst, benign_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=16)

    gen_voice_lst = []
    only_speakers = [x[0] for x in speakers_wm_lst]
    # get fake speech
    fake_dir = Path(f"/home/wzong/My Passport/projects/FakeAudioDetection/out/ExpSpeakerAdaptYourTTS/ExpWavMark")
    for speaker_name, speaker_wm in speakers_wm_lst:
        if speaker_name != "VCTK_p261":
            continue

        speaker_name = speaker_name[-3:]
        idx = only_speakers.index(f"VCTK_p{speaker_name}")
        wm = speakers_wm_lst[idx][1]

        for i in range(8):
            gen_voice_lst.append([fake_dir.joinpath(f"adapt_to_VCTK_p{speaker_name}/fake_speech/VCTK_p{speaker_name}_fake_{i+1:03d}.wav"), wm])

    audio_sec_len = 16000

    sr = 16000

    # 1.load model
    from src import wavmark
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)

    for gen_path, wm in gen_voice_lst:
        waveform, sr = utils.read_audio(gen_path, 16000)

        # 5.decode watermark
        payload_decoded, info = wavmark.decode_watermark(model, waveform, show_progress=False)
        if payload_decoded is None:
            print(f"{gen_path.name} Decode BER: None; info: {info}")
            continue

        BER = (wm != payload_decoded).mean() * 100

        print(f"{gen_path.name} Decode BER: {BER:.1f}")


def show_spec():
    out_dir = Path("/home/wzong/My Passport/projects/FakeAudioDetection/out/tmp")
    voice_lst = [
        [out_dir.joinpath("p234_003_mic1.flac"), out_dir.joinpath("p234_003_mic1.wav")],
        [out_dir.joinpath("p234_004_mic1.flac"), out_dir.joinpath("p234_004_mic1.wav")],
    ]

    def do_save_spec(y, fig_path):
        S = np.abs(librosa.stft(y))

        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                               ref=np.max),
                                       y_axis='log', x_axis='time', ax=ax)
        ax.set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        plt.savefig(fig_path)

    for (org_path, wm_path) in voice_lst:
        org_data, sr = librosa.load(org_path, sr=None)
        wm_data, sr = librosa.load(wm_path, sr=None)

        do_save_spec(org_data, org_path.with_suffix(".png"))
        do_save_spec(wm_data, wm_path.with_suffix(".png"))

        do_save_spec(org_data-wm_data, wm_path.with_name(f"{wm_path.stem}_diff.png"))

    return


def show_fake_spec():
    out_dir = Path("/home/wzong/My Passport/projects/FakeAudioDetection/out/tmp")
    voice_lst = [
        [out_dir.joinpath("234_online.wav"), out_dir.joinpath("p234_006_mic1.wav")],
    ]

    def do_save_spec(y, fig_path):
        S = np.abs(librosa.stft(y))

        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                               ref=np.max),
                                       y_axis='log', x_axis='time', ax=ax)
        ax.set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        plt.savefig(fig_path)

    for (org_path, wm_path) in voice_lst:
        org_data, sr = librosa.load(org_path, sr=16000)
        wm_data, sr = librosa.load(wm_path, sr=16000)

        do_save_spec(org_data[:len(wm_data)], org_path.with_suffix(".png"))
        do_save_spec(wm_data, wm_path.with_suffix(".png"))

    return

def voice_convert():
    text = "Legal writing is usually less discursive than writing in other humanities subjects."
    speaker_wav = [
        out_dir.joinpath("15 March PTE RA 1.wav"),
        out_dir.joinpath("15 March PTE RA 2.wav"),
        out_dir.joinpath("15 March PTE RA 3.wav")
    ]
    language = "en"
    file_path = out_dir.joinpath("15 March PTE RA 3_gen.wav")

    tts = api.TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True, gpu=True)
    tts.tts_to_file(
        "Legal writing is usually less discursive than writing in other humanities subjects.",
        language="en",
        speaker_wav=speaker_wav,
        file_path=file_path
    )

    if tts.voice_converter is None:
        tts.load_vc_model_by_name("voice_conversion_models/multilingual/vctk/freevc24")

    wav = tts.voice_converter.voice_conversion(str(out_dir.joinpath("VCTK_src.wav")), target_wav=str(out_dir.joinpath("VCTK_tgt.wav")))

    wav_op.write(file_path.with_name(f"VCTK_tgt_converted.wav"),
                 tts.voice_converter.vc_config.audio.output_sample_rate,
                 wav)


def combine_wm_audio(watermarked):
    speaker_id = "p347"
    speaker = f"VCTK_{speaker_id}"

    if watermarked is True:
        audio_dir = Path(f"/home/wzong/My Passport/projects/FakeAudioDetection/out/ExpEmbedWatermark/{speaker}")
        path_finder = "*.wav"
    else:
        # original dir
        audio_dir = Path(f"/home/wzong/My Passport/projects/data/vctk/wav48_silence_trimmed/{speaker_id}")
        path_finder = "*mic1.flac"

    max_len = 16000 * 250   # 3700000
    full_audio = np.zeros(max_len)
    cur_pos = 0

    all_audio_path_lst = list(audio_dir.glob(path_finder))
    all_audio_path_lst.sort()

    for audio_path in all_audio_path_lst:
        waveform, sr = utils.read_audio(audio_path, expected_sr=16000)
        wav_len = len(waveform)
        if cur_pos + wav_len > max_len:
            break

        full_audio[cur_pos: cur_pos + wav_len] = waveform

        cur_pos += wav_len

    full_audio = utils.resample_wav(full_audio, 16000, 24000)

    wav.write(Path(f"/home/wzong/My Passport/projects/FakeAudioDetection/out/tmp").joinpath(f"{speaker}_full_wm_{watermarked}.wav"), 24000, full_audio)


def sentence_for_playHT():
    all_lines = exp_setup.get_gen_sentences_lst(exp_cfg)
    assert len(all_lines) == 100

    # combine them into one file without new lines

    with open(Path(exp_cfg.out_dir).joinpath("PlayHT_sentences.txt"), "w") as f:
        for idx, line in enumerate(all_lines):
            f.write(line)
            if idx != len(all_lines) - 1:
                f.write(" ")


def my_func():
    deposit = 150000

    rent_pw = 280
    total_rent = (365 / 7) * rent_pw

    loan_rate_lst = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    house_price_lst = [650000, 700000, 750000, 800000, 850000, 900000]
    house_price_annual_increase = 0.05

    ongoing_costs = 8000

    for loan_rate in reversed(loan_rate_lst):

        bank_interest = loan_rate - 0.015

        print(f"deposit = {deposit/1000:.0f}k \tloan rate = {loan_rate:.3f}\tbank_interest = {bank_interest:.3f}\t"
              f"house_price_increase = {house_price_annual_increase:.2f}\t"
              f"rent = {rent_pw}:\n")

        for house_price in reversed(house_price_lst):

            wealth_without_house = (
                deposit * bank_interest                         # bank interest
                + (house_price - deposit) * loan_rate           # loan
                + ongoing_costs                                 # ongoing costs

                - total_rent                                    # rent
                - house_price * house_price_annual_increase     # price increase
            )

            print(f"house = {house_price/1000:.0f}k: wealth = {wealth_without_house:.2f}")
        print("\n")


def to_mp3():

    def do_convert(fpath):
        utils.wav_to_other_codec(fpath, format_name="mp3", bitrate="128k")

    do_convert(Path(exp_cfg.out_dir).joinpath(f"ExpOnline/Obama_combined_wm_ExpEmbedWatermark.wav"))
    do_convert(Path(exp_cfg.out_dir).joinpath(f"ExpOnline/Obama_combined_wm_None.wav"))


def main():

    # test_down_load_tts()

    # to_mp3()

    # my_func()

    # msg = 0x3
    # code = utils.hamming_encode(msg, 3)
    # print(code)
    #
    # org_msg = utils.hamming_decode(code, len(code))
    # print(int(org_msg, 2))
    #
    # print(int(utils.hamming_decode("011111", len(code)), 2))
    # print(int(utils.hamming_decode("011100", len(code)), 2))
    # print(int(utils.hamming_decode("011010", len(code)), 2))
    # print(int(utils.hamming_decode("010110", len(code)), 2))
    # print(int(utils.hamming_decode("001110", len(code)), 2))
    # print(int(utils.hamming_decode("111110", len(code)), 2))
    #
    # error_msg = utils.hamming_decode("010010", 6)
    # print(int(error_msg, 2))

    # combine_wm_audio(True)
    # combine_wm_audio(False)

    # calculate_wm()

    eval_Obama_wm()

    # wavmark_wm()

    # show_spec()

    # sentence_for_playHT()

    # show_fake_spec()

    # pred_cfg_dic_path = "/media/weizong/My Passport/projects/DefendAgainstUnrestricted/out/" \
    #               "MNIST/ExpTrainRobustModel/MyRobustModel_2/pred_cfgs/pred_cfg_maxnum_1500.bin"
    #
    # with open(pred_cfg_dic_path, 'rb') as handle:
    #     raw_pred_cfg_lst = pickle.load(handle)
    #
    # def _prepare_pred():
    #     # flatten all the embeddings into an array
    #     label_arr = []
    #     latent_arr = []
    #     recons_loss_arr = []
    #     sim_loss_arr = []
    #
    #     for detail_dic in raw_pred_cfg_lst:
    #         label_arr.append(detail_dic["label"])
    #         latent_arr.append(detail_dic["latent"])
    #         recons_loss_arr.append(detail_dic["recons_loss"])
    #         sim_loss_arr.append(detail_dic["sim_loss"])
    #
    #     label_arr = torch.LongTensor(np.array(label_arr)).to(utils.device)
    #     latent_arr = torch.FloatTensor(np.array(latent_arr)).to(utils.device)
    #     recons_loss_arr = torch.FloatTensor(np.array(recons_loss_arr)).to(utils.device)
    #     sim_loss_arr = torch.FloatTensor(np.array(sim_loss_arr)).to(utils.device)
    #
    #     return {
    #         "label_arr": label_arr, "label_arr_unique": label_arr.unique(),
    #         "latent_arr": latent_arr,
    #         "recons_loss_mean": torch.mean(recons_loss_arr), "recons_loss_std": torch.std(recons_loss_arr),
    #         "sim_loss_mean": torch.mean(sim_loss_arr), "sim_loss_std": torch.std(sim_loss_arr),
    #     }
    #
    # prepared_rlt = _prepare_pred()
    # prepared_rlt = prepared_rlt

    # cut_my_voice()

    # resample_benign()

    # test_stft()
    # noise_same_stft()

    # libro_test()

    # test_mfcc()

    # misllaneous()

    # test_hearing()


if __name__ == '__main__':

    main()
    sys.exit()



