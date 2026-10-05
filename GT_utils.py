import os
import random
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
import pyworld
import python_stretch as ps
from scipy.ndimage import gaussian_filter1d
from rich.progress import track
from torch.utils.data import DataLoader
import cma
import scipy.signal as spsig
import librosa
from losses import MultiMelSpectrogramLoss
from timbre_watermarking.model.conv2_mel_modules import (
    Encoder,
    Decoder,
    Discriminator,
)
from wavmark_main.src import wavmark


warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
TIMBRE_DIR = PROJECT_DIR / "timbre_watermarking"
TIMBRE_CONFIG_DIR = TIMBRE_DIR / "config_wm"

# set seeds
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
batch_size = 1
msg_length = 10
mel_loss = MultiMelSpectrogramLoss()

def cents_to_ratio(cents):
    return 2 ** (cents / 1200)

def generate_random_ratios(n_bins, cents_range=(-85, 85)): # yige array

    ratios = []

    for _ in range(n_bins):
        sign = 1
        cents = np.random.uniform(cents_range[0], cents_range[1])
        final_cents = sign * cents
        ratio = cents_to_ratio(final_cents)
        ratios.append(ratio)
    return np.array(ratios)



def generate_bounded_random_cuts(boundaries, max_cuts=3):

    refined_boundaries = np.array(boundaries)
    widths = np.diff(refined_boundaries)
    
    # 1. 计算每个 band 的概率 (宽度 / 最大宽度)
    max_w = np.max(widths)
    probs = widths / max_w
    cuts_made = 0
    new_points = []

    for i in range(len(widths)):

        if cuts_made >= max_cuts:
            break

        if np.random.rand() < probs[i]:
            low = refined_boundaries[i]
            high = refined_boundaries[i+1]

            new_pt = np.random.uniform(low, high)
            new_points.append(new_pt)
            cuts_made += 1

    final_boundaries = np.sort(np.append(refined_boundaries, new_points))
    
    return final_boundaries

def inherit_ratios(old_boundaries, old_ratios, new_boundaries):

    old_boundaries = np.array(old_boundaries)
    new_ratios = []

    for i in range(len(new_boundaries) - 1):

        mid_point = (new_boundaries[i] + new_boundaries[i+1]) / 2.0
        
        idx = np.searchsorted(old_boundaries, mid_point) - 1

        idx = max(0, min(idx, len(old_ratios) - 1))

        new_ratios.append(old_ratios[idx])

    return new_ratios


def get_encoder_decoder():
    # warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p", "--process_config", type=str,
        default=str(TIMBRE_CONFIG_DIR / "process.yaml"), help="Path to process.yaml"
    )
    parser.add_argument(
        "-m", "--model_config", type=str,
        default=str(TIMBRE_CONFIG_DIR / "model.yaml"), help="Path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str,
        default=str(TIMBRE_CONFIG_DIR / "train.yaml"), help="Path to train.yaml"
    )
    parser.add_argument(
        "-n", "--name", type=str,
        default=str(TIMBRE_DIR / "experiments_results"), help="Path to save results"
    )

    # Parse arguments with an empty list to avoid Jupyter errors
    args = parser.parse_args([])

    # Print parsed arguments for verification

    with open(args.process_config, "r") as config_file:
        process_config = yaml.safe_load(config_file)
    print()
    with open(args.model_config, "r") as config_file:
        model_config = yaml.safe_load(config_file)
    with open(args.train_config, "r") as config_file:
        train_config = yaml.safe_load(config_file)
    configs = (process_config, model_config, train_config)
    process_config, model_config, train_config = configs
    # audios = my_dataset(process_config=process_config, train_config=train_config,flag='test') #audio dataset
    # model config
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"] #512
    nlayers_encoder = model_config["layer"]["nlayers_encoder"] # 6 
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]# 6
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"] # 8
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]# 8
    msg_length = train_config["watermark"]["length"] # 10

    encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
    decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)

    path_model = Path(model_config["test"]["model_path"])
    if not path_model.is_absolute():
        path_model = TIMBRE_DIR / path_model
    model_name = model_config["test"]["model_name"]

    index = model_config["test"]["index"] 
    model_list = [path for path in path_model.iterdir() if path.is_file()]

    model_list = sorted(model_list, key=lambda path: path.stat().st_mtime) #按文件时间对checkpoints 排序
    if not model_list:
        raise FileNotFoundError(f"No Timbre watermark checkpoint found in: {path_model}")
    model_path = model_list[index]
    model = torch.load(model_path, map_location=device)
    logging.info("model <<{}>> loadded".format(model_path))

    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"], strict=False)
    encoder.eval()
    decoder.eval()

    return encoder, decoder

def query(msg, wav, decoder, watermark = 'timbre'):

    with torch.no_grad():
        if watermark == 'timbre':
            if type(wav) == np.ndarray:
                wav = torch.from_numpy(wav)

            # 解码
            with torch.no_grad():
                decoded = decoder.test_forward(wav.unsqueeze(0).cuda())

            decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
            decoder_acc = round(decoder_acc.item(),2)
        elif watermark == 'audioseal':
            result, msg = decoder.detect_watermark(torch.tensor(wav).unsqueeze(0), sample_rate=16000, message_threshold=0.5)
            decoder_acc = result
        
        elif watermark == 'robustdnn':

            from dnn_audio_watermarking.main import detect

            if wav.shape[0] == 1:
                wav = wav.squeeze(0)


            TARGET_LENGTH = 33226
            signal = tf.convert_to_tensor(wav, dtype=tf.float32)

            if len(signal) < TARGET_LENGTH:

                padding_length = TARGET_LENGTH - len(signal)
                signal = np.pad(signal, (0, padding_length), 'constant')

            elif len(signal) > TARGET_LENGTH:

                signal = signal[:TARGET_LENGTH]

            signal = tf.expand_dims(signal, axis=0)  # shape [1, num_samples]

            watermark = detect(decoder, signal)
            

            bits = np.where(watermark >= 0.5, 1, 0)
            matches = np.sum(bits == msg)
            total = len(bits[0])
            decoder_acc = np.round(matches / total,2)
        elif watermark == 'audiomarknet':

            if wav.shape[0] == 1:
                wav = wav.squeeze(0)

            int_number = len(wav)/16000
            need_detect = int(int_number)
            with torch.no_grad():
                x = wav
                wav_secs = decoder.split_waveform(x)

                logits = decoder.decoder(decoder.spec_for_classificiation(wav_secs)[..., decoder.min_freq_idx: decoder.max_freq_idx, :])
            pred_wm = (logits > 0.0).long()
            pred_wm = pred_wm[:need_detect,:]
            list_acc = []
            for one_wm in pred_wm:
                decoder_acc = (one_wm == msg).sum() / one_wm.numel()
                list_acc.append(decoder_acc.item())
                
            # print('acc is --------------', decoder_acc)
            decoder_acc = max(list_acc)
            # print(f'bestacc is {decoder_acc}')
            # print('decoder_acc is ', decoder_acc)
        elif watermark == 'wavmark':
            payload_decoded, _ = wavmark.decode_watermark(decoder, wav, show_progress=True)
            if payload_decoded is None:
                decoder_acc = 0
            else:
                decoder_acc=((payload_decoded > 0) == (msg > 0)).float().mean()
        
        elif watermark == 'silentcipher':
            result = decoder.decode_wav(np.squeeze(wav), 44100, phase_shift_decoding=False)
            if result['status'] == False:
                decoder_acc = 0
            else:
                decoder_acc =np.mean(np.array(result['messages'][0]) == np.array([123, 234, 111, 222, 11]))
        elif watermark == 'collaborative':
            wav = torch.tensor(wav).unsqueeze(0)
            result1, result = decoder(wav, wav)
            decoder_acc = np.round((1-result).item(),2)



    return decoder_acc



def custom_map_factory(ratio_or_array, sr, n_bins=1000, freq_points=None):

    if np.isscalar(ratio_or_array):
        def freq_map(f):
            return f * ratio_or_array
        return freq_map
    
    ratios = np.asarray(ratio_or_array)
    assert len(ratios) == n_bins, f"Length of ratios should be {n_bins}"


    if freq_points is None:
        freq_points = np.linspace(0, sr/2, n_bins+1)
    else:
        freq_points = np.asarray(freq_points)
        assert len(freq_points) == n_bins+1, f"Length of freq_points should be {n_bins+1}"

    def freq_map(f):
        freq_hz = f * sr
        idx = np.searchsorted(freq_points, freq_hz, side='right') - 1
        idx = np.clip(idx, 0, n_bins-1)
        return f * ratios[idx]
    
    return freq_map


def global_recover(attack_audio, current_msg, wmaudio, decoder, sr=22050, step_up_floor = 1.10, step_down_floor = 0.90):

    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 22050)
    min_mel = 100
    final_acc = 1
    recover_ratio =2
    scalar_ratio_list = np.linspace(step_down_floor, step_up_floor, 100)
    min_audio = attack_audio

    for _, ratio in enumerate(scalar_ratio_list):

        stretch.setFreqMap(custom_map_factory(ratio, sr))
        global_recover_audio = stretch.process(attack_audio)
        min_len = min(global_recover_audio.shape[1], wmaudio.shape[1])
        global_recover_audio = torch.tensor(global_recover_audio)

        a = global_recover_audio[:, :min_len]  
        b = wmaudio[:, :min_len]        
        mel = mel_loss(a, b)
        recover_acc = query(current_msg, global_recover_audio, decoder, watermark='timbre')
        if mel<min_mel:
            min_mel = mel
            min_audio = global_recover_audio
            final_acc = recover_acc
            recover_ratio = ratio

    return final_acc, min_audio, recover_ratio

def generate_uniform_freq_points(sr, n_bins):

    freq_points = np.linspace(0, sr / 2, n_bins + 1)
    return freq_points



def objective_function(ratios, ori_audio, oriaudio_2, current_msg, decoder, sr, n_bins, boundaries, iter = 10):

    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, sr)
    stretch.setFreqMap(custom_map_factory(ratios, sr, n_bins, boundaries))
    audio = stretch.process(ori_audio)
    audio = torch.tensor(mmse_stsa(audio)).unsqueeze(0)
    # audio = torch.tensor(audio)
    
    ori_audio_d = oriaudio_2
    
    # ori_audio_d = ori_audio
    print('shape is, ', audio.shape, 'ori shape is ', ori_audio_d.shape)

    acc = query(current_msg, audio, decoder)
    
    mel = mel_loss(audio, ori_audio_d)
    
    # if you want recover open this 
    recover_acc, _, _ = global_recover(
    current_msg=current_msg, attack_audio=audio, wmaudio=ori_audio_d, decoder=decoder
)
    
    loss = acc + mel + recover_acc 

    print(f"Current loss: {loss}, acc: {acc}, Mel_loss: {mel}, recover_acc: {recover_acc}")
    
    return (loss, {'acc': acc, 'mel_loss': mel, 'min_audio': audio, 'recover_acc':recover_acc, 'boundaries':boundaries})


def cmaes_optimize(ori_audio, oriaudio_2,current_msg, decoder, ratios, boundaries, threshold_wm = 0.8, threshold_recover = 0.8,sr=22050, max_iter=5000, step_up_floor=1.05, step_down_floor=0.98, iter=20): #0.95


    initial_ratios = ratios
    initial_ratios = np.clip(initial_ratios, step_down_floor, step_up_floor)
    initial_sigma = 0.1

    es = cma.CMAEvolutionStrategy(
        initial_ratios, initial_sigma, 
        {'bounds': [step_down_floor, step_up_floor], 'maxfevals': max_iter}
    )

    best_overall_loss = float('inf')
    best_overall_audio = None
    best_overall_ratios = None
    
    while not es.stop():
        solutions = es.ask()
        losses = []
        extra_data = []
        
        for raw_ratios in solutions:
            
            # raw_ratios = push_away_from_one(raw_ratios)
            loss, data = objective_function(raw_ratios, ori_audio, oriaudio_2, current_msg, decoder, sr, len(boundaries)-1, boundaries, iter = iter)
            
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            
            losses.append(loss_val)
            extra_data.append(data)
            
            if loss_val < best_overall_loss:
                best_overall_loss = loss_val
                best_overall_audio = data['min_audio']
                best_recover_acc = data['recover_acc']

                best_acc = data['acc']
                boundaries = data['boundaries']
                best_overall_ratios = raw_ratios 
                
            if data['acc'] <= threshold_wm and data['recover_acc'] <= threshold_recover:
                best_overall_loss = loss_val
                best_overall_audio = data['min_audio']
                best_recover_acc = data['recover_acc']

                best_acc = data['acc']
                boundaries = data['boundaries']
                best_overall_ratios = raw_ratios 
                return best_overall_audio, best_acc, best_recover_acc, best_overall_ratios, boundaries
            
        if best_acc <= threshold_wm and best_recover_acc <= threshold_recover:
            return best_overall_audio, best_acc, best_recover_acc, best_overall_ratios, boundaries
        
        es.tell(solutions, losses)
        print(f"Iter: {es.countiter}, Global Best Loss: {best_overall_loss}, Sigma: {es.sigma:.4f}")

    return best_overall_audio, best_acc, best_recover_acc, best_overall_ratios, boundaries

def estimate_noise_psd(M, frames_for_noise):
    """
    Estimate noise magnitude & power from first N 'noise' frames (simple, stationary).
    M: magnitude spectrogram (freq x time)
    frames_for_noise: int, number of initial frames assumed noise-only
    """
    N = min(frames_for_noise, M.shape[1])
    if N <= 0:
        N = min(6, M.shape[1])  # fallback
    noise_mag = np.maximum(np.mean(M[:, :N], axis=1, keepdims=True), 1e-8)
    noise_psd = noise_mag ** 2
    return noise_mag, noise_psd



def stft(y, n_fft=1024, hop=256, win="hann", center=True):
    w = spsig.get_window(win, n_fft, fftbins=True).astype(np.float32)
    return librosa.stft(y, n_fft=n_fft, hop_length=hop, window=w, center=center)


def istft(S, hop=256, win="hann", length=None, center=True):
    n_fft = (S.shape[0] - 1) * 2
    w = spsig.get_window(win, n_fft, fftbins=True).astype(np.float32)
    return librosa.istft(S, hop_length=hop, window=w, length=length, center=center)

def _mmse_stsa_gain(xi, gamma):
    nu = (gamma * xi) / (1.0 + xi + 1e-12)
    G = (xi / (1.0 + xi + 1e-12)) * np.exp(0.5 * np.exp(-np.clip(nu, 0, 40)))
    return np.clip(G, 0.0, 1.0).astype(np.float32)

def magphase(S):
    return np.abs(S), np.exp(1j * np.angle(S))

def mmse_stsa(y, n_fft=1024, hop=256, noise_frames=6, alpha_dd=0.98):
    y = np.array(y)
    if y.shape[0] == 1:
        y = y.squeeze(0)
    S = stft(y, n_fft=n_fft, hop=hop)
    M, P = magphase(S)
    _, Npsd = estimate_noise_psd(M, noise_frames)
    Ypsd = np.maximum(M**2, 1e-12)

    gamma = Ypsd / np.maximum(Npsd, 1e-12)
    xi_prev = np.maximum(gamma - 1.0, 0.0)

    M_hat = np.zeros_like(M)
    for t in range(M.shape[1]):
        if t == 0:
            xi = np.maximum(gamma[:, [t]] - 1.0, 0.0)
        else:
            xi = alpha_dd * (M_hat[:, [t-1]]**2) / np.maximum(Npsd, 1e-12) + (1 - alpha_dd) * np.maximum(gamma[:, [t]] - 1.0, 0.0)

        G = _mmse_stsa_gain(xi, gamma[:, [t]])
        M_hat[:, t:t+1] = G * M[:, t:t+1]

    S_hat = M_hat * P
    y_hat = istft(S_hat, hop=hop, length=len(y))
    return y_hat.astype(np.float32)

def embedding_surrogate(wav_path, encoder, decoder):
    batch_size = 1
    msg_length = 10

    encoder = encoder.cuda().eval()
    decoder = decoder.cuda().eval()

    msg = np.ones([batch_size, 1, msg_length], dtype=np.float32)
    msg = torch.from_numpy(msg).float() * 2 - 1
    msg = msg.cuda().contiguous()

    wav, sr = librosa.load(wav_path, sr=22050)
    # wav = wav[:sr * 10]
    ori_wav = torch.tensor(wav).unsqueeze(0)
    wav_matrix = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).cuda().contiguous()

    with torch.no_grad():
        encoded, carrier_watermarked = encoder.test_forward(wav_matrix, msg)

        decoded = decoder.test_forward(encoded)

    print('encoded shape is ---', encoded.shape)
    print('decoded shape is ---', decoded.shape)
    print('msg shape is ---', msg.shape)

    decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
    print('decoder_acc is ', decoder_acc)

    wm_audio = encoded.detach().cpu().squeeze(0).contiguous()
    msg = msg.detach().cuda().contiguous()

    print("return wm_audio:", wm_audio.shape, wm_audio.dtype, wm_audio.device)
    print("return msg:", msg.shape, msg.dtype, msg.device)

    return wm_audio.float(), ori_wav, msg
