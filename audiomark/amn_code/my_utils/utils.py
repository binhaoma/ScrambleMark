"""
    Part of code comes from https://github.com/moboehle/Pytorch-LRP
"""
import logging
import pickle
from enum import Enum

from datetime import datetime
from zipfile import ZipFile

import matplotlib
import matplotlib.pyplot as plt
import torch
import wget
from scipy.ndimage import zoom

import cv2
import pyaudio
import wave
from scipy import signal
import collections, queue
import webrtcvad

from pathlib import Path
import numpy as np

from tqdm import tqdm
import tarfile
import noisereduce as nr
import gdown
import scipy.io.wavfile as wav
import librosa
from torch.nn import functional as F
from pydub import AudioSegment
from audiomark.amn_code.my_utils import my_hamming_code
from pesq import pesq


# MNIST_MEAN = 0.1307
# MNIST_STD = 0.3081

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2471, 0.2435, 0.2616])

CIFAR100_MEAN = np.array([0.5070758,  0.4865503,  0.44091913])
CIFAR100_STD = np.array([0.26733097, 0.25643396, 0.27614763])


device = torch.device('cuda:0')     # use GPU


def set_config(obj, config_dic):
    for k, v in config_dic.items():
        setattr(obj, k, v)


class ConfigBase:
    def __init__(self, **kwargs):
        set_config(self, kwargs)


def pprint(*args):
    out = [str(argument) + "\n" for argument in args]
    print(*out, "\n")


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))


def scale_mask(mask, shape):

    if shape == mask.shape:
        print("No rescaling necessary.")
        return mask

    nmm_map = np.zeros(shape)
    print("Rescaling mask")
    for lbl_idx in np.unique(mask):
        nmm_map_lbl = mask.copy()
        nmm_map_lbl[lbl_idx != nmm_map_lbl] = 0
        nmm_map_lbl[lbl_idx == nmm_map_lbl] = 1
        zoomed_lbl = zoom(nmm_map_lbl, 1.5, order=3)
        zoomed_lbl[zoomed_lbl != 1] = 0
        remain_diff = np.array(nmm_map.shape) - np.array(zoomed_lbl.shape)
        pad_left = np.array(np.ceil(remain_diff / 2), dtype=int)
        pad_right = np.array(np.floor(remain_diff / 2), dtype=int)
        nmm_map[pad_left[0]:-pad_right[0], pad_left[1]:-pad_right[1], pad_left[2]:-pad_right[2]] += zoomed_lbl * lbl_idx

    return nmm_map


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as a heatmap.
    By default, the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return cam



class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech
        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        print("write wav %s" % filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=700, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()





def get_mean_and_std(dset):
    channels_sum, channels_squared_sum, num_data = 0, 0, 0
    for data, _ in tqdm(dset):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[1, 2])
        channels_squared_sum += torch.mean(data ** 2, dim=[1, 2])
        num_data += 1

    mean = channels_sum / num_data

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_data - mean ** 2) ** 0.5

    return mean.numpy(), std.numpy()


def permute_reverse_idx(rand_state, num):
    permu = np.array(list(reversed(range(num))))
    return permu


def non_repeatable_permute_idx(rand_state, num):
    assert False, "not used in this work!!!"
    permu = rand_state.permutation(num)

    # check each element to ensure that every number at each position is changed
    for idx in range(len(permu) - 1):
        x = permu[idx]
        if x == idx:
            permu[idx] = permu[idx+1]
            permu[idx + 1] = x

    assert np.sum(permu == np.arange(num)) == 0, "no repeated idx"
    return permu



def permute_model(model_state, reverse_input, seed=8462):
    """
    inverse_input: whether inverse the input
    """
    rand_state = np.random.RandomState(seed=seed)

    first_key = list(model_state)[0]
    first_weights = model_state[first_key]

    init_dim = None
    if len(first_weights.shape) == 2:
        # victim model starts with a linear layer
        init_dim = first_weights.shape[1]
        assert first_key.find("_ih_") > 0,  "we only support starts with a lstm now..."

    elif len(first_weights.shape) == 4:
        # victim model starts with a conv layer
        init_dim = first_weights.shape[1]

    assert init_dim is not None, "victim model should start with a linear or conv"
    last_perm = np.arange(start=0, stop=init_dim)
    res_last_perm = None        # for ResNet shortcut

    class LayerType(Enum):
        CONV = 1
        FC = 2
        LSTM = 3
    last_layer_type = None

    # permute all weights in a model
    for idx, (layer_name, layer_weights) in enumerate(model_state.items()):
        new_weights = None

        if len(layer_weights.shape) == 0:
            new_weights = layer_weights  # do nothing

        elif len(layer_weights.shape) == 4:
            # this is a conv layer

            if reverse_input:
                layer_weights = torch.flip(layer_weights, dims=[2])     # make it upside down

            if layer_name.find(".0.conv1.") > 0:
                # beginning of ResNet block
                res_last_perm = last_perm   # save this last perm

            if layer_name.find(".shortcut.") > 0:
                new_weights = layer_weights[:, res_last_perm, :, :]
                res_last_perm = None
            else:
                new_weights = layer_weights[:, last_perm, :, :]
                # generate new permutation for non shortcut
                last_perm = permute_reverse_idx(rand_state, layer_weights.shape[0])

            new_weights = new_weights[last_perm, :, :, :]

            last_layer_type = LayerType.CONV

        elif len(layer_weights.shape) == 1:
            # this is bias or batchnorm weigths
            if last_layer_type == LayerType.LSTM:
                # for lstm, we should permute all sections in the same way
                assert last_perm.shape[0] * 4 == layer_weights.shape[0], "LSTM layers weights have 4 sections"

                new_weights = layer_weights
                section_size = new_weights.shape[0] // 4
                new_weights = new_weights.reshape([4, section_size])
                new_weights = new_weights[:, last_perm]
                new_weights = new_weights.reshape(section_size * 4)

            else:
                # for other layers
                new_weights = layer_weights[last_perm]

        elif len(layer_weights.shape) == 2:
            if layer_name.find("_ih_") > 0 or layer_name.find("_hh_") > 0:
                # this is a LSTM layer
                assert idx < 2, "only support starting with one-layer LSTM now"

                # flip it first
                new_weights = layer_weights
                if reverse_input and last_layer_type is None:
                    # only flip for the first layer in LSTM
                    new_weights = torch.flip(new_weights, dims=[1])

                new_weights = new_weights[:, last_perm]

                # the first dim is divided into 4 sections
                assert new_weights.shape[0] % 4 == 0
                section_size = new_weights.shape[0]//4

                if last_layer_type is None:
                    # the section layer in LSTM follows the same permutation as the first layer
                    last_perm = permute_reverse_idx(rand_state, section_size)

                # we permute all sections in the same way
                new_weights = new_weights.reshape([4, section_size, -1])
                new_weights = new_weights[:, last_perm, :]
                new_weights = new_weights.reshape(section_size*4, -1)

                last_layer_type = LayerType.LSTM

            else:
                # this is a linear layer
                if last_layer_type is None:
                    # the model starts with this linear layer
                    raise NotImplementedError

                elif last_layer_type == LayerType.CONV or last_layer_type == LayerType.FC \
                        or last_layer_type == LayerType.LSTM:
                    # if last layer is a conv layer, the activation map may be larger than 1*1
                    assert layer_weights.shape[1] % last_perm.size == 0, "must be dividable"

                    act_map_size = layer_weights.shape[1] // last_perm.size     # the size of activation maps
                    filter_size = int(np.sqrt(act_map_size))
                    assert act_map_size == filter_size * filter_size, "we assume square filters."

                    new_weights = layer_weights.reshape(layer_weights.shape[0], last_perm.size,
                                                        filter_size, filter_size)

                    if reverse_input:
                        new_weights = torch.flip(new_weights, dims=[2])

                    new_weights = new_weights[:, last_perm, :, :]
                    new_weights = new_weights.reshape(layer_weights.shape[0], layer_weights.shape[1])

                    if (idx == len(model_state) - 2) or (idx == len(model_state) - 1):
                        # this is the last linear, we do not permute the output
                        last_perm = np.arange(start=0, stop=layer_weights.shape[0])
                    else:
                        # we permute the output if this is not the weight for output linear layer
                        last_perm = permute_reverse_idx(rand_state, layer_weights.shape[0])
                        new_weights = new_weights[last_perm, :]

                else:
                    assert False, "unexpected last_layer_type."

                last_layer_type = LayerType.FC

        assert new_weights is not None
        model_state[layer_name] = new_weights

    return model_state


def exclude_layers(org_model_state, exlude_layer_names):
    new_dict = {}
    for k, v in org_model_state.items():
        do_add = True
        for layer_name in exlude_layer_names:
            if k.find(layer_name) == 0:
                # find excluded layer, we do not add it
                do_add = False
                break
        if do_add is True:
            new_dict[k] = v

    return new_dict


def cal_ft_epoch(subset_split, ft_expected_epoch):
    if np.isclose(subset_split, 1.0):
        ft_epoch = ft_expected_epoch
    else:
        ft_epoch = int((1 / (1 - subset_split)) * ft_expected_epoch)

    return ft_epoch


def write_args_to_file(args, fpath):

    def do_write(_a, f):
        for k, v in vars(_a).items():
            if isinstance(v, ConfigBase):
                f.write(f"*"*80 + "\n")
                f.write(f"{type(v).__name__}:\n")
                do_write(v, f)
                f.write(f"-"*80 + "\n")
            else:
                f.write(f" • {k:<30}:\t{v}\n")

    with open(fpath, "w") as f_handle:
        do_write(args, f_handle)


def fingerprinting_get_pos_neg_ckpt_lst(out_dir, model_creator, train_model_num):
    victim_ckpt_dir_lst = []
    suspect_ckpt_dir_dic = {}
    for model_idx in range(train_model_num):
        # victim model checkpoint
        victim_ckpt_dir_lst.append(out_dir.joinpath(f"trained_models/{model_creator.__name__}_{model_idx}/ckpt"))

        # corresponding suspect
        suspect_lst = suspect_ckpt_dir_dic[model_idx] = []
        suspect_lst.append(out_dir.joinpath(f"ExpDestroyFingerprints/{model_creator.__name__}_{model_idx}/distill_ckpt"))

    return victim_ckpt_dir_lst, suspect_ckpt_dir_dic


def save_existing_log(log_path):
    if not log_path.exists():
        return

    dateTimeObj = datetime.now()

    save_path = log_path.with_name(
        f"{log_path.stem}_saved_at_"
        f"{dateTimeObj.day}-{dateTimeObj.month}-{dateTimeObj.year}_"
        f"{dateTimeObj.hour}-{dateTimeObj.minute}-{dateTimeObj.second}"
        f".log"
    )
    log_path.rename(save_path)

    print(f"log saved at {save_path}\n\n")


def plot_img_and_save(img, save_path, **extra_args):
    if len(img.shape) == 1:
        # waveform
        plt.figure()
        plt.plot(img)
        plt.savefig(save_path)
        plt.close()
    else:
        matplotlib.image.imsave(save_path, img, **extra_args)



def save_random_example(out_dir, dataset, num):
    rand_idx_arr = np.random.RandomState(seed=6548).permutation(len(dataset))[: num]

    for idx in rand_idx_arr:
        x, y = dataset[idx]
        x = x.squeeze().numpy()

        if len(x.shape) == 3:
            x = np.transpose(x, [1, 2, 0])

        matplotlib.image.imsave(out_dir.joinpath(f"id_{idx}_label_{y}.png"), x)


def stft_to_abs(data, device=None, return_freq=False, n_fft=256, hop_length=64):
    if device is not None:
        data = data.to(device)

    # transform wav data via stft
    freq_data = torch.stft(data, n_fft=n_fft, hop_length=hop_length,
                           window=torch.hann_window(n_fft).to(device), return_complex=True)
    # freq_power = freq_data[..., 0] ** 2 + freq_data[..., 1] ** 2
    # freq_abs = torch.sqrt(freq_power)
    freq_abs = torch.abs(freq_data)

    if return_freq is True:
        return freq_abs, freq_data

    return freq_abs




def dataset_examples(dset, out_dir, clip_min, clip_max,
                     autoencoder=None, device=None, num_per_label=10, mean=None, std=None):
    if not out_dir.exists():
        Path.mkdir(out_dir)

    # from each label we export num_per_label examples
    unique_labels = dset.targets.unique()
    cnt_dic = {}
    total_num = unique_labels.size(0) * num_per_label
    rand_idx = np.random.RandomState(12321).permutation(len(dset))

    if (mean is not None) and (std is not None):
        mean = np.array(mean).reshape([1, 1, len(mean)])
        std = np.array(std).reshape([1, 1, len(std)])

    for i in rand_idx:
        if sum(cnt_dic.values()) == total_num:
            # enough examples are generated
            break

        img, label = dset[i]
        cur_num = cnt_dic.get(label, 0)
        if cur_num >= num_per_label:
            continue
        cnt_dic[label] = cur_num + 1

        # output this
        extra_args = {}
        if len(img.shape) <= 2 or img.shape[0] == 1:
            extra_args = {"cmap": matplotlib.cm.gray}

        if autoencoder is not None:
            autoencoder.eval()
            gen_img = torch.FloatTensor(img).to(device)
            gen_img = gen_img.unsqueeze(0)
            gen_img = autoencoder(gen_img)
            gen_img = gen_img.cpu().detach().numpy().squeeze()

            if len(gen_img.shape) > 2:
                gen_img = np.transpose(gen_img, (1, 2, 0))

            plot_img_and_save(np.clip(gen_img, clip_min, clip_max),
                              out_dir.joinpath(f"label_{label}_id_{i}_reconstructed.png"), **extra_args)

        img = img.squeeze()
        img = img.numpy()

        if len(img.shape) > 2:
            img = np.transpose(img, (1, 2, 0))

        if (mean is not None) and (std is not None):
            if len(img.shape) <= 2:
                mean = mean.squeeze()
                std = std.squeeze()
            img = img * std + mean

        plot_img_and_save(np.clip(img, clip_min, clip_max),
                          out_dir.joinpath(f"label_{label}_id_{i}.png"), **extra_args)

    # finish


def save_tensor_img(img, clip_min, clip_max, save_path):
    img = img.squeeze().cpu().detach().numpy()

    extra_args = {}
    if len(img.shape) > 2:
        img = np.transpose(img, [1, 2, 0])
    else:
        extra_args = {"cmap": matplotlib.cm.gray}

    plot_img_and_save(np.clip(img, clip_min, clip_max), save_path, **extra_args)


def variable_len_wav_collate(batch):
    device = batch[0][0].device

    data_len_lst = [item[0].shape[1] for item in batch]
    max_len = max(data_len_lst)
    data = torch.zeros([len(batch), max_len]).to(device)

    for idx, item in enumerate(batch):
        waveform = item[0]
        wav_len = waveform.shape[1]
        data[idx][:wav_len] = waveform

    target = [item[1] for item in batch]
    target = torch.LongTensor(target).to(device)

    system_ids = [item[2] for item in batch]

    return [data, target, system_ids]


def remove_non_speech(wav, vad, sr, frame_sec=0.01):
    frame_len = int(sr * frame_sec)
    frame_num = wav.shape[0] // frame_len

    pure_wav = np.zeros_like(wav)
    pure_idx = 0

    for idx in range(frame_num):
        frame = wav[idx * frame_len: (idx + 1) * frame_len]

        pcm_frame = (frame * (2 ** 15)).astype(np.int32)
        pcm_frame = pcm_frame.tobytes()

        if vad.is_speech(pcm_frame, sr):
            pure_wav[pure_idx * frame_len: (pure_idx + 1) * frame_len] = frame
            pure_idx += 1

    pure_wav = pure_wav[:pure_idx * frame_len]
    return pure_wav


def extra_all_zip(data_dir):
    for zip_path in data_dir.glob("*.zip"):
        print(f"Extract all from {zip_path}")
        with ZipFile(zip_path, 'r') as _zip:
            _zip.extractall(path=data_dir)


def cat(path_1, path_2, out_path):
    with open(out_path, 'wb') as outFile:
        with open(path_1, 'rb') as f1, open(path_2, 'rb') as f2:
            outFile.write(f1.read())
            outFile.write(f2.read())


def download_and_unzip(url, out_file_path, gdrive=False):
    logging.info(f"Downloading from {url} -->> {out_file_path}")

    if gdrive is False:
        # non google drive files are downloaded using wget
        wget.download(url=url, out=str(out_file_path))
    else:
        # otherwise, we use gdown
        gdown.download(url, str(out_file_path), quiet=False)

    logging.info(f"Extract all from {out_file_path}")

    if out_file_path.suffix == ".zip":
        with ZipFile(out_file_path, 'r') as _zip:
            _zip.extractall(path=out_file_path.parent)

    elif str(out_file_path).endswith(".tar.gz"):
        with tarfile.open(out_file_path) as _tar:
            _tar.extractall(path=out_file_path.parent)

    else:
        assert False, f"Unsupported file type to extract: {out_file_path}"


def get_flag(flag_path):
    if not flag_path.exists():
        return None

    with open(flag_path, 'rb') as handle:
        flag = pickle.load(handle)
    logging.info(f"{flag_path} was created at {flag}")
    return flag


def set_flag(flag_path):
    with open(flag_path, 'wb') as handle:
        pickle.dump(datetime.now(), handle)


def pickle_save(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


def pickle_load(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def read_mp3(f, normalized=False):
    """MP3 to numpy array"""
    a = AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y


def read_audio(audio_path, expected_sr=None):
    audio_path = Path(audio_path)
    if audio_path.suffix == ".wav":
        sr, audio_np = wav.read(audio_path)

    elif audio_path.suffix == ".mp3":
        sr, audio_np = read_mp3(audio_path, normalized=True)

    else:
        audio_np, sr = librosa.load(audio_path, sr=None)

    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / (2 ** 15)

    assert len(audio_np.shape) == 1, "mono channel"
    assert audio_np.dtype == np.float32
    if expected_sr:
        assert sr == expected_sr, "sample rates must match"

    return audio_np, sr


def resample_wav(wav_in, sr, dest_sr):
    if isinstance(wav_in, Path):
        wav_in, sr = read_audio(wav_in, sr)

    resample_size = int(len(wav_in) / sr * dest_sr)
    resample = signal.resample(wav_in, resample_size)
    return resample


def cal_spec(raw_data, n_fft=2048, return_complex=False, win_length=None, hop_length=None):
    """
    return normalized log-scaled spectrogram, can be used by model
    if return_unnormalized==True, also return unnormalized log-scaled spectrogram
    """
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    # STFT
    tgt_device = device
    if raw_data.get_device() == -1:
        tgt_device = torch.device('cpu')

    D = torch.stft(raw_data, n_fft=n_fft, hop_length=hop_length,
                   win_length=win_length,
                   window=torch.hamming_window(win_length).to(tgt_device),
                   return_complex=True)

    if return_complex is True:
        return D

    # calculate the magnitude
    spec = torch.abs(D)
    spec = torch.log1p(spec)

    return spec


def inv_spec(spec, n_fft=2048, win_length=None, hop_length=None):
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    # STFT
    D = torch.istft(spec, n_fft=n_fft, hop_length=hop_length,
                    win_length=win_length,
                    window=torch.hamming_window(win_length).to(device))

    return D


def cal_masking_loss(delta, psd_max_arr, threshold_arr, window_size=2048):
    scale = 8. / 3.
    frame_step = int(window_size // 4)

    # win = torch.stft(
    #     deltas_with_noise,
    #     n_fft=window_size,
    #     hop_length=frame_step,
    #     win_length=window_size,
    #     window=torch.hann_window(window_size).to(device),
    #     center=False
    # )
    win = delta     # delta is already caculated by stft

    # calculate the magnitude
    win = win / window_size
    # win_power = win[:, :, 0] ** 2 + win[:, :, 1] ** 2
    win_power = win[..., 0] ** 2 + win[..., 1] ** 2
    win_abs = torch.sqrt(win_power)

    z = scale * win_abs     # z = scale * torch.abs(win / window_size)
    psd = z**2
    PSD = 10.0**9.6 / psd_max_arr * psd

    dim = PSD.dim()
    PSD = PSD.transpose(dim-2, dim-1)   # switch the last 2 dimensions

    # threshold_arr = torch.FloatTensor(threshold_arr).to(device)
    assert(threshold_arr.shape == PSD.shape)

    masking_loss = torch.mean(F.relu(PSD - threshold_arr))

    return masking_loss


def wav_to_other_codec(wav_path, out_path=None, format_name="mp3", bitrate="128k"):
    if out_path is None:
        out_path = wav_path.with_name(f"{wav_path.stem}_{bitrate}.{format_name}")

    # read wav file to an audio-segment
    audio_data = AudioSegment.from_wav(str(wav_path))

    # export audio segment to mp3
    audio_data.export(str(out_path), format=format_name, bitrate=bitrate)

    return out_path


def plot_hp_freqz(ir, cutoff, fs, path):
    w, h = signal.freqz(ir)
    h_dB = abs(h)

    plt.figure(figsize=(5, 5))
    plt.plot([0, cutoff, cutoff, fs / 2.0], [0, 0, 1, 1], "green", linewidth=3.0, alpha=0.5)
    plt.plot(w / max(w) * (fs / 2.0), h_dB, alpha=0.5)
    plt.ylabel('Magnitude')
    plt.xlabel(r'Frequency')
    plt.title(r'Frequency response')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_ir(cutoff, fs, type="highpass", freqz_path=None):
    assert type in ["highpass", "lowpass"]

    ir = signal.firwin(801, cutoff=cutoff, pass_zero=type, fs=fs)

    if freqz_path is not None:
        plot_hp_freqz(ir, cutoff, fs, freqz_path)

    ir = ir.astype(np.float32)
    return ir


def cal_snr(data_org, noise):
    assert data_org.shape == noise.shape

    # convert to double as precision of float is not enough
    data_org = data_org.astype(np.double)
    noise = noise.astype(np.double)

    data_sqr = np.sum(data_org**2)
    noise_sqr = np.sum(noise**2)

    if np.isclose(noise_sqr, 0):
        return 100      # avoid divided by 0

    snr_val = 10.0 * np.log10(data_sqr / noise_sqr)

    return float(snr_val)


def cal_pesq(fs, ref, deg):
    return pesq(fs=fs, ref=ref, deg=deg, mode='wb')


def hamming_encode(org_msg, org_msg_len):
    return my_hamming_code.encode(org_msg, org_msg_len)


def hamming_decode(code, code_len):
    return my_hamming_code.decode(code, code_len)































