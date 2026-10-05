import os
import numpy as np
import soundfile as sf

import os
import torch
import yaml
import logging
import argparse
import warnings
import numpy as np
import scipy.signal as signal
from torch.nn.functional import mse_loss
import random
import pdb
import argparse

warnings.filterwarnings("ignore")
import torchaudio
# set seeds
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)




def pitch_modulation_square(input_file, output_file, mod_freq=5, depth=0.2):

    wav, sr = torchaudio.load(f'{input_file}')
    sample_rate = sr
    waveform = wav[0]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    # 生成时间轴
    t = np.linspace(0, len(waveform) / sample_rate, num=len(waveform), endpoint=False)

    # 生成方波 (mod_freq Hz)
    square_wave = signal.square(2 * np.pi * mod_freq * t)

    # 计算时间偏移 (方波决定音高变化)
    pitch_variation = 1 + (square_wave * depth)  # 当方波为1时提升pitch，为-1时降低
    modulated_time = np.cumsum(pitch_variation) / sample_rate  # 计算变调后新的时间索引, 时间索引是需要累加的

    # 进行插值，调整音高
    modulated_waveform = np.interp(modulated_time, np.linspace(0, len(waveform)/sample_rate, num=len(waveform)), waveform, left=0, right=0)

    square_wave_alt = torch.tensor(modulated_waveform,dtype=torch.float32)
    print(output_file)
    torchaudio.save(f"{output_file}.wav", square_wave_alt.unsqueeze(0), sr)


def process_folder(source_folder, target_folder, mod_freq=1, depth=0.05):


    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith(('.wav', '.flac', '.aiff')):
            input_file = os.path.join(source_folder, file_name)
            output_file = os.path.join(target_folder, 'square_'+os.path.splitext(file_name)[0])
            print(f"Processing {input_file}...")
            pitch_modulation_square(input_file, output_file, mod_freq, depth)
            print(f"Saved to {output_file}")
