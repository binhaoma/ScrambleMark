import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from collaborative_watermarking_with_codecs.src.collaborative_watermarking.third_party.hifi_gan.env import AttrDict, build_env
from collaborative_watermarking_with_codecs.src.collaborative_watermarking.third_party.hifi_gan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss

from collaborative_watermarking_with_codecs.src.collaborative_watermarking.third_party.hifi_gan.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

from collaborative_watermarking_with_codecs.src.collaborative_watermarking.meldataset import MelDataset, mel_spectrogram, get_filelist

# from collaborative_watermarking_with_codecs.src.collaborative_watermarking.augmentation import get_augmentations

import torchaudio

import random

torch.backends.cudnn.benchmark = True

SUPPORTED_AUDIO_EXTENSIONS = ('.wav', '.flac', '.mp3', '.ogg')


def discover_audio_files(input_path):
    """Return audio files from a file or a non-recursive directory scan."""
    input_path = os.fspath(input_path)

    if os.path.isfile(input_path):
        if input_path.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
            return [input_path]
        raise ValueError(f'Unsupported audio file: {input_path}')

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f'Input audio path does not exist: {input_path}')

    filelist = sorted(
        os.path.join(input_path, filename)
        for filename in os.listdir(input_path)
        if filename.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)
        and os.path.isfile(os.path.join(input_path, filename))
    )
    if not filelist:
        raise ValueError(f'No supported audio files found in: {input_path}')

    return filelist


def render(rank, a, h):

    print(f"Arguments: {a}")

    print(f"Config: {h}")

    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        device = torch.device('cpu')

    generator = Generator(h)
    generator = generator.to(device)

    print(generator)
    print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')

    if cp_g is None:
        state_dict_do = None
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)

    if a.input_filelist is None:
        filelist = discover_audio_files(a.input_wavs_dir)
    else:
        filelist = get_filelist(
            filelist_path=a.input_filelist,
            wavs_dir=a.input_wavs_dir,
            ext=a.wavefile_ext)

    if a.max_files is not None:
        # suffle the filelist
        filelist = sorted(filelist)
        torch.manual_seed(42)
        perm = torch.randperm(len(filelist))
        filelist = [filelist[i] for i in perm[:a.max_files]]
        filelist = sorted(filelist)

    dataset = MelDataset(
        filelist,
        segment_size=h.segment_size,
        n_fft=h.n_fft, num_mels=h.num_mels,
        hop_size=h.hop_size, win_size=h.win_size,
        sampling_rate=h.sampling_rate,
        fmin=h.fmin, fmax=h.fmax,
        split=False, shuffle=False, 
        n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
        device=device, fine_tuning=False)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, shuffle=False,
        sampler=None, batch_size=1, pin_memory=True,
        drop_last=True)

    os.makedirs(a.output_wavs_dir, exist_ok=True)
    if a.original_wavs_dir is not None:
        os.makedirs(a.original_wavs_dir, exist_ok=True)

    generator.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            x, y, filename, y_mel = batch
            y_g_hat = generator(x.to(device))

            # save audio
            basename =  os.path.basename(filename[0])
            output_filename = os.path.join(a.output_wavs_dir, basename)
            torchaudio.save(output_filename, y_g_hat[0,:,:].cpu(), h.sampling_rate)
            print(f"Saved {output_filename} ({j+1}/{len(dataloader)})")

            if a.original_wavs_dir is not None:
                torchaudio.save(os.path.join(a.original_wavs_dir, basename), y[:,:].cpu(), h.sampling_rate)
                print(f"Saved {os.path.join(a.original_wavs_dir, basename)}")

    print("Finished rendering") 


def render_hifigan(
        config,
        input_wavs_dir='LJSpeech-1.1/wavs',
        output_wavs_dir='output',
        input_filelist=None,
        checkpoint_path='cp_hifigan',
        original_wavs_dir=None,
        wavefile_ext='.wav',
        seed=42,
        max_files=None,
        group_name=None):
    """Render watermarked audio from a file/directory or an optional filelist."""
    a = argparse.Namespace(
        group_name=group_name,
        config=str(config),
        input_wavs_dir=str(input_wavs_dir),
        output_wavs_dir=str(output_wavs_dir),
        original_wavs_dir=(
            None if original_wavs_dir is None else str(original_wavs_dir)
        ),
        input_filelist=(
            None if input_filelist is None else str(input_filelist)
        ),
        checkpoint_path=str(checkpoint_path),
        wavefile_ext=wavefile_ext,
        seed=seed,
        max_files=max_files,
    )

    config_path = os.path.join(a.config)
    with open(config_path) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)

    if h.num_gpus > 1:
        mp.spawn(render, nprocs=h.num_gpus, args=(a, h,))
    else:
        render(0, a, h)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--config')
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--output_wavs_dir', default='output')
    parser.add_argument('--original_wavs_dir', default=None)
    parser.add_argument('--input_filelist', '--input_file', dest='input_filelist')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    # parser.add_argument('--config', default='')
    parser.add_argument('--wavefile_ext', default='.wav', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_files', default=None, type=int)

    a = parser.parse_args()
    render_hifigan(**vars(a))


if __name__ == '__main__':
    main()
