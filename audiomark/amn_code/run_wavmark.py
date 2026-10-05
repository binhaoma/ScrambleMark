from __future__ import print_function

import pickle
import sys

from tqdm import tqdm

import exp_setup
from pathlib import Path
from exp.ExpEmbedWatermark import ExpEmbedWatermark, ExpConfig as ExpEmbedWatermarkCfg
import numpy as np
import torch
import src.wavmark as wavmark
from my_utils import my_utils
import scipy.io.wavfile as wav


def prepare_wm_samples_path(exp_dir, tgt_name):

    sample_saved_path = Path(exp_cfg.out_dir).joinpath(f"ExpEmbedWatermark/tgt_samples.bin")
    assert sample_saved_path.exists(), "ExpEmbedWatermark must be run before"

    with open(sample_saved_path, 'rb') as handle:
        tgt_samples = pickle.load(handle)

    filtered_samples = []
    for sample, wm in tgt_samples:
        if sample["speaker_name"] != tgt_name:
            continue

        assert Path(sample["audio_file"]).suffix == ".flac", "original audio files are .flac"
        assert sample["audio_file_wm"].suffix == ".wav", "existing watermarked files are .wav"

        # need to change the "audio_file_wm" to our new watermarked file
        sample["audio_file_wm"] = exp_dir.joinpath(tgt_name).joinpath(sample["audio_file_wm"].name)

        filtered_samples.append(sample)

    return filtered_samples


def embed_wm(org_audio_path, wm_path, wm):
    assert Path(org_audio_path).suffix == ".flac"

    sr = 16000

    # 1.load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)

    # 2.create 16-bit payload
    # payload = np.random.choice([0, 1], size=16)
    # print("Payload:", payload)
    payload = wm

    # 3.read host audio
    # the audio should be a single-channel 16kHz wav, you can read it using soundfile:
    signal, sample_rate = utils.read_audio(org_audio_path, sr)
    # Otherwise, you can use the following function to convert the host audio to single-channel 16kHz format:
    # from wavmark.utils import file_reader
    # signal = file_reader.read_as_single_channel("example.wav", aim_sr=16000)

    # 4.encode watermark
    watermarked_signal, _ = wavmark.encode_watermark(model, signal, payload, show_progress=False)
    # you can save it as a new wav:
    Path.mkdir(wm_path.parent, exist_ok=True, parents=True)
    wav.write(wm_path, sr, watermarked_signal)

    # 5.decode watermark
    payload_decoded, _ = wavmark.decode_watermark(model, watermarked_signal, show_progress=False)
    BER = (payload != payload_decoded).mean() * 100

    # print(f"{wav_path.name} Decode BER: {BER:.1f}")
    return BER


def main():
    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, wm_len=16)

    # using WavMark to embed messages into original speach
    exp_dir = Path(exp_cfg.out_dir).joinpath("ExpWavMark")
    samples_path = exp_dir.joinpath("tgt_samples.bin")

    flag_path = exp_dir.joinpath("exp_done.flag")
    if utils.get_flag(flag_path):
        assert samples_path.exists(), "samples must have been saved before."
        return

    tgt_samples = []

    for idx, (data_dic) in enumerate(speakers_wm_lst):
        speaker_name, wm = data_dic["speaker"], data_dic["org_wm"]

        samples_lst = prepare_wm_samples_path(exp_dir, speaker_name)

        print(f"\n******* Start handling speaker = {speaker_name} ({idx+1} / {len(speakers_wm_lst)}) **********\n")
        ber_lst = []
        pbar = tqdm(samples_lst)
        for sample in pbar:
            BER = embed_wm(org_audio_path=sample["audio_file"], wm_path=sample["audio_file_wm"], wm=wm)
            ber_lst.append(BER)

            pbar.set_description(f"{speaker_name} Decode BER: {np.mean(ber_lst):.1f}")

            tgt_samples.append([sample, wm])

    # save the samples for other experiments to use
    with open(samples_path, 'wb') as handle:
        pickle.dump(tgt_samples, handle)

    utils.set_flag(flag_path)


if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    main()

    sys.exit()




