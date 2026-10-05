import sys

from tqdm import tqdm

import exp_setup
from my_utils import utils
import scipy.io.wavfile as wav
from pathlib import Path
import numpy as np


tgt_sr = 44100


def combine_vctk_wm_audio(exp_dir, speaker_name, wm_exp_name, audio_secs):

    if wm_exp_name is not None:
        # combining watermarked data
        audio_dir = Path(exp_cfg.out_dir).joinpath(f"{wm_exp_name}/{speaker_name}")
        path_finder = "*.wav"
    else:
        # combining original data
        if speaker_name == "Obama":
            audio_dir = Path(exp_cfg.data_dir).joinpath(f"CopyrightFreeSpeeches/Obama/barackobamatransitionaddress7")
            path_finder = "*.wav"

        else:
            _pos = speaker_name.find("_")
            speaker_id = speaker_name[_pos + 1:]

            audio_dir = Path(exp_cfg.data_dir).joinpath(f"vctk/wav48_silence_trimmed/{speaker_id}")
            path_finder = "*mic1.flac"

    # get all audio file paths
    all_audio_path_lst = list(audio_dir.glob(path_finder))
    all_audio_path_lst.sort()

    max_len = exp_cfg.sr * audio_secs
    full_audio = np.zeros(max_len + exp_cfg.sr * 60).astype(np.float32)     # make the buffer slightly larger
    cur_pos = 0

    # load audio and fill in the output audio
    for audio_path in all_audio_path_lst:
        waveform, sr = utils.read_audio(audio_path, expected_sr=exp_cfg.sr)
        wav_len = len(waveform)

        full_audio[cur_pos: cur_pos + wav_len] = waveform

        cur_pos += wav_len

        if cur_pos > max_len:
            break

    assert cur_pos < len(full_audio)
    full_audio = full_audio[: cur_pos]

    exp_dir.mkdir(exist_ok=True)

    out_path = exp_dir.joinpath(f"{speaker_name}_combined_wm_{wm_exp_name}.wav")
    wav.write(out_path, exp_cfg.sr, full_audio)

    full_audio_resampled = utils.resample_wav(full_audio, exp_cfg.sr, tgt_sr)
    wav.write(out_path.with_stem(f"{out_path.stem}_{tgt_sr}"), tgt_sr, full_audio_resampled)


def main():
    speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, exp_cfg.wm_length)
    speakers_names_lst = [x["speaker"] for x in speakers_wm_lst]

    exp_dir = Path(exp_cfg.out_dir).joinpath("ExpPrepareForOnline")
    exp_dir.mkdir(exist_ok=True)

    audio_secs_lst = [270, ]

    for audio_secs in audio_secs_lst:

        for speaker_name in tqdm(speakers_names_lst):

            save_dir = exp_dir.joinpath(f"{audio_secs}_seconds")

            combine_vctk_wm_audio(exp_dir=save_dir, speaker_name=speaker_name, wm_exp_name=None,
                                  audio_secs=audio_secs)

            combine_vctk_wm_audio(exp_dir=save_dir, speaker_name=speaker_name, wm_exp_name="ExpEmbedWatermark",
                                  audio_secs=audio_secs)


if __name__ == "__main__":
    exp_cfg = exp_setup.init_config()

    main()
    sys.exit()



