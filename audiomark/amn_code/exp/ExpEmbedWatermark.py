import copy
import logging
import pickle
from tqdm import tqdm
from audiomark.amn_code.exp.ExpBase import ExpBase
from audiomark.amn_code.my_utils import utils
import numpy as np
from pathlib import Path

import scipy.io.wavfile as wav

from audiomark.amn_code.models import ModelTrainer
from audiomark.amn_code.my_utils.datasets import MyNumpyAudioData
from audiomark.amn_code.models.WatermarkNet import WatermarkNet


class ExpConfig(utils.ConfigBase):
    extra_audio_lst_dic = None              # extra speaker data that are directly specified

    speakers_wm_lst = None                  #
    vctk_dir = None                         #

    expected_sr = 16000                     # expected sampling rate

    audio_sec_len = 16000                   # audio section length

    benign_org_wm = None
    benign_encoded_wm = None

    wav2vec2_dir = None

    aug_normal_prob = None
    aug_normal_scale = None

    tgt_samples_ready_callback = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.extra_audio_lst_dic is not None or self.speakers_wm_lst is not None, "extra_audio_lst_dic and tgt_name cannot be both None"
        if self.speakers_wm_lst is not None:
            assert self.vctk_dir is not None, "vctk_dir must be set to the data directory if target speaker names are set."

        assert self.benign_org_wm is not None and self.benign_encoded_wm is not None

        assert self.wav2vec2_dir is not None, "set a dir to download wav2vec2 model"

        assert self.aug_normal_prob is not None and self.aug_normal_scale is not None, "need to augement with Gaussian"
        logging.info(f"aug_normal_prob = {self.aug_normal_prob:.2f}, aug_normal_scale = {self.aug_normal_scale:.2f}.")


class ExpEmbedWatermark(ExpBase):
    """

    """

    def __init__(self, out_dir, config, exp_status_fname="exp.status"):
        """
        exp_data_stat_fname: file name to save experiment data statistics
        """
        super().__init__(out_dir, config, exp_status_fname)

    def train_epoch_callback(self, epoch, train_loss, test_loss, train_epoch_callback_data):
        """
        evaluate watermark detection
        """
        if epoch % 9999 != 0:
            return

        wm_net, wav_lst, wm_lst = train_epoch_callback_data

        acc_wm_lst = []
        acc_org_lst = []

        # generate watermarked waveforms
        for wav_idx, (waveform, wm) in tqdm(enumerate(zip(wav_lst, wm_lst))):
            acc_wm, acc_org, _ = wm_net.test_wm_capability(waveform, wm)
            acc_wm_lst.append(acc_wm)
            acc_org_lst.append(acc_org)

        acc_wm_lst = np.array(acc_wm_lst)
        acc_org_lst = np.array(acc_org_lst)
        print(f" *************** accuracy on watermarked speech {acc_wm_lst.mean():.3f} ------- "
              f"accuracy on benign speech {acc_org_lst.mean():.3f} *************** ")

    def do_embed_wm(self, ckpt_dir, wav_lst, wm_lst):

        dset = MyNumpyAudioData(wav_lst, wm_lst, self.config.audio_sec_len)

        wm_net = WatermarkNet(benign_wm=self.config.benign_encoded_wm, sr=self.config.expected_sr,
                              audio_sec_len=self.config.audio_sec_len,
                              wav2vec2_dir=self.config.wav2vec2_dir,
                              aug_normal_prob=self.config.aug_normal_prob,
                              aug_normal_scale=self.config.aug_normal_scale)
        wm_net = wm_net.to(utils.device)

        wm_net.MIN_SPEECH_SPEC_POWER = 0

        # training our watermark net
        trainer_cfg = ModelTrainer.ModelTrainerConfig(
            batch_size=64,
            test_batch_size=64,
            num_workers=8,  # 8

            loss_func=WatermarkNet.wm_loss,
            loss_other_data=self,
            is_classifier=False,

            max_epochs=200,
            lr=1e-4,
            lr_gamma=1.0,
            lr_step_size=[999],

            # ckpt_dir=ckpt_dir,
            best_ckpt_dir=ckpt_dir,
            best_skip_epochs=30,

            train_epoch_callback=self.train_epoch_callback,
            train_epoch_callback_data=(wm_net, wav_lst, wm_lst),
        )

        trainer = ModelTrainer.ModelTrainer(
            model=wm_net,
            train_set=dset, test_set=dset, config=trainer_cfg
        )
        trainer.run()

        ########################################################
        # use the best model for evaluation
        dic_saved = trainer.load_latest_ckpt(ckpt_dir)

        wm_net = WatermarkNet(self.config.benign_encoded_wm, self.config.expected_sr, self.config.audio_sec_len,
                              wav2vec2_dir=self.config.wav2vec2_dir,
                              aug_normal_prob=self.config.aug_normal_prob, aug_normal_scale=self.config.aug_normal_scale)
        wm_net = wm_net.to(utils.device)
        wm_net.load_state_dict(dic_saved["model_state"])
        wm_net.eval()

        # then generate watermarked waveforms
        acc_wm_lst = []
        acc_org_lst = []

        for wav_idx, (waveform, wm) in tqdm(enumerate(zip(wav_lst, wm_lst))):
            # logging.info(f"watermarking wav_idx = {wav_idx}...")
            acc_wm, acc_benign, wm_waveform = wm_net.test_wm_capability(waveform, wm)
            acc_wm_lst.append(acc_wm)
            acc_org_lst.append(acc_benign)

            # save the watermarked back to the list
            wm_waveform = wm_waveform.flatten().cpu().detach().numpy()
            assert len(waveform) <= len(wm_waveform)
            wav_lst[wav_idx] = wm_waveform[: len(waveform)]

        acc_wm_lst = np.array(acc_wm_lst)
        acc_org_lst = np.array(acc_org_lst)
        print(f" *************** accuracy on watermarked speech {acc_wm_lst.mean():.3f} ------- "
              f"accuracy on benign speech {acc_org_lst.mean():.3f} *************** ")

        return wav_lst

    def load_vctk_all_train_samples(self):
        # Coqui TTS is only needed by the original AudioMarkNet training path,
        # not by checkpoint-based generation or validation.
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.datasets import load_tts_samples

        # init configs
        vctk_config = BaseDatasetConfig(
            formatter="vctk",
            dataset_name="vctk",
            meta_file_train="",
            meta_file_val="",
            path=str(self.config.vctk_dir),
            language="en",
        )
        train_samples, eval_samples = load_tts_samples(vctk_config, eval_split=False)

        assert eval_samples is None

        return train_samples

    def get_tgt_samples(self, wm_dir, train_samples, tgt_name):

        tgt_samples = []
        for sample in train_samples:
            if sample["speaker_name"] == tgt_name:
                tgt_samples.append(sample)
                sample["audio_file_wm"] = wm_dir.joinpath(Path(sample["audio_file"]).stem + ".wav")

        return tgt_samples

    def get_extra_samples(self, wm_dir):
        tgt_samples = []
        if self.config.extra_audio_lst_dic is not None:
            # add extra data
            for extra_dic in self.config.extra_audio_lst_dic:
                wm_dic = copy.copy(extra_dic)
                wm_dic["audio_file_wm"] = wm_dir.joinpath(wm_dic["audio_file"].stem + ".wav")
                tgt_samples.append(wm_dic)

        return tgt_samples

    def embed_wm(self):
        """
        embed watermarks into every audio in the config
        """
        samples_path = self.out_dir.joinpath("tgt_samples.bin")
        flag_path = self.out_dir.joinpath("embed_wm_done.flag")
        if utils.get_flag(flag_path):
            assert samples_path.exists(), "samples must have been saved before."
            return

        tgt_samples = []

        train_samples = self.load_vctk_all_train_samples()

        for data_dic in self.config.speakers_wm_lst:
            speaker_name, encoded_wm = data_dic["speaker"], data_dic["encoded_wm"]
            wm_dir = self.out_dir.joinpath(speaker_name)
            Path.mkdir(wm_dir, exist_ok=True)
            speaker_samples = self.get_tgt_samples(wm_dir, train_samples, speaker_name)

            if len(speaker_samples) == 0:
                # data should come from extra list
                speaker_samples = self.get_extra_samples(wm_dir)
                assert len(speaker_samples) > 0, f"cannot find any data for {speaker_name}"

            for sample in speaker_samples:
                tgt_samples.append([sample, encoded_wm])


        if self.config.tgt_samples_ready_callback is not None:
            # Give a chance to modify tgt_samples. Useful for pirate attack
            self.config.tgt_samples_ready_callback(tgt_samples)

        # read all the audio files
        wav_lst = []
        encoded_wm_lst = []
        audio_with_wm_path_list = []
        for sample, encoded_wm in tgt_samples:
            audio_path = sample["audio_file"]

            # waveform = self.speaker_manager.encoder_ap.load_wav(audio_path, sr=self.speaker_manager.encoder_ap.sample_rate)
            waveform, sr = utils.read_audio(audio_path, expected_sr=self.config.expected_sr)
            wav_lst.append(waveform)
            encoded_wm_lst.append(encoded_wm)

            audio_with_wm_path_list.append(sample["audio_file_wm"])

        wm_wav_lst = self.do_embed_wm(self.out_dir.joinpath("ckpt"), wav_lst, encoded_wm_lst)

        # save all the waveforms
        for wm_waveform, wm_path in zip(wm_wav_lst, audio_with_wm_path_list):
            wav.write(wm_path, self.config.expected_sr, wm_waveform)

        utils.set_flag(flag_path)

        # save the samples for other experiments to use
        with open(samples_path, 'wb') as handle:
            pickle.dump(tgt_samples, handle)

        return

    def run(self):
        self.embed_wm()


# some helper functions related to watermarking experiments
def load_existing_wm_exp_data(wm_exp_dir, expected_sr=16000):
    # load previously completed experiments
    our_saved_path = wm_exp_dir.joinpath(f"tgt_samples.bin")

    with open(our_saved_path, 'rb') as handle:
        our_tgt_samples = pickle.load(handle)

    speaker_audio_wm_dic = {}
    org_audio_len_arr = []

    for our_sample, our_wm in tqdm(our_tgt_samples):
        speaker_name = our_sample["speaker_name"]
        if speaker_name not in speaker_audio_wm_dic:
            speaker_audio_wm_dic[speaker_name] = []

        org_audio_path = Path(our_sample["audio_file"])
        assert org_audio_path.suffix == ".flac", "original audio files are .flac"
        assert our_sample["audio_file_wm"].suffix == ".wav", "our watermarked files are .wav"

        our_wm_audio_path = our_sample["audio_file_wm"]

        org_audio, _ = utils.read_audio(org_audio_path, expected_sr)
        our_wm_audio, _ = utils.read_audio(our_wm_audio_path, expected_sr)

        org_audio_len_arr.append(len(org_audio))

        speaker_audio_wm_dic[speaker_name].append({
            "org_audio": org_audio, "org_audio_path": org_audio_path,
            "our_wm_audio": our_wm_audio, "our_wm_audio_path": our_wm_audio_path,
            "our_wm": our_wm,

        })

    print(f"avg org audio len = {np.mean(org_audio_len_arr) / expected_sr:.2f}")

    return speaker_audio_wm_dic


def get_existing_wm_samples(tgt_name, exp_dir, wm_ratio=1.0):
    """
    Original file path will be replaced by watermarked file path.
    wm_ratio: the ratio of watermarked audio in the return set.
        1.0 -> all returned audios are watermarked
        0.1 -> 10% returned audios are watermarked
    """

    sample_saved_path = exp_dir.joinpath(f"tgt_samples.bin")
    assert sample_saved_path.exists(), f"{exp_dir.name} must be run before"

    with open(sample_saved_path, 'rb') as handle:
        tgt_samples = pickle.load(handle)

    # first get the number of audios for this speaker
    num_samples = 0
    for sample, wm in tgt_samples:
        if sample["speaker_name"] == tgt_name:
            num_samples += 1
    wm_num = int(num_samples * wm_ratio)    # number of watermarked audio
    redirect_idx_arr = np.random.RandomState(seed=3783511).permutation(num_samples)
    redirect_idx_arr = redirect_idx_arr[:wm_num]
    redirect_num = 0

    filtered_samples = []
    for sample, wm in tgt_samples:
        if sample["speaker_name"] != tgt_name:
            continue

        # assert Path(sample["audio_file"]).suffix == ".flac", "original audio files are .flac"
        assert sample["audio_file_wm"].suffix == ".wav", "our watermarked files are .wav"

        sample["audio_file_org"] = sample["audio_file"]

        if len(filtered_samples) in redirect_idx_arr:
            # redirect the audio file to watermarked file for speaker adaptation
            sample["audio_file"] = sample["audio_file_wm"]
            redirect_num += 1       # for checking purpose

        filtered_samples.append(sample)

    assert len(filtered_samples) == num_samples and redirect_num == wm_num
    return filtered_samples

