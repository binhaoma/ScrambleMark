import pickle
from pathlib import Path

import logging
from tqdm import tqdm
import scipy.io.wavfile as wav
from exp.ExpBase import ExpBase
from my_utils import utils
from exp.ExpEmbedWatermark import load_existing_wm_exp_data
from models import ModelTrainer
from models.WatermarkNet import WatermarkNet
from models.AutoEncoder import AutoEncoder, ae_denoise_loss
from my_utils.datasets import PairedNumpyAudioData
import torch
from torch.nn import functional as F
import numpy as np
from run_evaluate import eval_metrics


class ExpConfig(utils.ConfigBase):

    wav2vec2_dir = None             # used to initialize wav2vec2 model

    train_speakers_wm_lst = None
    train_wm_exp_dir = None        # dir for training data, which should be different speakers
    train_benign_encoded_wm = None

    tgt_speakers_wm_lst = None
    tgt_wm_exp_dir = None          # dir for target watermarked audio
    tgt_benign_encoded_wm = None    #

    sr = None
    audio_sec_len = None

    # config for the autoencoder
    encoder_cfg = [64, 64, "M", 128, 128, "M", 256, "M", 256, "M", 512, "M"]
    decoder_cfg = ["M", 256, "M", 256, "M", 128, 128, "M", 64, 64, "M", 2]
    in_shape = [2]  # only the first dimension is useful
    features_shape = [512, 1, 4]


def collate_fn(data, min_freq_idx, max_freq_idx):
    # x contains a list of tuples, each tuple is [src_slice, tgt_slice]
    x = torch.stack([d[0] for d in data])
    y = torch.stack([d[1] for d in data])

    x = WatermarkNet.spec_for_reconstruct(x)[..., min_freq_idx: max_freq_idx, :]
    y = WatermarkNet.spec_for_reconstruct(y)[..., min_freq_idx: max_freq_idx, :]

    return x, y


class ExpAdaptiveAutoEncoder(ExpBase):
    """
    training the backdoor net
    """
    def __init__(self, out_dir, config,  exp_status_fname="exp.status"):
        """
        exp_data_stat_fname: file name to save experiment data statistics
        """
        super().__init__(out_dir, config, exp_status_fname)

    def save_status(self):
        super().save_status()

    def prepare_data(self):

        train_speaker_audio_wm_dic = load_existing_wm_exp_data(self.config.train_wm_exp_dir)
        tgt_speaker_audio_wm_dic = load_existing_wm_exp_data(self.config.tgt_wm_exp_dir)

        return train_speaker_audio_wm_dic, tgt_speaker_audio_wm_dic

    def cal_freq_idx_range(self):
        max_freq = self.config.sr // 2
        spec_dim_enc = WatermarkNet.N_FFT // 2 + 1
        max_freq_idx = int(spec_dim_enc * WatermarkNet.DELTA_MAX_FREQ / max_freq)
        min_freq_idx = int(spec_dim_enc * WatermarkNet.DELTA_MIN_FREQ / max_freq)

        return min_freq_idx, max_freq_idx

    def train_ae(self, speaker_audio_wm_dic, ckpt_dir):
        flag_path = self.out_dir.joinpath("train_ae_done.flag")
        if utils.get_flag(flag_path):
            return

        def get_all_audio(k):
            audio_lst = []
            for speaker, audio_wm_lst in speaker_audio_wm_dic.items():
                speaker_audio_lst = []
                for audio_wm_dic in audio_wm_lst:
                    speaker_audio_lst.append(audio_wm_dic[k])

                audio_lst.extend(speaker_audio_lst)
            return audio_lst

        # use a de-noise autoencoder to remove our watermarks
        dset = PairedNumpyAudioData(src_data_lst=get_all_audio("our_wm_audio"),
                                    tgt_data_lst=get_all_audio("org_audio"),
                                    expected_x_len=self.config.audio_sec_len,
                                    )

        ae = AutoEncoder(encoder_cfg=self.config.encoder_cfg, decoder_cfg=self.config.decoder_cfg,
                         in_shape=self.config.in_shape, features_shape=self.config.features_shape)
        ae = ae.to(utils.device)

        min_freq_idx, max_freq_idx = self.cal_freq_idx_range()

        trainer_cfg = ModelTrainer.ModelTrainerConfig(
            batch_size=64,
            test_batch_size=64,
            num_workers=12,  # 8

            collate_fn=lambda _v: collate_fn(_v, min_freq_idx=min_freq_idx, max_freq_idx=max_freq_idx),  # transform waveform to specs
            pin_memory=False,

            loss_func=ae_denoise_loss,
            is_classifier=False,

            max_epochs=200,
            lr=1e-4,
            lr_gamma=1.0,
            lr_step_size=[999],

            # ckpt_dir=ckpt_dir,
            best_ckpt_dir=ckpt_dir,
            # best_skip_epochs=30,
        )

        trainer = ModelTrainer.ModelTrainer(
            model=ae,
            train_set=dset, test_set=dset, config=trainer_cfg
        )
        trainer.run()

        utils.set_flag(flag_path)

    def remove_watermark(self, purified_dir, speaker_audio_wm_dic, ckpt_dir):

        def get_purified_fname(_org_audio_path):
            return _org_audio_path.stem+"_purified.wav"

        def print_stat_dic(_stat_dic):
            _pesq_wm_lst, _snr_wm_lst = _stat_dic["pesq_wm_lst"], _stat_dic["snr_wm_lst"]
            _pesq_purified_lst, _snr_purified_lst = _stat_dic["pesq_purified_lst"], _stat_dic["snr_purified_lst"]

            with open(purified_dir.joinpath("quality_compare.txt"), "w") as f:
                print(f"{purified_dir.name}:\n"
                      f"pesq_wm = {np.mean(_pesq_wm_lst):.2f} +- {np.std(_pesq_wm_lst):.2f};\t"
                      f"snr_wm = {np.mean(_snr_wm_lst):.2f} +- {np.std(_snr_wm_lst):.2f}\n"
                      f"pesq_purified = {np.mean(_pesq_purified_lst):.2f} +- {np.std(_pesq_purified_lst):.2f};\t"
                      f"snr_purified = {np.mean(_snr_purified_lst):.2f} +- {np.std(_snr_purified_lst):.2f}",
                      file=f)

        stat_path = purified_dir.joinpath("stat.bin")
        flag_path = purified_dir.joinpath("remove_watermark_done.flag")

        if utils.get_flag(flag_path):
            with open(stat_path, 'rb') as handle:
                stat_dic = pickle.load(handle)
                print_stat_dic(stat_dic)

            # purification has been done before. Directly read purified audio
            for speaker, audio_wm_lst in speaker_audio_wm_dic.items():
                for audio_wm_dic in audio_wm_lst:
                    # save the purified waveform
                    speaker_dir = purified_dir.joinpath(speaker)
                    purified_path = speaker_dir.joinpath(get_purified_fname(audio_wm_dic["org_audio_path"]))
                    purified_wav, _ = utils.read_audio(purified_path, self.config.sr)

                    audio_wm_dic["purified_audio_path"] = purified_path
                    audio_wm_dic["purified_audio"] = purified_wav
            return

        # use the trained ae to remove watermarks
        dic_saved = ModelTrainer.ModelTrainer.load_latest_ckpt(ckpt_dir)
        ae = AutoEncoder(encoder_cfg=self.config.encoder_cfg, decoder_cfg=self.config.decoder_cfg,
                         in_shape=self.config.in_shape, features_shape=self.config.features_shape)
        ae.load_state_dict(dic_saved["model_state"])
        ae = ae.to(utils.device)
        ae.eval()

        pesq_purified_lst, snr_purified_lst, pesq_wm_lst, snr_wm_lst = [], [], [], []

        min_freq_idx, max_freq_idx = self.cal_freq_idx_range()
        wm_net = WatermarkNet(np.zeros(5), self.config.sr, self.config.audio_sec_len,
                              wav2vec2_dir=self.config.wav2vec2_dir)   # just for split waveforms

        for speaker_idx, (speaker, audio_wm_lst) in enumerate(speaker_audio_wm_dic.items()):
            # for each waveform, we split it into sections and use autoencoder to remove watermarks from each section
            print(f"Purifying speaker: {speaker} ({speaker_idx+1} / {len(speaker_audio_wm_dic)}):")
            for audio_wm_dic in tqdm(audio_wm_lst):
                wm_audio = audio_wm_dic["our_wm_audio"]

                wav_secs = wm_net.split_waveform(wm_audio)
                spec_secs = WatermarkNet.spec_for_reconstruct(wav_secs)

                # the ae only cares part of freq
                partial_spec_secs = spec_secs[..., min_freq_idx: max_freq_idx, :]
                purified_partial_spec_secs = partial_spec_secs + ae(partial_spec_secs)

                # replace original partial frequency with purified frequency
                purified_spec_secs = spec_secs.clone()
                purified_spec_secs[..., min_freq_idx: max_freq_idx, :] = purified_partial_spec_secs

                # recover spectrograms back to time domain
                purified_spec_secs = utils.inv_spec(WatermarkNet.recover_compelx(purified_spec_secs),
                                                    n_fft=WatermarkNet.N_FFT, hop_length=WatermarkNet.HOP_LEN)
                purified_spec_secs = torch.clip(purified_spec_secs, -1.0, 1.0)
                if purified_spec_secs.shape != wav_secs.shape:
                    # need to pad x_prime if inverse fft does not preserve the length
                    assert purified_spec_secs.shape[1] < wav_secs.shape[1], "only consider the case that the signal is shorter"
                    diff = wav_secs.shape[1] - purified_spec_secs.shape[1]
                    purified_spec_secs = F.pad(purified_spec_secs, (0, diff))
                    assert purified_spec_secs.shape == wav_secs.shape

                # flatten all sections into a long waveform
                purified_wav = purified_spec_secs.flatten().cpu().detach().numpy()
                assert len(wm_audio) <= len(purified_wav)

                purified_wav = purified_wav[: len(wm_audio)]

                # before we store the purified data, let's calculate some metrics
                org_wav = audio_wm_dic["org_audio"]
                wm_wav = audio_wm_dic["our_wm_audio"]

                pesq_wm_lst.append(utils.cal_pesq(fs=self.config.sr, ref=org_wav, deg=wm_wav))
                snr_wm_lst.append(utils.cal_snr(data_org=org_wav, noise=org_wav - wm_wav))

                pesq_purified_lst.append(utils.cal_pesq(fs=self.config.sr, ref=org_wav, deg=purified_wav))
                snr_purified_lst.append(utils.cal_snr(data_org=org_wav, noise=org_wav-purified_wav))

                audio_wm_dic["purified_audio"] = purified_wav

                # save the purified waveform
                speaker_dir = purified_dir.joinpath(speaker)
                Path.mkdir(speaker_dir, parents=True, exist_ok=True)
                audio_wm_dic["purified_audio_path"] = speaker_dir.joinpath(get_purified_fname(audio_wm_dic["org_audio_path"]))
                wav.write(audio_wm_dic["purified_audio_path"], self.config.sr, purified_wav)

        # save statistic results
        stat_dic = {
            "pesq_wm_lst": pesq_wm_lst, "snr_wm_lst": snr_wm_lst,
            "pesq_purified_lst": pesq_purified_lst, "snr_purified_lst": snr_purified_lst,
        }
        print_stat_dic(stat_dic)
        with open(stat_path, 'wb') as handle:
            pickle.dump(stat_dic, handle)

        utils.set_flag(flag_path)

    def eval_wm(self, eval_dir, wav_net_ckpt, benign_encoded_wm, speaker_audio_wm_dic, speakers_wm_lst):

        # load our watermark net
        wm_net = WatermarkNet(benign_encoded_wm, self.config.sr, self.config.audio_sec_len,
                              wav2vec2_dir=self.config.wav2vec2_dir)
        dic_saved = ModelTrainer.ModelTrainer.load_latest_ckpt(wav_net_ckpt)
        wm_net.load_state_dict(dic_saved["model_state"])
        wm_net = wm_net.to(utils.device)
        wm_net.eval()

        # directly use the evaluation code
        eval_metrics(exp_dir=eval_dir.with_name(f"{eval_dir.name}_wm"),
                     speaker_audio_wm_dic=speaker_audio_wm_dic,
                     wm_net=wm_net, wavmark_net=None, speakers_wm_lst=speakers_wm_lst)

        # replace org_audio with purified_audio to be compatible with evaluation code
        speaker_audio_wm_dic_clone = speaker_audio_wm_dic.copy()

        for speaker, audio_wm_lst in speaker_audio_wm_dic_clone.items():
            for audio_wm_dic in audio_wm_lst:
                audio_wm_dic["our_wm_audio"] = audio_wm_dic["purified_audio"]

        eval_metrics(exp_dir=eval_dir.with_name(f"{eval_dir.name}_purified"),
                     speaker_audio_wm_dic=speaker_audio_wm_dic_clone,
                     wm_net=wm_net, wavmark_net=None, speakers_wm_lst=speakers_wm_lst)

    def combine_purified_tgt(self, exp_dir, tgt_purified_dir):

        flag_path = exp_dir.joinpath("combine_purified_tgt_done.flag")
        if utils.get_flag(flag_path):
            return

        def do_combine(speaker_name):
            audio_secs = 270
            tgt_sr = 44100

            # combining watermarked data
            audio_dir = Path(tgt_purified_dir).joinpath(f"{speaker_name}")
            path_finder = "*.wav"

            # get all audio file paths
            all_audio_path_lst = list(audio_dir.glob(path_finder))
            all_audio_path_lst.sort()

            max_len = self.config.sr * audio_secs
            full_audio = np.zeros(max_len + self.config.sr * 60).astype(np.float32)  # make the buffer slightly larger
            cur_pos = 0

            # load audio and fill in the output audio
            for audio_path in all_audio_path_lst:
                waveform, sr = utils.read_audio(audio_path, expected_sr=self.config.sr)
                wav_len = len(waveform)

                full_audio[cur_pos: cur_pos + wav_len] = waveform

                cur_pos += wav_len

                if cur_pos > max_len:
                    break

            assert cur_pos < len(full_audio)
            full_audio = full_audio[: cur_pos]

            exp_dir.mkdir(exist_ok=True)

            out_path = exp_dir.joinpath(f"{speaker_name}_combined_wm_purified.wav")
            wav.write(out_path, self.config.sr, full_audio)

            full_audio_resampled = utils.resample_wav(full_audio, self.config.sr, tgt_sr)
            wav.write(out_path.with_stem(f"{out_path.stem}_{tgt_sr}"), tgt_sr, full_audio_resampled)

        speaker_name_lst = [x["speaker"] for x in self.config.tgt_speakers_wm_lst]
        for name in speaker_name_lst:
            do_combine(name)

        utils.set_flag(flag_path)

    def load_fake_speech_for_eval(self):

        data_dir = Path(self.out_dir).joinpath("playht_purified_tgt")

        speaker_lst = [x["speaker"] for x in self.config.tgt_speakers_wm_lst]

        speaker_audio_wm_dic = {}
        for speaker_name in speaker_lst:
            # get all the fake speech for this speaker
            speaker_audio_wm_dic[speaker_name] = []

            # original speech are voice cloned without watermarks

            def read_audio_may_resample(_audio_path):
                _audio, _sr = utils.read_audio(_audio_path, None)
                if _sr != self.config.sr:
                    logging.info(f"Resampling {_audio_path.name} from {_sr} to {self.config.sr}.")
                    _audio = utils.resample_wav(_audio, _sr, self.config.sr)
                return _audio

            audio_path = data_dir.joinpath(f"{speaker_name}_wm_purified_playht.wav")
            if not audio_path.exists():
                # cannot find audio, just return
                return None

            our_wm_audio = read_audio_may_resample(audio_path)

            # find corresponding wm
            tgt_wm = None
            for data_dic in self.config.tgt_speakers_wm_lst:
                if data_dic["speaker"] == speaker_name:
                    tgt_wm = data_dic["encoded_wm"]
                    break
            assert tgt_wm is not None
            speaker_audio_wm_dic[speaker_name].append({
                "org_audio": our_wm_audio,      # FPR has already been calculated in previous experiments.
                "our_wm_audio": our_wm_audio, "our_wm": tgt_wm,
                "purified_audio": our_wm_audio,
            })

        return speaker_audio_wm_dic

    def run(self):
        """
        This adaptive attack exploits a paired dataset following the exact same wm scheme,
        but using a different set of speech and watermark values
        It then trains an autoencoder to remove our watermarks.
        """

        # first prepare all data
        train_speaker_audio_wm_dic, tgt_speaker_audio_wm_dic = self.prepare_data()

        # train a de-noise autoencoder
        ae_ckpt_dir = self.out_dir.joinpath("ckpt")
        self.train_ae(train_speaker_audio_wm_dic, ckpt_dir=ae_ckpt_dir)

        # remove watermarks
        self.remove_watermark(self.out_dir.joinpath("train_purified"), train_speaker_audio_wm_dic, ae_ckpt_dir)
        self.remove_watermark(self.out_dir.joinpath("tgt_purified"), tgt_speaker_audio_wm_dic, ae_ckpt_dir)

        # check detection performance on purified waveform
        self.eval_wm(
            eval_dir=self.out_dir.joinpath("eval_train_data"),
            wav_net_ckpt=self.config.train_wm_exp_dir.joinpath("ckpt"),
            benign_encoded_wm=self.config.train_benign_encoded_wm,
            speaker_audio_wm_dic=train_speaker_audio_wm_dic,
            speakers_wm_lst=self.config.train_speakers_wm_lst,
        )

        self.eval_wm(
            eval_dir=self.out_dir.joinpath("eval_tgt_data"),
            wav_net_ckpt=self.config.tgt_wm_exp_dir.joinpath("ckpt"),
            benign_encoded_wm=self.config.tgt_benign_encoded_wm,
            speaker_audio_wm_dic=tgt_speaker_audio_wm_dic,
            speakers_wm_lst=self.config.tgt_speakers_wm_lst,
        )

        # # want to test whether purified speech can still transfer to commercial PlayHT?
        # combine purified audio into a long audio
        self.combine_purified_tgt(
            exp_dir=self.out_dir.joinpath("combine_purified_tgt"),
            tgt_purified_dir=self.out_dir.joinpath("tgt_purified"),
        )

        # evaluate fake speech generated by purified speech
        playht_speaker_audio_wm_dic = self.load_fake_speech_for_eval()
        if playht_speaker_audio_wm_dic is not None:
            self.eval_wm(
                eval_dir=self.out_dir.joinpath("eval_playht_purified_tgt"),
                wav_net_ckpt=self.config.tgt_wm_exp_dir.joinpath("ckpt"),
                benign_encoded_wm=self.config.tgt_benign_encoded_wm,
                speaker_audio_wm_dic=playht_speaker_audio_wm_dic,
                speakers_wm_lst=self.config.tgt_speakers_wm_lst,
            )
        else:
            print(f"Did not find playht output. Skip evaluation on fake speech.")

        self.save_status()









