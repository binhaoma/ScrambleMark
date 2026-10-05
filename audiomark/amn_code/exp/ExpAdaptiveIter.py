import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wav
from my_utils import utils

from exp.ExpBase import ExpBase
from exp.ExpSpeakerAdaptYourTTS import ExpSpeakerAdaptYourTTS
from exp.ExpEmbedWatermark import load_existing_wm_exp_data, get_existing_wm_samples

from models import ModelTrainer as MyModelTrainer
from models.WatermarkNet import WatermarkNet
from run_evaluate import eval_metrics

from SpeakerNet import *
from run_trainSpeakerNet import WrappedModel, args
from DatasetLoader import loadWAV


args.gpu = 0


def slice_wav(audio):
    """
    Modified from DatasetLoader from voxceleb_trainer
    """
    max_frames = args.eval_frames
    num_eval = 10
    evalmode = True

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = numpy.stack(feats, axis=0).astype(float)

    feat = torch.FloatTensor(feat)
    return feat


class ExpConfig(utils.ConfigBase):

    wav2vec2_dir = None                 # used to initialize wav2vec2 model
    wm_exp_dir = None                   # dir for the watermarking experiments, used to find saved WatermarkNet

    speakers_wm_lst = None
    benign_encoded_wm = None

    sr = None
    audio_sec_len = None
    
    tgt_name = None
    tgt_wm = None
    enrol_speech = None                 # enrolment speech for speaker verification

    exp_speaker_adapt_exp_cfg = None
    exp_speaker_adapt_exp_class = None


class ExpAdaptiveIter(ExpBase):
    """
    Adaptive iteration attack
    """
    def __init__(self, out_dir, config,  exp_status_fname="exp.status"):
        """
        exp_data_stat_fname: file name to save experiment data statistics
        """
        super().__init__(out_dir, config, exp_status_fname)

        # load model for speaker verification
        self.speaker_net = SpeakerNet(**vars(args))
        self.speaker_net = WrappedModel(self.speaker_net).to(utils.device)

        # load model parameters
        speaker_trainer = ModelTrainer(self.speaker_net, **vars(args))
        speaker_trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))

        # set to eval
        self.speaker_net.eval()

        # calculate features of enrolment speech
        enrol_slice = slice_wav(self.config.enrol_speech).to(utils.device)
        self.enrol_feat = self.speaker_net(enrol_slice)

        self.exp_running = None

    def save_status(self):
        super().save_status()

    def gen_fake_speech(self):
        """
        check whether fake speech can fool the speaker recognition model while not having our watermarks
        """
        flag_path = self.out_dir.joinpath(f"gen_fake_speech_done_{self.config.tgt_name}.flag")
        if utils.get_flag(flag_path):
            return

        self.exp_running = self.config.exp_speaker_adapt_exp_class(self.out_dir, self.config.exp_speaker_adapt_exp_cfg)
        self.exp_running.run()

        utils.set_flag(flag_path)

    def eval_speaker_verification(self, fake_speech):
        ref_feat = self.enrol_feat
        num_eval = 10

        fake_feat = slice_wav(fake_speech).to(utils.device)
        fake_feat = self.speaker_net(fake_feat)

        if self.speaker_net.module.__L__.test_normalize:
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            fake_feat = F.normalize(fake_feat, p=2, dim=1)

        dist = torch.cdist(ref_feat.reshape(num_eval, -1), fake_feat.reshape(num_eval, -1)).detach().cpu().numpy()

        score = -1 * numpy.mean(dist)

        return score

    def eval_fake_speech(self):
        """
        check whether the attack can defeat our watermark
        """
        flag_path = self.out_dir.joinpath(f"eval_fake_speech_done_{self.config.tgt_name}.flag")
        if utils.get_flag(flag_path):
            return

        fake_dir_lst = list(self.out_dir.glob("iter_*"))
        fake_dir_lst = sorted(fake_dir_lst)

        eval_wm_dir_base = self.out_dir.joinpath(f"eval_wm")
        eval_wm_dir_base.mkdir(exist_ok=True)

        # load our watermark net
        wm_net = WatermarkNet(self.config.benign_encoded_wm, self.config.sr, self.config.audio_sec_len,
                              wav2vec2_dir=self.config.wav2vec2_dir)
        dic_saved = MyModelTrainer.ModelTrainer.load_latest_ckpt(self.config.wm_exp_dir.joinpath("ckpt"))
        wm_net.load_state_dict(dic_saved["model_state"])
        wm_net = wm_net.to(utils.device)
        wm_net.eval()

        scores_lst = []
        ufl_lst = []
        bin_ufl_lst = []
        acc_lst = []
        bin_acc_lst = []
        fpr_lst = []

        for fake_dir in fake_dir_lst:
            iter_num = int(fake_dir.name[-4:])

            # find all fake speech in the folder
            fake_speech_path = fake_dir.joinpath(f"{self.config.tgt_name}_fake_001.wav")
            fake_speech, _ = utils.read_audio(fake_speech_path, expected_sr=self.config.sr)

            # speaker verification
            score = self.eval_speaker_verification(fake_speech)
            scores_lst.append(score)

            # watermark detection
            speaker_audio_wm_dic = {
                self.config.tgt_name: [
                    {
                        "org_audio": self.config.enrol_speech,
                        "our_wm_audio": fake_speech,
                        "our_wm": self.config.tgt_wm,
                    }
                ]
            }

            # directly use the evaluation code
            our_ufl_arr, our_bin_ufl_arr, our_fpr_arr, our_acc_arr, our_bin_acc_arr, our_eer_arr, our_bin_eer_arr = eval_metrics(
                exp_dir=eval_wm_dir_base.joinpath(f"{self.config.tgt_name}_iter_{iter_num:04d}"),
                speaker_audio_wm_dic=speaker_audio_wm_dic,
                wm_net=wm_net, wavmark_net=None,
                speakers_wm_lst=self.config.speakers_wm_lst
            )

            ufl_lst.append(our_ufl_arr)
            bin_ufl_lst.append(our_bin_ufl_arr)

            fpr_lst.append(our_fpr_arr)

            acc_lst.append(our_acc_arr)
            bin_acc_lst.append(our_bin_acc_arr)

        # save the data
        self.status_dic["scores_lst"] = scores_lst
        self.status_dic["ufl_lst"] = ufl_lst
        self.status_dic["bin_ufl_lst"] = bin_ufl_lst

        self.status_dic["fpr_lst"] = fpr_lst

        self.status_dic["acc_lst"] = acc_lst
        self.status_dic["bin_acc_lst"] = bin_acc_lst

        self.status_dic["iter_lsts"] = self.config.exp_speaker_adapt_exp_cfg.iter_lsts

        print(self.status_dic)

        self.save_status()

        utils.set_flag(flag_path)

    def run(self):
        """
        This adaptive attack tries to limit the number of iterations.
        The goal is to get fake voice while not learning our watermarks.
        """
        self.gen_fake_speech()

        self.eval_fake_speech()

        self.save_status()


