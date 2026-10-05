import logging
import pickle

import numpy as np

from exp.ExpBase import ExpBase
from my_utils import utils
from pathlib import Path
import scipy.io.wavfile as wav

import shutil

from my_sv2tts import synthesizer_preprocess_audio, synthesizer_preprocess_embeds, synthesizer_train
from my_sv2tts import vocoder_preprocess, vocoder_train

from encoder import inference as encoder
from synthesizer.inference import Synthesizer as SynWrapper
from vocoder import inference as VocWrapper


class ExpConfig(utils.ConfigBase):

    tgt_name = None
    wm_exp_dir = None

    restore_encoder_path = None
    restore_tacotron_path = None
    restore_vocoder_path = None

    dataset_name = "wm_audios"
    sub_folder = "our_wm_audios"
    chapter_dir_name = "0001"

    iter_lsts = None

    del_data_for_vocoder = True             # whether delete data for training vocoder to save space


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.tgt_name is not None, "tgt_name must be set"
        assert self.wm_exp_dir is not None, "wm_exp_dir must be set"

        assert (self.restore_encoder_path is not None
                and self.restore_tacotron_path is not None
                and self.restore_vocoder_path is not None), "Where are the pretrained weights?"

        assert self.iter_lsts is not None, "iter_lsts must be set"


class ExpSpeakerAdaptSV2TTS(ExpBase):

    """
        Speaker adaptation using Bark.
    """

    def __init__(self, out_dir, config,
                 exp_status_fname="exp.status"):
        """
        restore_path:
        exp_data_stat_fname: file name to save experiment data statistics
        """
        super().__init__(out_dir, config, exp_status_fname)

        self.syn_steps_run = None
        self.syn_model = None

    def prepare_data_structure(self):
        """
        Create a folder structure in a similar way as libriTTS dataset
        dataset/subfolder/Speakerid/chapter_id/wavfile
        dataset/subfolder/Speakerid/chapter_id/transfile
        """

        flag_path = self.out_dir.joinpath("prepare_data_structure_done.flag")
        if utils.get_flag(flag_path) is not None:
            return

        our_saved_path = self.config.wm_exp_dir.joinpath(f"tgt_samples.bin")
        assert our_saved_path.exists(), "Must run the watermarking experiments first!"

        with open(our_saved_path, 'rb') as handle:
            our_tgt_samples = pickle.load(handle)

        for sample, our_wm in our_tgt_samples:
            speaker_name = sample["speaker_name"]
            if speaker_name != self.config.tgt_name:
                continue

            wm_path = sample["audio_file_wm"]

            save_dir = self.out_dir.joinpath(f"{self.config.dataset_name}/{self.config.sub_folder}/"
                                             f"{speaker_name}/{self.config.chapter_dir_name}")
            save_dir.mkdir(exist_ok=True, parents=True)

            new_path = save_dir.joinpath(f"{wm_path.name}")

            shutil.copyfile(src=wm_path, dst=new_path)

            # also create the transcript file
            trans_path = new_path.with_suffix(".txt")
            with open(trans_path, "w") as f:
                f.write(sample["text"])

        utils.set_flag(flag_path)

    def prepare_syn_audio(self):
        """
        transform audio to MFCCs for training the synthesizer
        """

        flag_path = self.out_dir.joinpath("prepare_syn_audio_done.flag")
        if utils.get_flag(flag_path) is not None:
            return

        synthesizer_preprocess_audio.run(
            datasets_root=self.out_dir,
            datasets_name=self.config.dataset_name,
            subfolders=self.config.sub_folder,
        )

        utils.set_flag(flag_path)

    def prepare_syn_embeds(self):
        """
        calculate embeddings for each audio
        """

        flag_path = self.out_dir.joinpath("prepare_syn_embeds_done.flag")
        if utils.get_flag(flag_path) is not None:
            return

        synthesizer_preprocess_embeds.run(
            synthesizer_root=self.out_dir.joinpath("SV2TTS/synthesizer"),       # default one
            encoder_model_fpath=self.config.restore_encoder_path,
        )

        utils.set_flag(flag_path)

    def prepare_data(self):
        self.prepare_data_structure()
        self.prepare_syn_audio()
        self.prepare_syn_embeds()

    def on_train_step_start_vocoder(self, cur_iter, model):
        # cur_iter - 1 equals how many steps have run already
        steps_run = cur_iter - 1
        if steps_run == self.syn_steps_run:
            # generate fake speech and stop training vocoder
            self.generate_fake_speech(
                synthesizer=self.syn_model, vocoder=model,
                embed_dir=self.out_dir.joinpath("SV2TTS/synthesizer/embeds"),
                fake_speech_dir=self.out_dir.joinpath(f"iter_{cur_iter:04d}"),
                speaker_name=self.config.tgt_name, gen_sentences_lst=self.config.gen_sentences_lst,
            )

            return False

        # continue training since the number of iters has not reached yet
        return True

    def prepare_voc_data(self, model):

        training = model.training

        out_dir = vocoder_preprocess.run(
            datasets_root=self.out_dir,
            existing_model=model,
        )

        # reset the state of the model
        if training:
            model.train()
        else:
            model.eval()

        return out_dir

    def on_train_step_start_synthesizer(self, cur_iter, model):
        # may generate fake speech
        if cur_iter not in self.config.iter_lsts:
            return True

        # prepare the data for training vocoder
        voc_data_dir = self.prepare_voc_data(model)

        ### train the vocoder for the same number of iterations

        # vocoder callback will use self.syn_iter to determine how many iters to run
        # minus 1 because this is the on start callback and the training step has not run yet.
        self.syn_steps_run = cur_iter - 1
        self.syn_model = model

        vocoder_train.run(
            datasets_root=self.out_dir,
            models_dir=self.out_dir.joinpath("tmp"),
            model_ckpt_path=self.config.restore_vocoder_path,
            on_train_step_start=self.on_train_step_start_vocoder
        )

        if self.config.del_data_for_vocoder is True:
            # delete the data for vocoder
            shutil.rmtree(voc_data_dir)

        if cur_iter == self.config.iter_lsts[-1]:
            return False        # stop further training

        return True

    def speaker_adapt(self):
        """
        train the synthesizer for iters and then train vocoder for the same number of iters
        """
        flag_path = self.out_dir.joinpath("speaker_adapt_done.flag")
        if utils.get_flag(flag_path) is not None:
            return

        tmp_out_dir = self.out_dir.joinpath("tmp")
        tmp_out_dir.mkdir(exist_ok=True)

        synthesizer_train.run(
            syn_dir=self.out_dir.joinpath("SV2TTS/synthesizer"),
            models_dir=tmp_out_dir,        # it will find synthesizer.pt
            model_ckpt_path=self.config.restore_tacotron_path,
            on_train_step_start=self.on_train_step_start_synthesizer,
        )

        utils.set_flag(flag_path)

    def run(self):

        # first prepare data for fine-tuning
        self.prepare_data()

        # let's do it!
        self.speaker_adapt()

    @staticmethod
    def generate_fake_speech(synthesizer, vocoder, embed_dir, fake_speech_dir, speaker_name, gen_sentences_lst):

        # calculate the mean speaker embedding for the target speaker
        all_embed_path = list(embed_dir.glob(f"embed-*.npy"))
        assert len(all_embed_path) > 20, "Fail to find all embeds?"

        embed_arr = []
        for embed_path in all_embed_path:
            embed_load = np.load(embed_path)
            embed_arr.append(embed_load)
        embed_arr = np.stack(embed_arr)

        tgt_embed = embed_arr.mean(0)
        tgt_embed = tgt_embed / np.linalg.norm(tgt_embed, 2)

        syn_training = synthesizer.training
        synthesizer.eval()
        voc_training = vocoder.training
        vocoder.eval()

        syn_wrapper = SynWrapper(model_fpath=None)     # we will give it our own initialized model
        syn_wrapper._model = synthesizer

        assert syn_wrapper.sample_rate == 16000, "we are using 16000 sampling rate for open-source experiments"

        VocWrapper._model = vocoder     # directly use our vocoder

        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        for idx, sentence in enumerate(gen_sentences_lst):
            texts = [sentence]
            embeds = [tgt_embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = syn_wrapper.synthesize_spectrograms(texts, embeds)
            spec = specs[0]

            ## Generating the waveform

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = VocWrapper.infer_waveform(spec).astype(np.float32)

            # clip values beyond [-1, 1]
            generated_wav = np.clip(generated_wav, a_min=-1, a_max=1)

            ## Post-generation
            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            generated_wav = encoder.preprocess_wav(generated_wav)

            # Save it on the disk
            fake_speech_dir.mkdir(exist_ok=True)
            fake_speech_path = fake_speech_dir.joinpath(f"{speaker_name}_fake_{idx + 1:03d}.wav")
            assert generated_wav.dtype == np.float32, "the generated waveform should be in float32 format"

            wav.write(fake_speech_path, syn_wrapper.sample_rate, generated_wav)

        if syn_training:
            synthesizer.train()

        if voc_training:
            vocoder.train()



