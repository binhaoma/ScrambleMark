import logging
import os
import shutil
import warnings

from audiomark.amn_code.exp.ExpBase import ExpBase
from audiomark.amn_code.my_utils import utils
from pathlib import Path
import scipy.io.wavfile as wav
import numpy as np

import torch
from trainer import Trainer, TrainerArgs

from TTS.utils.download import extract_archive
from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig


class ExpConfig(utils.ConfigBase):
    # Name of the run for the Trainer
    run_name = "YourTTS"

    iter_lsts = None

    lr_gen: float = 0.0001
    lr_disc: float = 0.0001

    # Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
    # model_out_dname = "model_out_dir"

    # This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
    skip_train_epoch = False

    # Set here the batch size to be used in training and evaluation
    batch_size = 32

    # If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
    # e.g, "tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth"
    restore_path = None

    # Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
    # Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
    sample_rate = 16000

    # Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
    max_audio_len_in_seconds = 10

    vctk_dir = None

    # adaptation to a new speaker
    tgt_audio_dic_lst = None
    tgt_audio_keep_existing = False
    tgt_speaker_name = None

    # single speaker model
    single_speaker_name = None

    # sentences to generate
    gen_sentences_lst = None

    # whether delete fine-tuned models to save space
    del_ftuned_models = True

    speaker_encoder_checkpoint_path = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
    speaker_encoder_config_path = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.iter_lsts is not None, "iter_lsts must be provided"

        assert self.tgt_speaker_name is None or self.single_speaker_name is None, "either tgt_speaker_name or single_speaker_name is set."

        # checks
        assert self.vctk_dir is not None, "vctk_dir must be set"

        if self.restore_path is None:
            logging.info("restore_path = None, train YourTTS from scratch?")
        else:
            self.restore_path = str(self.restore_path)
            logging.info(f"Will restore model from {self.restore_path}.")

        assert self.gen_sentences_lst is not None, "gen_sentences_lst must be set"


class ExpSpeakerAdaptYourTTS(ExpBase):

    """
        Speaker adaptation using YourTTS.
        The code is extracted from YourTTS opensource code.
        This recipe replicates the first experiment proposed in the YourTTS paper (https://arxiv.org/abs/2112.02418).
        YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
        In addition, YourTTS can be trained in multilingual data, however, this recipe replicates the single language training using the VCTK dataset.
        If you are interested in multilingual training, we have commented on parameters on the VitsArgs class instance that should be enabled for multilingual training.
        In addition, you will need to add the extra datasets following the VCTK as an example.
    """

    def __init__(self, out_dir, config,
                 exp_status_fname="exp.status"):
        """
        restore_path:
        exp_data_stat_fname: file name to save experiment data statistics
        """
        super().__init__(out_dir, config, exp_status_fname)

        self.cur_iter = 1       # used in the training step callback

    def prepare_vctk(self):
        vctk_dir = self.config.vctk_dir
        flag_path = vctk_dir.joinpath("extract_vctk_done.flag")
        if utils.get_flag(flag_path) is not None:
            return

        zip_path = vctk_dir.joinpath("VCTK-Corpus-0.92.zip")
        assert zip_path.exists(), "VCTK dataset must be downloaded beforehand."

        print(f" > Extracting archive file: {zip_path} ...")
        extract_archive(str(zip_path))

        print(f" > resampling datafiles ...")
        resample_files(vctk_dir, self.config.sample_rate, file_ext="flac")

        utils.set_flag(flag_path)

    def add_tgt(self, train_samples):
        assert self.config.tgt_audio_dic_lst is not None

        for idx, tgt_dic in enumerate(self.config.tgt_audio_dic_lst):

            # add the target into training data
            tgt_sample = {
                "text": tgt_dic["text"],
                "audio_file": tgt_dic["audio_file"],
                "speaker_name": self.config.tgt_speaker_name,
                "root_path": "/home/wzong/My Passport/projects/data/vctk",
                "language": "en",
                "audio_unique_name": f"vctk#wav48_silence_trimmed/p999/p999_{idx+1}_mic1",
            }
            train_samples.append(tgt_sample)

    def may_register_embed(self, speaker_manager):
        assert self.config.tgt_audio_dic_lst is not None

        # also calculate the d_vector and add it into speaker manager
        speaker_mapping = {}
        for fields in self.config.tgt_audio_dic_lst:
            class_name = fields["speaker_name"]
            audio_file = fields["audio_file"]
            embedding_key = fields["audio_unique_name"]

            assert embedding_key not in speaker_mapping, "Replicates cannot exist"

            if class_name in speaker_manager.name_to_id:
                print("do not register this embedding as it already exists")
                return

            # extract the embedding
            embedd = speaker_manager.compute_embedding_from_clip(audio_file)

            # create speaker_mapping if target dataset is defined
            speaker_mapping[embedding_key] = {}
            speaker_mapping[embedding_key]["name"] = class_name
            speaker_mapping[embedding_key]["embedding"] = embedd

        # store values
        speaker_manager.name_to_id[self.config.tgt_speaker_name] = len(speaker_manager.name_to_id)

        clip_ids = [x["audio_unique_name"] for x in self.config.tgt_audio_dic_lst]
        speaker_manager.clip_ids.extend(clip_ids)

        embeddings_by_names = {}
        for x in speaker_mapping.values():
            if x["name"] not in embeddings_by_names.keys():
                embeddings_by_names[x["name"]] = [x["embedding"]]
            else:
                embeddings_by_names[x["name"]].append(x["embedding"])
        assert len(embeddings_by_names) == 1, "only support one target now"
        speaker_manager.embeddings_by_names.update(embeddings_by_names)

        speaker_manager.embeddings.update(speaker_mapping)

        print(f"Successfully register {self.config.tgt_speaker_name} embeddings.")
        return

    def single_speaker(self, train_samples):
        # only keep the single speaker
        assert self.config.single_speaker_name is not None

        single_samples = []

        for sample in train_samples:
            if sample["speaker_name"] == self.config.single_speaker_name:
                single_samples.append(sample)

        return single_samples

    def run(self):

        previous_run_dir = self.out_dir.joinpath(self.config.run_name)
        flag_path = previous_run_dir.joinpath("run_done.flag")
        if utils.get_flag(flag_path):
            return

        self.prepare_vctk()

        # init configs
        vctk_config = BaseDatasetConfig(
            formatter="vctk",
            dataset_name="vctk",
            meta_file_train="",
            meta_file_val="",
            path=str(self.config.vctk_dir),
            language="en",
            ignored_speakers=[
                "p261",
                "p225",
                "p294",
                "p347",
                "p238",
                "p234",
                "p248",
                "p335",
                "p245",
                "p326",
                "p302",
            ],  # Ignore the test speakers to full replicate the paper experiment
        )

        # Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to add new datasets, just add them here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
        datasets_config_list = [vctk_config]

        ### Extract speaker embeddings
        speaker_encoder_checkpoint_path = self.config.speaker_encoder_checkpoint_path
        speaker_encoder_config_path = self.config.speaker_encoder_config_path

        d_vector_files = []  # List of speaker embeddings/d-vectors to be used during the training

        # Iterates all the dataset configs checking if the speakers embeddings are already computated, if not compute it
        for dataset_conf in datasets_config_list:
            # Check if the embeddings weren't already computed, if not compute it
            embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
            if not os.path.isfile(embeddings_file):
                print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
                compute_embeddings(
                    speaker_encoder_checkpoint_path,
                    speaker_encoder_config_path,
                    embeddings_file,
                    old_speakers_file=None,
                    config_dataset_path=None,
                    formatter_name=dataset_conf.formatter,
                    dataset_name=dataset_conf.dataset_name,
                    dataset_path=dataset_conf.path,
                    meta_file_train=dataset_conf.meta_file_train,
                    meta_file_val=dataset_conf.meta_file_val,
                    disable_cuda=False,
                    no_eval=False,
                )
            d_vector_files.append(embeddings_file)

        # Audio config used in training.
        audio_config = VitsAudioConfig(
            sample_rate=self.config.sample_rate,
            hop_length=256,
            win_length=1024,
            fft_size=1024,
            mel_fmin=0.0,
            mel_fmax=None,
            num_mels=80,
        )

        # Init VITSArgs setting the arguments that are needed for the YourTTS model
        model_args = VitsArgs(
            d_vector_file=d_vector_files,
            use_d_vector_file=True,
            d_vector_dim=512,
            num_layers_text_encoder=10,
            speaker_encoder_model_path=speaker_encoder_checkpoint_path,
            speaker_encoder_config_path=speaker_encoder_config_path,
            resblock_type_decoder="2",
            # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
            # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
            # use_speaker_encoder_as_loss=True,
            # Useful parameters to enable multilingual training
            # use_language_embedding=True,
            # embedded_language_dim=4,
        )

        # General training config, here you can change the batch size and others useful parameters
        # model_out_dir = str(self.out_dir.joinpath(self.config.model_out_dname))
        model_out_dir = str(self.out_dir)

        test_sentence_tgt = self.config.single_speaker_name
        if test_sentence_tgt is None:
            test_sentence_tgt = self.config.tgt_speaker_name

        config = VitsConfig(
            output_path=model_out_dir,
            epochs=99,  # will update epochs after we get the number of samples
            lr_gen=self.config.lr_gen,
            lr_disc=self.config.lr_disc,

            model_args=model_args,
            run_name=self.config.run_name,
            project_name="YourTTS",
            run_description="""
                            - finetune YourTTS using VCTK dataset and unseen data
                        """,
            dashboard_logger="tensorboard",
            logger_uri=None,
            audio=audio_config,
            batch_size=self.config.batch_size,
            batch_group_size=48,
            eval_batch_size=self.config.batch_size,
            num_loader_workers=8,
            eval_split_max_size=256,
            print_step=50,
            plot_step=100,
            log_model_step=1000,
            save_step=5000,
            save_n_checkpoints=2,
            save_checkpoints=True,
            target_loss="loss_1",
            print_eval=False,
            use_phonemes=False,
            phonemizer="espeak",
            phoneme_language="en",
            compute_input_seq_cache=True,
            add_blank=True,
            text_cleaner="multilingual_cleaners",
            characters=CharactersConfig(
                characters_class="TTS.tts.models.vits.VitsCharacters",
                pad="_",
                eos="&",
                bos="*",
                blank=None,
                characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00af\u00b7\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u00ff\u0101\u0105\u0107\u0113\u0119\u011b\u012b\u0131\u0142\u0144\u014d\u0151\u0153\u015b\u016b\u0171\u017a\u017c\u01ce\u01d0\u01d2\u01d4\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451\u0454\u0456\u0457\u0491\u2013!'(),-.:;? ",
                punctuations="!'(),-.:;? ",
                phonemes="",
                is_unique=True,
                is_sorted=True,
            ),
            phoneme_cache_path=None,
            precompute_num_workers=12,
            start_by_longest=True,
            datasets=datasets_config_list,
            cudnn_benchmark=False,
            max_audio_len=self.config.sample_rate * self.config.max_audio_len_in_seconds,
            mixed_precision=False,
            test_sentences=[
                [
                    self.config.gen_sentences_lst[0],
                    test_sentence_tgt,
                    None,
                    "en",
                ]   # only one sentence here. We will generate all sentences after adaptation
            ],

            # Enable the weighted sampler
            use_weighted_sampler=True,
            # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
            weighted_sampler_attrs={"speaker_name": 1.0},
            weighted_sampler_multipliers={},
            # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
            speaker_encoder_loss_alpha=9.0,
        )

        # Load all the datasets samples and split traning and evaluation sets
        train_samples, eval_samples = load_tts_samples(
            config.datasets,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
        )

        # Init the model
        model = Vits.init_from_config(config)

        # insert target voice for speaker adaptation 🤩
        if self.config.tgt_audio_dic_lst is not None:
            if self.config.tgt_audio_keep_existing:
                self.add_tgt(train_samples)
            else:
                train_samples = self.config.tgt_audio_dic_lst
                eval_samples = train_samples

            self.may_register_embed(model.speaker_manager)

        elif self.config.single_speaker_name is not None:
            train_samples = self.single_speaker(train_samples)
            eval_samples = train_samples

        # Init the trainer and 🚀
        trainer = Trainer(
            TrainerArgs(restore_path=self.config.restore_path, skip_train_epoch=self.config.skip_train_epoch),
            config,
            output_path=model_out_dir,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,

            callbacks={
                "on_train_step_start": lambda _x: self.on_train_step_start(_x, test_sentence_tgt=test_sentence_tgt)
            },
        )

        self.cur_iter = 1       # we track the number of iterations we have run. It starts from 1.

        # calculate the number of epochs that are enough for iterations
        max_iter = self.config.iter_lsts[-1]
        assert trainer.config.epochs == 99, "Not the value we set before???"
        iter_per_epch = len(train_samples) // trainer.config.batch_size
        trainer.config.epochs = int(np.ceil(max_iter / iter_per_epch * 1.25)) # * 1.25 in case some data are removed by VITSDataset

        trainer.fit()

        # find the directory starting with our run name and rename it to "only" run name
        if previous_run_dir.exists():
            logging.info(f"deleting previous run directory: {previous_run_dir}")
            shutil.rmtree(previous_run_dir)

        run_dir_lst = list(self.out_dir.glob(f"{self.config.run_name}*"))
        assert len(run_dir_lst) == 1, "there should be only one run directory here"
        cur_run_dir = run_dir_lst[0]
        cur_run_dir = cur_run_dir.rename(previous_run_dir)

        # do not generate fake speech here as we have done it in the callback
        # self.generate_fake_speech(model, fake_speech_dir=cur_run_dir.joinpath("fake_speech"),
        #                           test_sentence_tgt=test_sentence_tgt,
        #                           gen_sentences_lst=self.config.gen_sentences_lst,
        #                           sr=self.config.sample_rate)

        if self.config.del_ftuned_models:
            # delete fine-tuned models to save space
            for model_path in cur_run_dir.glob("*.pth"):
                logging.info(f"deleting {model_path}")
                model_path.unlink()

        utils.set_flag(flag_path)

    def on_train_step_start(self, trainer, test_sentence_tgt):
        # Even if we have reached the maximum number of iterations, we cannot stop training immediately.

        if self.cur_iter not in self.config.iter_lsts:
            self.cur_iter += 1
            return

        # generate fake speech
        training = trainer.model.training
        trainer.model.eval()
        self.generate_fake_speech(
            model=trainer.model,
            fake_speech_dir=self.out_dir.joinpath(f"iter_{self.cur_iter:04d}"),
            test_sentence_tgt=test_sentence_tgt,
            gen_sentences_lst=self.config.gen_sentences_lst,
            sr=self.config.sample_rate)

        if training:
            trainer.model.train()

        self.cur_iter += 1

        return

    @staticmethod
    def generate_fake_speech(model, fake_speech_dir, test_sentence_tgt, gen_sentences_lst, sr):
        assert model.training is False, "model must be in the eval mode"

        # generate all fake speech into files
        old_test_sentences = model.config.test_sentences
        model.config.test_sentences = [
            [
                x,
                test_sentence_tgt,
                None,
                "en",
            ] for x in gen_sentences_lst
        ]
        gen_dic = model.test_run(assets=None)
        gen_audios_dic = gen_dic["audios"]

        fake_speech_dir.mkdir(exist_ok=True)
        for idx, (audio_name, waveform) in enumerate(gen_audios_dic.items()):
            fake_speech_path = fake_speech_dir.joinpath(f"{test_sentence_tgt}_fake_{idx + 1:03d}.wav")
            wav.write(fake_speech_path, sr, waveform)

        # restore the previous setting
        model.config.test_sentences = old_test_sentences




