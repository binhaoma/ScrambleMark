from __future__ import print_function

import logging
import pickle
import sys

import numpy as np

import exp_setup
from pathlib import Path
from exp.ExpAdaptiveIter import ExpAdaptiveIter, ExpConfig as ExpAdaptiveIterCfg
from exp.ExpSpeakerAdaptYourTTS import ExpSpeakerAdaptYourTTS, ExpConfig as ExpSpeakerAdaptYourTTSCfg
from my_utils import utils
import copy
from TTS.tts.datasets.dataset import TTSDataset
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


def main():
    """
    Clarify how many audio samples would be required to achieve effective fine-tuning.
    This experiment is based on the adaptive iteration attack.
    """

    (wm_exp_dir, speakers_wm_lst, benign_encoded_wm, tgt_name, enrol_speech,
     tgt_audio_dic_lst, speaker_wm_dic, gen_sentences_lst, run_times) = exp_setup.setup_adaptive_iter_exps(exp_cfg)

    out_dir = Path(exp_cfg.out_dir)
    exp_dir = out_dir.joinpath("ExpEffectiveFinetuneYourTTS")
    exp_dir.mkdir(exist_ok=True)

    # num_audios_lst = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    num_audios_lst = [1, 5, 9, 13, 17, 21]

    for num_audios in num_audios_lst:

        sub_exp_dir = exp_dir.joinpath(f"num_audios_{num_audios}")

        sub_audio_dic_lst = tgt_audio_dic_lst[: num_audios]

        # save the audio total seconds
        total_secs = 0
        for audio_dic in sub_audio_dic_lst:
            tmp_audio, sr = utils.read_audio(audio_dic["audio_file"], expected_sr=exp_cfg.sr)
            total_secs += len(tmp_audio) / sr

        print(f"total seconds for {num_audios} audios = {total_secs:.1f}.")

        # may need to replicate the list to reach at least 100 size.
        # otherwise the overall number may be less than the batch size
        tgt_lst_len = 100
        if len(sub_audio_dic_lst) < tgt_lst_len:
            replicate_times = int(np.ceil(tgt_lst_len / len(sub_audio_dic_lst)))
            sub_audio_dic_lst = sub_audio_dic_lst * replicate_times

            logging.info(f"sub_audio_dic_lst is replicated for {replicate_times} times and size = {len(sub_audio_dic_lst)}")

            # make each name unique by appending an increasing int to the end
            for idx, audio_dic in enumerate(sub_audio_dic_lst):
                sub_audio_dic_lst[idx] = copy.copy(audio_dic)
                audio_dic = sub_audio_dic_lst[idx]
                audio_dic["audio_unique_name"] = audio_dic["audio_unique_name"] + f"_{idx}"

        # check that no audio will be removed
        new_samples = TTSDataset._compute_lengths(sub_audio_dic_lst)
        text_lengths = np.array([i["text_length"] for i in new_samples])
        audio_lengths = np.array([i["audio_length"] for i in new_samples])
        assert text_lengths.min() > 1
        assert (audio_lengths.min() > 1) and (audio_lengths.max() < 160000)

        cfg = ExpAdaptiveIterCfg(
            wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir,
            wm_exp_dir=wm_exp_dir,

            speakers_wm_lst=speakers_wm_lst,
            benign_encoded_wm=benign_encoded_wm,

            sr=exp_cfg.sr,
            audio_sec_len=exp_cfg.sr,  # 1 second in the same way as embedding

            tgt_name=tgt_name,
            tgt_wm=speaker_wm_dic["encoded_wm"],
            enrol_speech=enrol_speech,  # enrolment speech for speaker verification

            exp_speaker_adapt_exp_cfg=ExpSpeakerAdaptYourTTSCfg(
                run_name=f"adapt_to_{tgt_name}",
                vctk_dir=Path(exp_cfg.data_dir).joinpath("vctk"),

                restore_path=Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(
                    "exp1_vctk/best_model_latest.pth.tar"),

                tgt_audio_dic_lst=sub_audio_dic_lst,
                tgt_speaker_name=tgt_name,

                gen_sentences_lst=gen_sentences_lst,

                iter_lsts=exp_cfg.adap_iter_attack_iters,

                speaker_encoder_checkpoint_path=str(Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(exp_cfg.speaker_encoder_checkpoint_relpath)),
                speaker_encoder_config_path=str(Path(exp_cfg.coqui_ai_pretrained_dir).joinpath(exp_cfg.speaker_encoder_config_relpath)),
            ),
            exp_speaker_adapt_exp_class=ExpSpeakerAdaptYourTTS,
        )

        exp = ExpAdaptiveIter(sub_exp_dir, cfg)
        exp.run()

        with open(sub_exp_dir.joinpath(f"exp_info.txt"), "w") as f:
            print(f"total seconds for {num_audios} audios = {total_secs:.1f}.", file=f)

    # draw plots here
    plot_effective_ftune_exps(exp_dir, num_audios_lst, "YourTTS")


def plot_effective_ftune_exps(exp_base_dir, num_audios_lst, tts_name):

    # read all the data
    all_scores_arr = []
    all_ufl_arr = []
    all_bin_ufl_arr = []

    all_acc_arr = []
    all_bin_acc_arr = []

    exp_iter_lsts = None

    for num_audios in num_audios_lst:

        with open(exp_base_dir.joinpath(f"num_audios_{num_audios}/exp.status"), 'rb') as handle:
            status_dic = pickle.load(handle)

            all_scores_arr.append(status_dic["scores_lst"])

            all_acc_arr.append(status_dic["acc_lst"])
            all_bin_acc_arr.append(status_dic["bin_acc_lst"])

            def get_mean(_arr):
                mean_arr = []
                for _v in _arr:
                    assert len(_v) == 1, "Should have only one speaker"
                    _v = _v[0]
                    if len(_v) == 0:
                        mean_arr.append(0)
                    else:
                        mean_arr.append(np.mean(_v))
                return mean_arr

            all_ufl_arr.append(get_mean(status_dic["ufl_lst"]))
            all_bin_ufl_arr.append(get_mean(status_dic["bin_ufl_lst"]))

            iter_lsts = status_dic["iter_lsts"]

            if exp_iter_lsts is None:
                exp_iter_lsts = iter_lsts
            else:
                # all experiment runs must have the same set of iterations
                for (a, b) in zip(exp_iter_lsts, iter_lsts):
                    assert a == b

    all_scores_arr = np.array(all_scores_arr)
    all_ufl_arr = np.array(all_ufl_arr)
    all_bin_ufl_arr = np.array(all_bin_ufl_arr)

    all_acc_arr = np.array(all_acc_arr).squeeze(-1) * 100.0
    all_bin_acc_arr = np.array(all_bin_acc_arr).squeeze(-1) * 100.0

    def do_plot(fig_path, title, data_arr, ylim, label, threshold=None):
        # plot overall figure
        sns.set_style("darkgrid")
        matplotlib.rc('xtick', labelsize=14)
        matplotlib.rc('ytick', labelsize=14)
        plt.figure(figsize=(6, 6))
        plt.title(f"{tts_name} {title}", fontsize=20)

        plot_iters = np.array(exp_iter_lsts)
        plot_iters = plot_iters - 1  # offset one as our callback is on start.

        for _num, x in zip(num_audios_lst, data_arr):
            label_str = f"{_num} audio"
            if _num > 1:
                label_str += "s"
            plt.plot(plot_iters, x, "-", linewidth=2, label=label_str, alpha=0.75)

        if threshold is not None:
            plt.plot(plot_iters, [threshold]*len(plot_iters), "--r", linewidth=2, label="threshold")

        plt.xlabel(r"Iterations", fontsize=20)
        plt.ylabel(label, fontsize=20)
        plt.ylim(ylim[0], ylim[1])
        plt.legend(fontsize=14)
        # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

    do_plot(exp_base_dir.joinpath(f"{exp_base_dir.name}_ufl.png"), title="UFL",
            data_arr=all_ufl_arr,
            ylim=[0, 60], label="UFL")

    do_plot(exp_base_dir.joinpath(f"{exp_base_dir.name}_bin_ufl.png"), title="BUFL",
            data_arr=all_bin_ufl_arr,
            ylim=[0, 60], label="BUFL")

    do_plot(exp_base_dir.joinpath(f"{exp_base_dir.name}_scores.png"), title="Speaker Verification",
            data_arr=all_scores_arr,
            ylim=[-1.4, -0.8], label="scores", threshold=-1.11)

    do_plot(exp_base_dir.joinpath(f"{exp_base_dir.name}_acc.png"), title="TPR",
            data_arr=all_acc_arr,
            ylim=[-1, 101], label="TPR (%)")

    do_plot(exp_base_dir.joinpath(f"{exp_base_dir.name}_bin_acc.png"), title="BTPR",
            data_arr=all_bin_acc_arr,
            ylim=[-1, 101], label="BTPR (%)")



if __name__ == '__main__':
    # Training settings
    exp_cfg = exp_setup.init_config()
    main()

    sys.exit()






