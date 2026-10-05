from audioseal_main.src.audioseal import AudioSeal
import numpy as np
import torch
import torchaudio
import librosa
from wavmark_main.src import wavmark
from tqdm import tqdm
from audiomark.amn_code.models.WatermarkNet import WatermarkNet
from audiomark.amn_code.my_utils import utils as audiomark_utils
from pathlib import Path
import audiomark.amn_code.exp_setup as exp_setup
from audiomark.amn_code.exp.ExpEmbedWatermark import ExpEmbedWatermark, ExpConfig as ExpEmbedWatermarkCfg
import os
import gc


def get_wm_pool(speaker, speakers_wm_lst):
    """Return the AudioMarkNet watermark pool and the requested speaker index."""
    wm_pool = []
    speaker_idx = None
    for idx, data_dic in enumerate(speakers_wm_lst):
        wm_pool.append(data_dic["org_wm"])
        if data_dic["speaker"] == speaker:
            if speaker_idx is not None:
                raise ValueError(f"Duplicate AudioMarkNet speaker: {speaker}")
            speaker_idx = idx

    if speaker_idx is None:
        raise ValueError(f"Unknown AudioMarkNet speaker: {speaker}")

    wm_pool = torch.from_numpy(np.array(wm_pool)).to(audiomark_utils.device)
    return wm_pool, speaker_idx


def load_model(model_name, device):
    device = torch.device(device)

    if model_name == 'audioseal':
        model = AudioSeal.load_generator(
            "audioseal_wm_16bits"
        )

        model = model.to(device)

        model.eval()


        return model

    elif model_name == 'wavmark':
        model = wavmark.load_model()
        model = model.to(device)
        model.eval()

        return model


    else:
        raise ValueError(
            f"Unsupported watermark model: {model_name}"
        )

def load_detector_model(model_name, device):
    device = torch.device(device)

    if model_name == 'audioseal':
        detector = AudioSeal.load_detector(
            "audioseal_detector_16bits"
        )
        detector = detector.to(device)
        detector.eval()

        return detector

    elif model_name == 'wavmark':
        detector = wavmark.load_model()
        detector = detector.to(device)
        detector.eval()

        return detector
    
    elif model_name == 'audiomarknet':
        from audiomark.amn_code.models.ModelTrainer import ModelTrainer

        exp_cfg = exp_setup.init_config()

        speakers_wm_lst, benign_org_wm, benign_encoded_wm = \
            exp_setup.get_speakers_and_wm(
                exp_cfg,
                exp_cfg.wm_length
            )

        cfg = ExpEmbedWatermarkCfg(
            # Detector construction only needs the watermark metadata. Loading
            # and splitting the Obama training audio here made validation depend
            # on an unrelated dataset.
            extra_audio_lst_dic=None,
            speakers_wm_lst=speakers_wm_lst,
            vctk_dir=exp_setup.get_vctk_dir(exp_cfg),
            benign_org_wm=benign_org_wm,
            benign_encoded_wm=benign_encoded_wm,
            wav2vec2_dir=exp_cfg.wav2vec2_pretrained_dir,
            aug_normal_prob=exp_cfg.wmnet_aug_normal_prob,
            aug_normal_scale=exp_cfg.wmnet_aug_normal_scale,
        )

        model_dir = Path(exp_cfg.out_dir).joinpath(
            "ExpEmbedWatermark"
        )

        checkpoint = ModelTrainer.load_latest_ckpt(
            model_dir.joinpath("ckpt"),
            return_path=True
        )
        if checkpoint is None:
            raise FileNotFoundError(
                "No AudioMarkNet checkpoint (*.tar) found in "
                f"{model_dir.joinpath('ckpt')}. Place the trained checkpoint "
                "there; this path is relative to the repository's audiomark directory."
            )
        dic_saved, _ = checkpoint

        wm_net = WatermarkNet(
            cfg.benign_encoded_wm,
            cfg.expected_sr,
            cfg.audio_sec_len,
            wav2vec2_dir=cfg.wav2vec2_dir,
            aug_normal_prob=cfg.aug_normal_prob,
            aug_normal_scale=cfg.aug_normal_scale
        )

        wm_net.load_state_dict(dic_saved["model_state"])
        wm_net = wm_net.to(device)
        wm_net.eval()

        return wm_net

    else:
        raise ValueError(
            f"Unsupported detector model: {model_name}"
        )


def watermark_audio_generation(model_name, input_dir, output_dir=None, device='cuda'):
    model = load_model(model_name=model_name, device=device)
    try:
        if model_name != 'audiomarknet':
            if output_dir is None:
                output_dir = f'{model_name}_watermark_audio'

            output_dir_audio = os.path.join(output_dir, 'data')
            output_dir_msg = os.path.join(output_dir, 'msg')

            os.makedirs(output_dir_audio, exist_ok=True)
            os.makedirs(output_dir_msg, exist_ok=True)

        audios = [
            audio_name
            for audio_name in os.listdir(input_dir)
            if audio_name.lower().endswith(('.wav', '.flac', '.mp3', '.ogg'))
        ]
        if model_name == 'audioseal':
            sample_rate = 16000
            with torch.no_grad():
                for audio_name in tqdm(
                        audios,
                        desc="AudioSeal watermarking",
                        unit="audio"
                    ):
                    audio_path = os.path.join(input_dir, audio_name)

                    audio, sr = librosa.load(audio_path, sr=sample_rate)
                    audio = torch.tensor(audio, dtype=torch.float32).to(device)

                    secret_message = torch.randint(
                        0, 2, (1, 16),
                        dtype=torch.int32,
                        device=device
                    )

                    watermarked_audio = model(
                        audio.unsqueeze(0).unsqueeze(0),
                        sample_rate=sample_rate,
                        message=secret_message,
                        alpha=1
                    )

                    torchaudio.save(
                        os.path.join(output_dir_audio, audio_name),
                        watermarked_audio.squeeze(0).cpu(),
                        sample_rate
                    )

                    audio_base_name = os.path.splitext(audio_name)[0]

                    torch.save(
                        secret_message.cpu(),
                        os.path.join(
                            output_dir_msg,
                            f'{audio_base_name}_msg_tensor.pt'
                        )
                    )

        elif model_name == 'wavmark':
            with torch.no_grad():
                for audio_name in audios:
                    audio_path = os.path.join(input_dir, audio_name)

                    signal, sr = librosa.load(audio_path, sr=16000)

                    payload = np.random.choice([0, 1], size=16)

                    watermarked_signal, _ = wavmark.encode_watermark(
                        model,
                        signal,
                        payload,
                        show_progress=True
                    )

                    audio_base_name = os.path.splitext(audio_name)[0]

                    # 保存 message
                    torch.save(
                        torch.from_numpy(payload),
                        os.path.join(
                            output_dir_msg,
                            f'{audio_base_name}_msg_tensor.pt'
                        )
                    )

                    # 保存 watermarked audio
                    torchaudio.save(
                        os.path.join(
                            output_dir_audio,
                            audio_name
                        ),
                        torch.from_numpy(watermarked_signal)
                            .unsqueeze(0)
                            .float()
                            .cpu(),
                        16000
                    )
    finally:
        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()




def WM_validation(model_name, watermark_dir, audio_subdir='data', device='cuda'):
    if model_name == 'audiomarknet':
        audio_dir = watermark_dir
        msg_dir = None
    else:
        audio_dir = os.path.join(watermark_dir, audio_subdir)
        msg_dir = os.path.join(watermark_dir, 'msg')

    audios = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg'))]

    if len(audios) == 0:
        raise ValueError(f"No audio files found in: {audio_dir}")

    detector = load_detector_model(model_name, device)

    try:
        if model_name == 'audioseal':
            sample_rate = 16000
            detection_score_list = []
            attack_success_count = 0

            with torch.no_grad():
                for audio_name in tqdm(audios, desc="AudioSeal validation", unit="audio"):
                    audio_path = os.path.join(audio_dir, audio_name)
                    audio, _ = librosa.load(audio_path, sr=sample_rate)

                    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                    result, _ = detector.detect_watermark(
                        audio_tensor,
                        sample_rate=sample_rate,
                        message_threshold=0.5
                    )

                    detection_score = result.item() if torch.is_tensor(result) else float(result)
                    detection_score_list.append(detection_score)

                    if detection_score < 0.5:
                        attack_success_count += 1

            avg_detection_score = sum(detection_score_list) / len(detection_score_list)
            asr = attack_success_count / len(audios)

            print("\nModel: AudioSeal")
            print(f"Number of audio files: {len(audios)}")
            print(f"Average detection score: {avg_detection_score:.4f}")
            print(f"Attack success count: {attack_success_count}/{len(audios)}")
            print(f"ASR: {asr:.4f}")

        elif model_name == 'wavmark':
            bit_acc_list = []
            attack_success_count = 0
            decode_success_count = 0

            with torch.no_grad():
                for audio_name in tqdm(audios, desc="WavMark validation", unit="audio"):
                    audio_path = os.path.join(audio_dir, audio_name)
                    signal, _ = librosa.load(audio_path, sr=16000)

                    audio_base_name = os.path.splitext(audio_name)[0]
                    msg_path = os.path.join(msg_dir, f'{audio_base_name}_msg_tensor.pt')

                    secret_message = torch.load(msg_path, map_location='cpu')
                    secret_message = torch.as_tensor(secret_message)

                    payload_decoded, _ = wavmark.decode_watermark(
                        detector,
                        signal,
                        show_progress=False
                    )

                    if payload_decoded is None:
                        attack_success_count += 1
                        continue

                    decode_success_count += 1
                    payload_decoded = torch.as_tensor(payload_decoded)

                    bit_acc = ((payload_decoded > 0) == (secret_message > 0)).float().mean().item()
                    bit_acc_list.append(bit_acc)

            avg_bit_acc = sum(bit_acc_list) / len(bit_acc_list) if bit_acc_list else 0.0
            asr = attack_success_count / len(audios)

            print(f"\nModel: WavMark")
            print(f"Number of audio files: {len(audios)}")
            print(f"Decode success count: {decode_success_count}/{len(audios)}")
            print(f"Average bit accuracy: {avg_bit_acc:.4f}")
            print(f"Attack success count: {attack_success_count}/{len(audios)}")
            print(f"ASR: {asr:.4f}")
        
        elif model_name == 'audiomarknet':

            exp_cfg = exp_setup.init_config()
            speakers_wm_lst, benign_org_wm, benign_encoded_wm = exp_setup.get_speakers_and_wm(exp_cfg, exp_cfg.wm_length)

            audios = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg'))]

            attack_success_count = 0
            attack_fail_count = 0
            bit_acc_list = []

            with torch.no_grad():
                for audio_name in tqdm(audios, desc="AudioMarkNet validation", unit="audio"):
                    audio_path = os.path.join(audio_dir, audio_name)
                    waveform, _ = librosa.load(audio_path, sr=16000)

                    need_detect = int(len(waveform) / 16000)

                    if need_detect == 0:
                        print(f"Skip {audio_name}: audio is shorter than 1 second")
                        continue

                    wav_secs = detector.split_waveform(waveform)
                    spec = detector.spec_for_classificiation(wav_secs)[..., detector.min_freq_idx:detector.max_freq_idx, :]
                    logits = detector.decoder(spec)

                    pred_wm = (logits > 0.0).long()
                    pred_wm = pred_wm[:need_detect, :]

                    speaker_name = f"VCTK_{audio_name.split('_')[0]}"
                    wm_pool, speaker_idx = get_wm_pool(speaker_name, speakers_wm_lst)
                    target_wm = wm_pool[speaker_idx]

                    watermark_recovered = False
                    sample_acc_list = []

                    for one_wm in pred_wm:
                        acc_wm = ((one_wm == target_wm).sum() / one_wm.numel()).item()
                        sample_acc_list.append(acc_wm)

                        if acc_wm == 1.0:
                            watermark_recovered = True
                            break

                    if sample_acc_list:
                        bit_acc_list.append(max(sample_acc_list))

                    if watermark_recovered:
                        attack_fail_count += 1
                    else:
                        attack_success_count += 1

            valid_count = attack_success_count + attack_fail_count
            avg_bit_acc = sum(bit_acc_list) / len(bit_acc_list) if bit_acc_list else 0.0
            asr = attack_success_count / valid_count if valid_count > 0 else 0.0

            print("\nModel: AudioMarkNet")
            print(f"Number of valid audio files: {valid_count}")
            print(f"Average max bit accuracy: {avg_bit_acc:.4f}")
            print(f"Attack success count: {attack_success_count}/{valid_count}")
            print(f"Attack fail count: {attack_fail_count}/{valid_count}")
            print(f"ASR: {asr:.4f}")


        else:
            raise ValueError(f"Unsupported validation model: {model_name}")

    finally:
        del detector
        gc.collect()

        if torch.cuda.is_available() and str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    



    
