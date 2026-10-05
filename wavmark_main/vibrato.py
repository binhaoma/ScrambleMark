import os
import numpy as np
import soundfile as sf


def apply_vibrato_to_file(input_file, output_file, vibrato_rate=5, vibrato_depth=0.001):
    """
    Apply a vibrato effect to an audio file and save the output with high quality.
    """
    # Load the audio file with soundfile to avoid changing the sample rate
    y, sr = sf.read(input_file)

    # Ensure audio is mono
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    t = np.arange(len(y)) / sr
    vibrato = vibrato_depth * sr * np.sin(2 * np.pi * vibrato_rate * t)
    y_vibrato = np.zeros_like(y)

    for i in range(len(y)):
        vibrato_index = int(i + vibrato[i])
        if vibrato_index < 0 or vibrato_index >= len(y):
            continue
        y_vibrato[i] = y[vibrato_index]

    # Save the audio with high quality
    sf.write(output_file, y_vibrato, sr, format='WAV', subtype='PCM_16')


def process_folder(source_folder, target_folder, vibrato_rate=0.8, vibrato_depth=0.010):
    """
    Apply a vibrato effect to all compatible audio files in the source folder and save them to the target folder with high quality.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith(('.wav', '.flac', '.aiff')):
            input_file = os.path.join(source_folder, file_name)
            output_file = os.path.join(target_folder, 'vibrato_'+ os.path.splitext(file_name)[0]+'.wav')
            print(f"Processing {input_file}...")
            apply_vibrato_to_file(input_file, output_file, vibrato_rate, vibrato_depth)
            print(f"Saved to {output_file}")



def remove_vibrato_from_file(input_file, output_file, vibrato_rate=5, vibrato_depth=0.001):
    """
    Attempt to remove a vibrato effect from an audio file and restore the original sound.
    """
    # Load the audio file
    y_vibrato, sr = sf.read(input_file)

    # Ensure audio is mono
    if len(y_vibrato.shape) > 1:
        y_vibrato = y_vibrato.mean(axis=1)

    t = np.arange(len(y_vibrato)) / sr
    vibrato = vibrato_depth * sr * np.sin(2 * np.pi * vibrato_rate * t)
    
    y_restored = np.zeros_like(y_vibrato)

    for i in range(len(y_vibrato)):
        original_index = int(i - vibrato[i])  # Inverting the vibrato shift
        if 0 <= original_index < len(y_vibrato):
            y_restored[i] = y_vibrato[original_index]

    # Save the restored audio file
    sf.write(output_file, y_restored, sr, format='WAV', subtype='PCM_16')

def remove_process_folder(source_folder, target_folder, vibrato_rate=0.8, vibrato_depth=0.010):
    """
    Apply a vibrato effect to all compatible audio files in the source folder and save them to the target folder with high quality.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith(('.wav', '.flac', '.aiff')):
            input_file = os.path.join(source_folder, file_name)
            output_file = os.path.join(target_folder, os.path.splitext(file_name)[0] + '_remove.wav')
            print(f"Processing {input_file}...")
            remove_vibrato_from_file(input_file, output_file, vibrato_rate, vibrato_depth)
            print(f"Saved to {output_file}")



