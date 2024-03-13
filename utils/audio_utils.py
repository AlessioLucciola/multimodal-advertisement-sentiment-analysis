import librosa
import numpy as np

def extract_waveform_from_audio_file(file, desired_length_seconds, trim_seconds, desired_sample_rate):
    waveform, _ = librosa.load(path=file, duration=desired_length_seconds, offset=trim_seconds, sr=desired_sample_rate)
    
    # Pad the waveform if it is shorter than the desired length
    desired_length_samples = int(desired_length_seconds * desired_sample_rate)
    if len(waveform) < desired_length_samples:
        # If the waveform is shorter than desired, pad it with zeros at the beginning
        waveform_padded = np.pad(waveform, (desired_length_samples - len(waveform), 0), mode='constant')
    else:
        # If the waveform is longer than desired, truncate it from the beginning
        waveform_padded = waveform[-desired_length_samples:]
    return waveform_padded