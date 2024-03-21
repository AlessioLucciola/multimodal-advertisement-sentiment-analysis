import librosa
import numpy as np

def extract_waveform_from_audio_file(file, desired_length_seconds, offset, desired_sample_rate):
    #waveform, _ = librosa.load(path=file, duration=desired_length_seconds, offset=trim_seconds, sr=desired_sample_rate)
    waveform, _ = librosa.load(path=file, duration=desired_length_seconds, offset=offset, sr=desired_sample_rate)

    # Pad the waveform if it is shorter than the desired length
    desired_length_samples = int(desired_length_seconds * desired_sample_rate)
    if len(waveform) < desired_length_samples:
        # If the waveform is shorter than desired, pad it with zeros at the beginning
        waveform_padded = np.pad(waveform, (desired_length_samples - len(waveform), 0), mode='constant')
    else:
        # If the waveform is longer than desired, truncate it from the beginning
        waveform_padded = waveform[-desired_length_samples:]
    return waveform_padded

def extract_mfcc_features(waveform, sample_rate, n_mfcc, n_fft, win_length, n_mels, window):
    mfcc = librosa.feature.mfcc(
        y=waveform, 
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        win_length=win_length,
        n_mels=n_mels,
        window=window,
        fmax=sample_rate/2
    )
    return mfcc

def extract_rmse_features(waveform, frame_length, hop_length):
    rmse = np.squeeze(librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length))
    return rmse

def extract_zcr_features(waveform, frame_length, hop_length):
    zcr = np.squeeze(librosa.feature.zero_crossing_rate(y=waveform, frame_length=frame_length, hop_length=hop_length))
    return zcr