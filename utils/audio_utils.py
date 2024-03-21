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

def apply_AWGN(waveform, bits=16, snr_min=15, snr_max=30): 
    wave_len = len(waveform) # Get length of waveform
    noise = np.random.normal(size=wave_len) # Generate Gaussian noise
    norm_constant = 2.0 ** (bits - 1) # Normalize waveform and noise
    norm_wave = waveform / norm_constant
    norm_noise = noise / norm_constant
    signal_power = np.sum(norm_wave ** 2) / wave_len # Compute signal power
    noise_power = np.sum(norm_noise ** 2) / wave_len # Compute noise power
    snr = np.random.randint(snr_min, snr_max) # Compute SNR
    covariance = np.sqrt((signal_power / noise_power) * 10 ** (- snr / 10)) # Compute covariance for whitening transformation
    augmented_waveform = waveform + covariance * noise # Apply whitening transformation
    augmented_waveform = augmented_waveform.astype(np.float32)
    
    return augmented_waveform