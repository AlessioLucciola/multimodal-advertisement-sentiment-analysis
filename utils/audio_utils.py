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

def extract_multiple_waveforms_from_audio_file(file, desired_length_seconds, desired_sample_rate, overlap_seconds=2.5):
    # Load the entire audio file
    waveform, _ = librosa.load(file, sr=desired_sample_rate)

    # Calculate the number of samples corresponding to the desired length and overlap
    desired_length_samples = int(desired_length_seconds * desired_sample_rate)
    overlap_samples = int(overlap_seconds * desired_sample_rate)

    # Determine the step size for sliding the window
    step_size = desired_length_samples - overlap_samples

    # Determine the number of segments needed
    num_segments = (len(waveform) - desired_length_samples) // step_size + 1

    waveforms = []

    # Extract overlapping segments
    for i in range(num_segments):
        start_index = i * step_size
        end_index = start_index + desired_length_samples
        segment = waveform[start_index:end_index]
        waveforms.append(segment)

    return waveforms

def extract_mfcc_features(waveform, sample_rate, n_mfcc, n_fft, win_length, n_mels, window):
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, win_length=win_length, n_mels=n_mels, window=window, fmax=sample_rate/2)
    return mfcc

def extract_rms_features(waveform, frame_length, hop_length):
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)
    return rms

def extract_zcr_features(waveform, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(y=waveform, frame_length=frame_length, hop_length=hop_length)
    return zcr

def extract_chroma_features(waveform, sample_rate):
    stft = np.abs(librosa.stft(waveform))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    return chroma

def extract_mel_features(waveform, sample_rate):
    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    return mel

def extract_zcr_features(waveform, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(y=waveform, frame_length=frame_length, hop_length=hop_length)
    return zcr

def extract_features(waveform, sample_rate, n_mfcc, n_fft, win_length, n_mels, window, frame_length, hop_length):
    mfcc = extract_mfcc_features(waveform, sample_rate, n_mfcc, n_fft, win_length, n_mels, window)
    rms = extract_rms_features(waveform, frame_length, hop_length)
    zcr = extract_zcr_features(waveform, frame_length, hop_length)
    chroma = extract_chroma_features(waveform, sample_rate)
    mel = extract_mel_features(waveform, sample_rate)
    #features = np.hstack((mfcc.T, rms.T, zcr.T, chroma.T, mel.T))
    features = mfcc
    return features

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