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
    waveform, sr = librosa.load(file, sr=desired_sample_rate)

    # Calculate the number of samples corresponding to the desired length and overlap
    desired_length_samples = int(desired_length_seconds * desired_sample_rate)
    overlap_samples = int(overlap_seconds * desired_sample_rate)

    # Determine the step size for sliding the window
    step_size = desired_length_samples - overlap_samples

    # Determine the number of segments needed
    num_segments = (len(waveform) - desired_length_samples) // step_size + 1

    segments = []

    # Extract overlapping segments
    for i in range(num_segments):
        start_index = i * step_size
        end_index = start_index + desired_length_samples
        segment_waveform = waveform[start_index:end_index]
        start_time = start_index / sr
        end_time = end_index / sr
        segments.append({
            "waveform": segment_waveform,
            "start_time": start_time,
            "end_time": end_time
        })

    return segments

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

def detect_speech(waveform, start_time, end_time, sr, frame_length=2048, hop_length=512, threshold_energy=0.05):
    # Calculate energy for each frame
    energy = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]

    # Compute the energy threshold based on the maximum energy
    threshold = threshold_energy * np.max(energy)

    # Determine speech segments based on energy thresholding
    speech_segments = []
    speech_start = None
    for i in range(len(energy)):
        if energy[i] > threshold:
            if speech_start is None:
                speech_start = start_time + i * hop_length / sr
        else:
            if speech_start is not None:
                speech_end = start_time + i * hop_length / sr
                # Only add the speech segment if it lies within the specified start and end times
                if speech_end <= end_time:
                    speech_segments.append((speech_start, speech_end))
                speech_start = None

    # If speech extends to the end of the waveform, add the last segment
    if speech_start is not None:
        speech_end = end_time
        speech_segments.append((speech_start, speech_end))
    speech_segments = unify_segments(speech_segments)
    speech_segments = discard_short_segments(speech_segments)

    return speech_segments

def unify_segments(speech_segments):
    unified_segments = []
    if not speech_segments:
        return unified_segments
    current_segment_start, current_segment_end = speech_segments[0]
    for next_segment_start, next_segment_end in speech_segments[1:]:
        # If the time gap between current and next segment is less than 0.3 seconds
        if next_segment_start - current_segment_end < 0.3:
            current_segment_end = next_segment_end # Merge the segments
        else:
            unified_segments.append((current_segment_start, current_segment_end)) # Add the current unified segment to the list
            current_segment_start, current_segment_end = next_segment_start, next_segment_end # Update current segment to the next segment
    unified_segments.append((current_segment_start, current_segment_end)) # Add the last unified segment
    return unified_segments

def discard_short_segments(speech_segments, min_duration=1):
    filtered_segments = []
    for start, end in speech_segments:
        segment_duration = end - start
        if segment_duration >= min_duration:
            filtered_segments.append((start, end))
    return filtered_segments

def extract_speech_segment_from_waveform(waveform, speech_segments, start_time, end_time, sr):
    # Convert start and end time to sample indices
    start_index = int(start_time * sr)
    end_index = int(end_time * sr)

    # Find the longest speech segment
    longest_segment_length = 0
    longest_segment_start = None
    longest_segment_end = None
    for segment_start, segment_end in speech_segments:
        segment_length = segment_end - segment_start
        if segment_length > longest_segment_length:
            longest_segment_length = segment_length
            longest_segment_start = segment_start-start_time
            longest_segment_end = segment_end-start_time
        #print(segment_length, longest_segment_length, start_time, end_time, segment_start, segment_end)

    # Repeat the longest speech segment from start to end
    longest_segment_start_index = int(longest_segment_start * sr)
    longest_segment_end_index = int(longest_segment_end * sr)
    repeated_segment = np.tile(waveform[longest_segment_start_index:longest_segment_end_index], (end_index - start_index) // (longest_segment_end_index - longest_segment_start_index) + 1)
    #print(repeated_segment)
    # Clip the repeated segment to match the length of the segment we are replacing
    clipped_repeated_segment = repeated_segment[:end_index - start_index]
    return clipped_repeated_segment