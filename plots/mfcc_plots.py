from config import AUDIO_RAVDESS_FILES_DIR, AUDIO_SAMPLE_RATE, AUDIO_OFFSET, AUDIO_DURATION, AUDIO_FILES_DIR, NUM_MFCC, USE_RAVDESS_ONLY
from matplotlib import pyplot as plt
import librosa
import os

def create_plot_mfcc(audio_path, audio_duration, audio_offset, sample_rate, n_mfcc, n_fft, win_length, n_mels, window):
    y, sr = librosa.load(path=audio_path, sr=sample_rate, duration=audio_duration, offset=audio_offset)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, win_length=win_length, n_mels=n_mels, window=window)
    print(len(mfccs))
    print(mfccs.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', cmap='viridis', sr=sr, hop_length=512)
    plt.colorbar(format='%+2.0f dB')
    plt.yticks(range(0, n_mfcc, 10))
    plt.title('MFCC sample')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficient')
    plt.show()

if __name__ == '__main__':
    audio_file_name = "03-01-01-01-01-01-23.wav"
    audio_path = os.path.join(os.path.join(AUDIO_RAVDESS_FILES_DIR if USE_RAVDESS_ONLY else AUDIO_FILES_DIR, audio_file_name))
    create_plot_mfcc(audio_path=audio_path,
                     audio_duration=AUDIO_DURATION,
                     audio_offset=AUDIO_OFFSET,
                     sample_rate=AUDIO_SAMPLE_RATE,
                     n_mfcc=NUM_MFCC,
                     n_fft=1024,
                     win_length=512,
                     n_mels=128,
                     window='hamming')