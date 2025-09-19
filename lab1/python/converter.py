import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt


def wav_to_spectrogram(wav_path, img_path):
    try:
        y, sr = librosa.load(wav_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(4, 4))
        librosa.display.specshow(S_dB, sr=sr, cmap='magma')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
