import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt


def wav_to_spectrogram(wav_path, img_path, image_size=112):
    try:
        y, sr = librosa.load(wav_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=image_size)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(0.01*image_size, 0.01*image_size), dpi=100)
        librosa.display.specshow(S_dB, sr=sr, cmap='gray', ax=ax)
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
