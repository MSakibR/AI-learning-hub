import numpy as np
import librosa


def load_audio(file_path, sr=22050):
    """Load audio file"""
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def extract_mfcc(audio, sr=22050, n_mfcc=40, max_len=100):
    """
    Extract MFCC features and pad/truncate to max_len
    Returns: (max_len, n_mfcc)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_len]
    return mfcc
