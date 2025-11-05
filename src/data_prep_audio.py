import os
import pandas as pd
import numpy as np
import librosa

SR = 16000
N_MFCC = 13

def extract_mfcc(path: str, sr: int = SR, n_mfcc: int = N_MFCC):
    y, sr = librosa.load(path, sr=sr, mono=True)
    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Aggregate over time (mean + std) to fixed-size vector
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat

def load_audio_dataset(metadata_csv: str, clips_dir: str):
    meta = pd.read_csv(metadata_csv)
    X, y = [], []
    for _, row in meta.iterrows():
        fpath = os.path.join(clips_dir, row["file"])
        X.append(extract_mfcc(fpath))
        y.append(row["label"].strip().lower())
    return np.vstack(X), np.array(y)
