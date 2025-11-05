import numpy as np
from src.data_prep_audio import extract_mfcc

def audio_to_features(file_path: str) -> np.ndarray:
    return extract_mfcc(file_path)
