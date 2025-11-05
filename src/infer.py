import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils_audio import audio_to_features

LABELS = ["low", "medium", "high"]

class StressSense:
    def __init__(self):
        self.text_model = joblib.load("models/text_model.joblib")
        # audio model optional; load if present
        try:
            self.audio_model = joblib.load("models/audio_model.joblib")
        except Exception:
            self.audio_model = None
        self.le: LabelEncoder = joblib.load("models/label_encoder.joblib")

    def predict_text(self, text: str) -> np.ndarray:
        dec = self.text_model.decision_function([text])  # shape (1, n_classes)
        dec = dec - dec.min(axis=1, keepdims=True)
        probs = dec / (dec.sum(axis=1, keepdims=True) + 1e-9)
        return probs[0]

    def predict_audio(self, wav_path: str) -> np.ndarray:
        if self.audio_model is None:
            raise RuntimeError("Audio model not found. Train with: python -m src.train_audio")
        feats = audio_to_features(wav_path)
        probs = self.audio_model.predict_proba([feats])[0]
        return probs

    def fused_predict(self, text: str | None, wav_path: str | None, alpha: float = 0.5) -> tuple[str, np.ndarray]:
        p = np.zeros(3)
        used = 0.0
        if text:
            pt = self.predict_text(text)
            p += alpha * pt
            used += alpha
        if wav_path:
            if self.audio_model is None:
                # fallback to text-only if audio model missing
                pass
            else:
                pa = self.predict_audio(wav_path)
                p += (1 - alpha) * pa
                used += (1 - alpha)
        if used > 0:
            p = p / used
        idx = int(np.argmax(p))
        return LABELS[idx], p
