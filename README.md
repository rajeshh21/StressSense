# StressSense – Multimodal Stress Detection (Voice + Text)

https://stresssenserajesh.streamlit.app/

StressSense detects stress levels from **text** and **voice**. It predicts **Low / Medium / High** stress and offers quick tips. Built for learning + showcasing AIML on a resume.

## Features
- Text emotion using TF‑IDF + Linear SVM
- Voice stress cues via MFCC features + RBF SVM
- Late fusion (weighted combine) when both are present
- Streamlit UI (upload .wav or type text)

## Stack
Python, scikit‑learn, librosa, Streamlit, joblib

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Provide small sample datasets:
- `data/text/train.csv` with columns: `text,label` (labels in {low, medium, high})
- (Optional) `data/audio/clips/` WAV files and `data/audio/metadata.csv` mapping `file,label`

## Train
```bash
python -m src.train_text
# Optional if you added audio:
# python -m src.train_audio
```

## Run
```bash
streamlit run app/streamlit_app.py
```

## Notes
This is a wellness demo, not a medical device.
