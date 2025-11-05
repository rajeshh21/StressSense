import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import tempfile
from src.infer import StressSense, LABELS

st.set_page_config(page_title="StressSense", page_icon="🧠")
st.title("🧠 StressSense — Voice + Text Stress Detection")

st.write("Provide **text**, **voice**, or both. The model will predict your stress level.")

model = StressSense()

with st.form("input_form"):
    text = st.text_area("Type something you're feeling:", height=120, placeholder="e.g., I am juggling many deadlines and feel overwhelmed")
    audio_file = st.file_uploader("Upload a short .wav (3–5s, 16kHz mono recommended)", type=["wav"])
    alpha = st.slider("Text vs Voice weight (alpha)", 0.0, 1.0, 0.5, 0.05)
    submitted = st.form_submit_button("Analyze")

if submitted:
    wav_path = None
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            wav_path = tmp.name

    if not text and not wav_path:
        st.warning("Please provide text, audio, or both.")
    else:
        label, probs = model.fused_predict(text if text else None, wav_path, alpha=alpha)
        st.subheader(f"Predicted Stress Level: **{label.upper()}**")
        st.write({k: float(v) for k, v in zip(LABELS, probs.round(3))})

        tips = {
            "low": "You're doing fine. Keep a healthy routine: sleep, hydration, light exercise.",
            "medium": "Consider a 2–5 minute breathing exercise and a short walk.",
            "high": "Try box-breathing (4-4-4-4), reduce caffeine, and take a brief break."
        }
        st.info(tips[label])

st.caption("Note: This tool is for wellness insights, not medical diagnosis.")
