# app/streamlit_app.py (improved UI)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import tempfile
from src.infer import StressSense, LABELS

st.set_page_config(page_title="StressSense", page_icon="🧠", layout="wide")

st.markdown("""
<div style="display:flex;align-items:center;gap:12px">
  <span style="font-size:28px">🧠 <b>StressSense</b></span>
  <span style="opacity:0.7">Voice + Text Stress Detection</span>
</div>
""", unsafe_allow_html=True)
st.caption("For wellness insights only — not a medical device.")

with st.sidebar:
    st.subheader("Fusion & Options")
    alpha = st.slider("Text vs Voice weight (α)", 0.0, 1.0, 0.5, 0.05)
    st.write("α=1 uses only text, α=0 uses only voice.")
    st.divider()
    st.markdown("**Tips**\n- 3–5s WAV, 16kHz mono\n- Try a few different sentences")

model = StressSense()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Input")
    with st.form("input_form"):
        text = st.text_area(
            "Type how you're feeling:",
            height=120,
            placeholder="e.g., I am juggling many deadlines and feel overwhelmed"
        )
        audio_file = st.file_uploader("Upload a short .wav (3–5s, 16kHz mono)", type=["wav"])
        submitted = st.form_submit_button("Analyze", use_container_width=True)

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

            st.subheader("Result")
            st.markdown(f"### Predicted Stress Level: **{label.upper()}**")

            for k, v in zip(LABELS, probs):
                st.write(f"{k.capitalize()} — {v:.2f}")
                st.progress(float(v))

            tips = {
                "low": "You're doing fine. Keep a healthy routine: sleep, hydration, light exercise.",
                "medium": "Consider a 2–5 minute breathing exercise and a short walk.",
                "high": "Try box-breathing (4-4-4-4), reduce caffeine, and take a brief break."
            }
            st.info(tips[label])

            hist = st.session_state.get("history", [])
            hist.insert(0, {"text": text or "(voice only)", "label": label,
                            "probs": [float(x) for x in probs]})
            st.session_state["history"] = hist[:10]

with col_right:
    st.subheader("Recent Analyses")
    hist = st.session_state.get("history", [])
    if not hist:
        st.write("No history yet. Run an analysis to see it here.")
    else:
        for i, h in enumerate(hist, start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {h['label'].upper()}**")
                st.caption(h["text"])
                st.write({k: round(p, 3) for k, p in zip(LABELS, h["probs"])})

st.caption("© 2025 StressSense — built with Streamlit & scikit-learn")

