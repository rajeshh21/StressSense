#!/usr/bin/env bash
# ==== Setup + Train + Run (macOS/Linux) ====
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
python -m src.train_text
echo
echo "Training finished. Starting the app..."
streamlit run app/streamlit_app.py
