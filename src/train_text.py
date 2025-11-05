# train_text.py  — train on ALL rows (no split)
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

LABELS = ["low", "medium", "high"]

if __name__ == "__main__":
    # load the small sample dataset
    df = pd.read_csv("data/text/train.csv")

    # encode labels (fixed order so it's consistent with the app)
    le = LabelEncoder().fit(LABELS)
    y = le.transform(df["label"])

    # simple, strong baseline model
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("svm", LinearSVC())
    ])

    # fit on ALL rows (no train/val split needed for this demo)
    clf.fit(df["text"], y)

    # save artifacts
    joblib.dump(clf, "models/text_model.joblib")
    joblib.dump(le, "models/label_encoder.joblib")

    print(f"OK: trained on {len(df)} samples and saved models/*.joblib")
