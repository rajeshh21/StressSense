import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.data_prep_audio import load_audio_dataset

LABELS = ["low", "medium", "high"]

if __name__ == "__main__":
    X, y = load_audio_dataset("data/audio/metadata.csv", "data/audio/clips")

    le = LabelEncoder()
    le.fit(LABELS)
    y_enc = le.transform(y)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True))
    ])

    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_va)
    print(classification_report(y_va, preds, target_names=LABELS))

    joblib.dump(clf, "models/audio_model.joblib")
    joblib.dump(le, "models/label_encoder.joblib")
