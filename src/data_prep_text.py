import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["low", "medium", "high"]

def load_text_data(train_csv: str, test_csv: str | None = None):
    df = pd.read_csv(train_csv)
    if test_csv:
        df_test = pd.read_csv(test_csv)
        return df, df_test
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    return train_df, val_df
