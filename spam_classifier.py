import argparse
import io
import os
import re
import zipfile
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
MODEL_PATH = "spam_classifier.joblib"


def download_sms_spam_dataset() -> pd.DataFrame:
    """
    Downloads the UCI SMS Spam Collection and returns a DataFrame with columns:
    text (str), label (str in {'ham','spam'}).
    """
    resp = requests.get(UCI_ZIP_URL, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("SMSSpamCollection") as f:
            # File is tab-separated: label \t text
            data = [line.decode("utf-8", errors="ignore").strip().split("\t", 1) for line in f]
            rows = [(lbl, txt) for lbl, txt in data if len(lbl) > 0 and len(txt) > 0]

    df = pd.DataFrame(rows, columns=["label", "text"])
    # Basic sanity
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    return df


def load_from_csv(path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Loads a generic CSV with 'text' and 'label' columns (or specify your own).
    label values should be 'spam' or 'ham' (case-insensitive).
    """
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns.")
    df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    # Map common variants
    df["label"] = df["label"].replace({"spam": "spam", "ham": "ham", "0": "ham", "1": "spam"})
    df = df[df["label"].isin(["spam", "ham"])].dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df = df.reset_index(drop=True)
    return df


def normalize_text(s: str) -> str:
    """
    Lightweight normalization: strip, collapse whitespace.
    (TF-IDF handles casing/stopwords; keep it fast.)
    """
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_pipeline() -> Pipeline:
    """
    TF-IDF (unigrams + bigrams) + MultinomialNB.
    Fast to train, strong baseline for spam.
    """
    return Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,            # ignore extremely rare tokens
            max_df=0.98          # ignore extremely common tokens
        )),
        ("clf", MultinomialNB(alpha=0.1))  # slight smoothing
    ])


def train_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[Pipeline, dict]:
    # Clean text
    df["text"] = df["text"].astype(str).map(normalize_text)
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=["ham", "spam"])

    report = classification_report(y_test, preds, digits=3, target_names=["ham", "spam"])
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": report
    }
    return pipe, metrics


def save_model(model: Pipeline, path: str = MODEL_PATH) -> None:
    joblib.dump(model, path)


def load_model(path: str = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_text(model: Pipeline, text: str) -> str:
    return model.predict([normalize_text(text)])[0]


def main():
    parser = argparse.ArgumentParser(description="Spam (SMS/Email) Classifier - train and predict.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional: path to your CSV with columns [text,label]. If omitted, downloads UCI SMS dataset.")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name in your CSV (default: text)")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name in your CSV (default: label)")
    parser.add_argument("--only-predict", action="store_true",
                        help="Skip training and use saved model to predict interactively.")
    args = parser.parse_args()

    if args.only_predict:
        if not os.path.exists(MODEL_PATH):
            print(f"[!] {MODEL_PATH} not found. Train first (run without --only-predict).")
            return
        model = load_model(MODEL_PATH)
    else:
        if args.csv:
            print(f"[i] Loading dataset from CSV: {args.csv}")
            df = load_from_csv(args.csv, text_col=args.text_col, label_col=args.label_col)
        else:
            print("[i] Downloading UCI SMS Spam Collection (fastest start)...")
            df = download_sms_spam_dataset()
        print(f"[i] Dataset size: {len(df)} rows. Spam ratio: {np.mean(df['label']=='spam'):.2%}")

        model, metrics = train_evaluate(df)
        print("\n=== Metrics ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Confusion Matrix [rows/cols=ham,spam]:")
        print(np.array(metrics["confusion_matrix"]))
        print("\nClassification Report:")
        print(metrics["report"])

        save_model(model)
        print(f"[i] Saved model to {MODEL_PATH}")

    # Interactive prediction loop
    print("\nType a message/email to classify (or just press Enter to exit).")
    model = load_model(MODEL_PATH) if args.only_predict else model
    while True:
        try:
            txt = input("> ")
        except EOFError:
            break
        if not txt.strip():
            break
        label = predict_text(model, txt)
        print(f"→ Predicted: {label.upper()}")
        if label == "spam":
            print("   (Tip: contains spam-like patterns. Be cautious with links, OTPs, prizes, time pressure.)")


if __name__ == "__main__":
    main()
