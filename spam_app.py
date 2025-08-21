import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io, zipfile, requests, os, re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ======================
# Backend functions
# ======================

UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
MODEL_PATH = "spam_classifier.joblib"


def download_sms_spam_dataset() -> pd.DataFrame:
    resp = requests.get(UCI_ZIP_URL, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("SMSSpamCollection") as f:
            data = [line.decode("utf-8", errors="ignore").strip().split("\t", 1) for line in f]
            rows = [(lbl, txt) for lbl, txt in data if len(lbl) > 0 and len(txt) > 0]
    df = pd.DataFrame(rows, columns=["label", "text"])
    return df


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98
        )),
        ("clf", MultinomialNB(alpha=0.1))
    ])


def train_model(df: pd.DataFrame):
    df["text"] = df["text"].astype(str).map(normalize_text)
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=["ham", "spam"])
    report = classification_report(y_test, preds, output_dict=True)

    joblib.dump(pipe, MODEL_PATH)

    return pipe, acc, cm, report


def load_model():
    return joblib.load(MODEL_PATH)


def predict_text(model: Pipeline, text: str) -> str:
    return model.predict([normalize_text(text)])[0]


# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="📧 Spam Classifier", page_icon="📨", layout="centered")

st.title("📧 Spam Email/SMS Classifier")
st.write("A simple **TF-IDF + Naive Bayes** model trained on the UCI SMS Spam dataset.")

# Sidebar
st.sidebar.header("⚙️ Options")
if st.sidebar.button("Train Model (UCI Dataset)"):
    st.write("⏳ Training on UCI SMS Spam dataset...")
    df = download_sms_spam_dataset()
    st.write(f"✅ Loaded dataset with {len(df)} rows. Spam ratio: {np.mean(df['label']=='spam'):.2%}")
    model, acc, cm, report = train_model(df)
    st.success(f"Model trained! Accuracy: {acc:.4f}")
    st.write("### Classification Report")
    st.json(report)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

# Prediction Section
st.write("---")
st.subheader("🔍 Try It Out")

if os.path.exists(MODEL_PATH):
    model = load_model()
    user_input = st.text_area("Enter an email/SMS message:", height=120)

    if st.button("Classify"):
        if user_input.strip():
            label = predict_text(model, user_input)
            if label == "spam":
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: HAM (Not Spam)")
        else:
            st.warning("Please enter a message first.")
else:
    st.info("ℹ️ No trained model found. Please train using the sidebar first.")
