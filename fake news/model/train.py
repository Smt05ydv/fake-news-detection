import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# =========================
# 1. LOAD DATASET
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"

fake = pd.read_csv(DATASET_DIR / "Fake.csv")
true = pd.read_csv(DATASET_DIR / "True.csv")

print("Fake articles:", len(fake))
print("Real articles:", len(true))


# =========================
# 2. ADD LABELS
# =========================

fake["label"] = 0
true["label"] = 1


# =========================
# 3. COMBINE DATA
# =========================

data = pd.concat([fake, true], ignore_index=True)

print("Total articles:", len(data))


# =========================
# 4. CREATE CONTENT
# =========================

data["content"] = (
    data["title"].fillna("") + " " +
    data["text"].fillna("")
)


# =========================
# 5. SELECT FEATURES
# =========================

X = data["content"]
y = data["label"]


# =========================
# 6. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 7. TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)


# =========================
# 8. TRAIN MODEL
# =========================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


# =========================
# 9. PREDICT
# =========================

y_pred = model.predict(X_test_tfidf)


# =========================
# 10. EVALUATE
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake", "Real"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# 11. SAVE MODEL
# =========================

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    model,
    MODEL_DIR / "fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

print("\nModel saved successfully!")