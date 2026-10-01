import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================
# 1. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"


# =========================
# 2. LOAD DATA
# =========================

fake = pd.read_csv(DATASET_DIR / "Fake.csv")
true = pd.read_csv(DATASET_DIR / "True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat(
    [fake, true],
    ignore_index=True
)


# =========================
# 3. CREATE CONTENT
# =========================

data["content"] = (
    data["title"].fillna("") + " " +
    data["text"].fillna("")
)

X = data["content"]
y = data["label"]


# =========================
# 4. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================
# 5. TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =========================
# 6. TRAIN LINEAR SVM
# =========================

svm = LinearSVC()

model = CalibratedClassifierCV(
    svm,
    cv=3
)

model.fit(
    X_train_tfidf,
    y_train
)


# =========================
# 7. PREDICTION
# =========================

predictions = model.predict(
    X_test_tfidf
)


# =========================
# 8. EVALUATION
# =========================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("LINEAR SVM RESULTS")
print("==============================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Fake", "Real"]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# =========================
# 9. SAVE SVM MODEL
# =========================

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    model,
    MODEL_DIR / "svm_fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "svm_tfidf_vectorizer.pkl"
)

print("\nSVM model saved successfully!")