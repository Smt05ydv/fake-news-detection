import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================
# 1. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"


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
# 4. SAME TRAIN/TEST SPLIT
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
# 6. MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC()
}


# =========================
# 7. TRAIN + EVALUATE
# =========================

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# =========================
# 8. DISPLAY RESULTS
# =========================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("=" * 70)