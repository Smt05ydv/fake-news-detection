import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"


# =========================
# 2. LOAD ORIGINAL DATA
# =========================

fake = pd.read_csv(DATASET_DIR / "Fake.csv")
true = pd.read_csv(DATASET_DIR / "True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true], ignore_index=True)


# =========================
# 3. CREATE SAME CONTENT
# =========================

data["content"] = (
    data["title"].fillna("") + " " +
    data["text"].fillna("")
)


X = data["content"]
y = data["label"]


# =========================
# 4. RECREATE SAME TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================
# 5. LOAD SAVED MODEL
# =========================

model = joblib.load(
    MODEL_DIR / "fake_news_model.pkl"
)

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)


# =========================
# 6. TRANSFORM TEST DATA
# =========================

X_test_tfidf = vectorizer.transform(X_test)


# =========================
# 7. PREDICT
# =========================

predictions = model.predict(X_test_tfidf)



# Find incorrect predictions
wrong_indices = []

for position, (actual, predicted) in enumerate(zip(y_test, predictions)):
    if actual != predicted:
        wrong_indices.append(position)

print("\n==============================")
print("INCORRECT PREDICTIONS")
print("==============================")

print("Total incorrect:", len(wrong_indices))

for position in wrong_indices[:20]:

    index = X_test.index[position]
    article = data.loc[index]

    actual_label = "FAKE" if article["label"] == 0 else "REAL"
    predicted_label = "FAKE" if predictions[position] == 0 else "REAL"

    print("\n--------------------------------")
    print("Title:", article["title"])
    print("Actual:", actual_label)
    print("Predicted:", predicted_label)


# =========================
# 8. OVERALL ACCURACY
# =========================

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("TEST SET EVALUATION")
print("==============================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# =========================
# 9. SHOW SOME ARTICLES
# =========================

print("\n==============================")
print("SAMPLE TEST PREDICTIONS")
print("==============================")


# Get the original indices from X_test
test_indices = X_test.index[:10]

for i in test_indices:

    article = data.loc[i]

    actual = article["label"]

    # Find position of this index inside X_test
    position = X_test.index.get_loc(i)

    predicted = predictions[position]

    actual_label = "FAKE" if actual == 0 else "REAL"
    predicted_label = "FAKE" if predicted == 0 else "REAL"

    if actual == predicted:
        result = "CORRECT"
    else:
        result = "WRONG"

    print("\n--------------------------------")
    print("Title:", article["title"])
    print("Actual:", actual_label)
    print("Predicted:", predicted_label)
    print("Result:", result)


# =========================
# 10. CLASSIFICATION REPORT
# =========================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Fake", "Real"]
    )
)