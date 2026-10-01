import joblib
from pathlib import Path


# =========================
# 1. LOAD SVM MODEL
# =========================

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "svm_fake_news_model.pkl"
)

vectorizer = joblib.load(
    BASE_DIR / "svm_tfidf_vectorizer.pkl"
)


# =========================
# 2. GET NEWS
# =========================

print("\n==============================")
print("       FAKE NEWS DETECTOR")
print("==============================\n")

title = input("Enter news title: ")

print("\nEnter the news article.")
print("Press ENTER twice when finished:\n")

lines = []

while True:
    line = input()

    if line == "":
        break

    lines.append(line)

text = " ".join(lines)


# =========================
# 3. CHECK INPUT
# =========================

if len(text.strip()) < 50:
    print("\nPlease enter a longer news article.")
    exit()


# =========================
# 4. COMBINE TITLE + TEXT
# =========================

content = title + " " + text


# =========================
# 5. TF-IDF
# =========================

content_tfidf = vectorizer.transform([content])


# =========================
# 6. PREDICT
# =========================

prediction = model.predict(content_tfidf)[0]

probabilities = model.predict_proba(content_tfidf)[0]

confidence = max(probabilities) * 100


# =========================
# 7. RESULT
# =========================

print("\n==============================")
print("           RESULT")
print("==============================")

if prediction == 0:
    print("Prediction : FAKE NEWS")
else:
    print("Prediction : REAL NEWS")

print(f"Confidence : {confidence:.2f}%")

print("==============================\n")