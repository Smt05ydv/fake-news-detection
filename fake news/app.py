from flask import Flask, render_template, request
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model = joblib.load(MODEL_DIR / "svm_fake_news_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "svm_tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        title = request.form["title"]
        text = request.form["text"]

        content = title + " " + text

        vector = vectorizer.transform([content])

        prediction_value = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]

        confidence = round(max(probabilities) * 100, 2)

        if prediction_value == 0:
            prediction = "FAKE NEWS"
        else:
            prediction = "REAL NEWS"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
    