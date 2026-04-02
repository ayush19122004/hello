import joblib
import os

model_path = os.path.join("models", "lr_model.pkl")
vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")

try:
    rating_model = joblib.load(model_path)
    rating_vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    print("Error loading model:", e)
    rating_model = None
    rating_vectorizer = None

def predict_rating(text):
    """Return 'High' or 'Low' based on logistic regression output (1 or 0)."""
    if rating_model is None or rating_vectorizer is None:
        return "Low"

    if not text.strip():
        return "Low"

    vec = rating_vectorizer.transform([text])
    pred = rating_model.predict(vec)[0]

    return "High" if pred == 1 else "Low"

