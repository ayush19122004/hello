import pickle

with open("models/lr_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    sentiment_vectorizer = pickle.load(f)

def predict_sentiment(text):
    vec = sentiment_vectorizer.transform([text])
    pred = sentiment_model.predict(vec)[0]
    return "Positive 😊" if pred == 1 else "Negative 😠"
