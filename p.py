import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load dataset
df = pd.read_csv("data/amazon_combined_20k.csv")
df.fillna("", inplace=True)

# ✅ Create combined_meta from title, brand, category
df["combined_meta"] = (
    df["title"].astype(str) + " " +
    df["brand"].astype(str) + " " +
    df["category"].astype(str)
)

# Label: high rating (1) if overall >= 3, else 0
df["label"] = (df["overall"] >= 3).astype(int)

# TF-IDF on combined_meta
X_text = df["combined_meta"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X_text)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model and vectorizer
os.makedirs("assets", exist_ok=True)
joblib.dump(model, "assets/lr_model.pkl")
joblib.dump(vectorizer, "assets/tfidf_vectorizer.pkl")

print("✅ Model trained and saved using combined_meta.")
