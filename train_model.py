import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load dataset
df = pd.read_csv("Cyberbullying aggression_parsed_dataset.csv")

# Correct column names based on your dataset
X = df["Text"]
y = df["oh_label"]

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

# Create model folder if not exists
if not os.path.exists("model"):
    os.makedirs("model")

# Save model and vectorizer
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
pickle.dump(model, open("model/cyber_model.pkl", "wb"))

print("MODEL SAVED SUCCESSFULLY!")
