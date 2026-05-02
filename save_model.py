import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_excel("final_dataset.xlsx")

df["english"] = df["english"].str.lower().str.strip()
df["merged_intent"] = df["merged_intent"].str.lower().str.strip()

X = df["english"]
y = df["merged_intent"]

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=7000)
X_vec = vectorizer.fit_transform(X)

# Train model
model = LinearSVC()
model.fit(X_vec, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved!")