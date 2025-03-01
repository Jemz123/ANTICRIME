import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# ---------------- FAKE NEWS DETECTION MODEL ----------------
# Load datasets (fixing file path issue)
fake_df = pd.read_csv("C:\\Users\\Administrator\\Desktop\\pythonprojects\\News_dataset\\Fake.csv")

true_df = pd.read_csv(r"C:\\Users\\Administrator\\Desktop\\pythonprojects\\News_dataset\\True.csv")

# Add labels
fake_df["label"] = 0  # Fake news
true_df["label"] = 1  # Real news

# Combine datasets and shuffle
data = pd.concat([fake_df, true_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(f"[{string.punctuation}]", '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing
data["text"] = data["text"].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save Fake News Model & Vectorizer
with open("news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("news_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Fake News Model and Vectorizer saved successfully!")

# ---------------- CRIME DETECTION MODEL ----------------
# Sample Crime-related dataset (expand this with a real dataset)
X_train_crime = [
    "The suspect was caught with illegal substances",
    "A robbery occurred last night at a local bank",
    "There was a shooting incident in downtown",
    "Family went on vacation to Hawaii",
    "The mayor announced a new city development project",
]
y_train_crime = [1, 1, 1, 0, 0]  # 1 = Crime-related, 0 = Not crime-related

# Vectorization
crime_vectorizer = TfidfVectorizer()
X_train_crime_transformed = crime_vectorizer.fit_transform(X_train_crime)

# Train Naive Bayes Model
crime_model = MultinomialNB()
crime_model.fit(X_train_crime_transformed, y_train_crime)

# Save Crime Detection Model & Vectorizer
with open("crime_model.pkl", "wb") as model_file:
    pickle.dump(crime_model, model_file)
with open("crime_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(crime_vectorizer, vectorizer_file)

print("✅ Crime Detection Model and Vectorizer saved successfully!")
