import pandas as pd
import numpy as np
import re
import string
import pickle  # Import pickle for saving models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ---------------- CRIME DETECTION MODEL ----------------
X_train_crime = ["some crime-related text", "normal text", "criminal activity"]
y_train_crime = [1, 0, 1]  # 1 = Crime-related, 0 = Not crime-related

crime_vectorizer = TfidfVectorizer()
X_train_transformed = crime_vectorizer.fit_transform(X_train_crime)

crime_model = MultinomialNB()
crime_model.fit(X_train_transformed, y_train_crime)

# Save crime detection model & vectorizer
with open("crime_model.pkl", "wb") as model_file:
    pickle.dump(crime_model, model_file)
with open("crime_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(crime_vectorizer, vectorizer_file)

# ---------------- FAKE NEWS DETECTION MODEL ----------------
# Load datasets
fake_df = pd.read_csv(r"C:\Users\Administrator\Desktop\pythonprojects\News _dataset\Fake.csv")
true_df = pd.read_csv(r"C:\Users\Administrator\Desktop\pythonprojects\News _dataset\True.csv")

# Add labels
fake_df["label"] = 0  # Fake news
true_df["label"] = 1  # Real news

# Combine datasets
data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

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
X_train_news, X_test_news, y_train_news, y_test_news = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# Vectorization using TF-IDF
news_vectorizer = TfidfVectorizer()
X_train_tfidf = news_vectorizer.fit_transform(X_train_news)
X_test_tfidf = news_vectorizer.transform(X_test_news)

# Train news classification model
news_model = LogisticRegression()
news_model.fit(X_train_tfidf, y_train_news)

# Save fake news detection model & vectorizer
with open("news_model.pkl", "wb") as model_file:
    pickle.dump(news_model, model_file)
with open("news_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(news_vectorizer, vectorizer_file)

print("Models and vectorizers saved successfully!")
