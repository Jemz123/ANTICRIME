from flask import Flask, request, jsonify, render_template
import pickle
import re
import string
import os

app = Flask(__name__)

# ---------------- LOAD TRAINED MODELS ----------------
MODELS = {
    "news": {"model": "news_model.pkl", "vectorizer": "news_vectorizer.pkl"},
    "crime": {"model": "crime_model.pkl", "vectorizer": "crime_vectorizer.pkl"}
}

def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as file:
            return pickle.load(file)
    return None

for key in MODELS:
    MODELS[key]["model"] = load_model(MODELS[key]["model"])
    MODELS[key]["vectorizer"] = load_model(MODELS[key]["vectorizer"])

# Check if all models loaded successfully
if all(MODELS[key]["model"] and MODELS[key]["vectorizer"] for key in MODELS):
    print("✅ Models and vectorizers loaded successfully!")
else:
    print("❌ Error loading models.")

# ---------------- TEXT PREPROCESSING FUNCTION ----------------
def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(f"[{string.punctuation}]", '', text)  
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

# ---------------- FLASK ROUTES ----------------
@app.route('/')
def home():
    return render_template("index.html")  

@app.route('/predict-news', methods=['POST'])
def predict_news():
    model, vectorizer = MODELS["news"]["model"], MODELS["news"]["vectorizer"]
    if not model or not vectorizer:
        return jsonify({"error": "Fake News model is not loaded."}), 500

    try:
        news_text = request.form.get("news_text", "").strip() or request.json.get("news_text", "").strip()
        if not news_text:
            return jsonify({"error": "No news text provided."}), 400

        processed_text = preprocess_text(news_text)
        transformed_text = vectorizer.transform([processed_text])
        prediction = model.predict(transformed_text)[0]
        result = "Real News" if prediction == 1 else "Fake News"

        return render_template("index.html", news_text=news_text, news_prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-crime', methods=['POST'])
def predict_crime():
    model, vectorizer = MODELS["crime"]["model"], MODELS["crime"]["vectorizer"]
    if not model or not vectorizer:
        return jsonify({"error": "Crime model is not loaded."}), 500

    try:
        crime_text = request.form.get("crime_text", "").strip() or request.json.get("crime_text", "").strip()
        if not crime_text:
            return jsonify({"error": "No crime text provided."}), 400

        processed_text = preprocess_text(crime_text)
        transformed_text = vectorizer.transform([processed_text])
        prediction = model.predict(transformed_text)[0]
        result = "Crime-related" if prediction == 1 else "Not Crime-related"

        return render_template("index.html", crime_text=crime_text, crime_prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN FLASK APP ----------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
