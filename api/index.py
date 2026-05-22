import pickle
import re
import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np

app = Flask(__name__)
CORS(app)

nltk.data.path.append('/tmp/nltk_data')
os.makedirs('/tmp/nltk_data', exist_ok=True)
try:
    nltk.download('stopwords', quiet=True, download_dir='/tmp/nltk_data')
except:
    pass

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

model = None
vectorizer = None
load_error = None

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(base_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
        
    print("✅ Model loaded successfully")
except Exception as e:
    load_error = str(e)
    print("❌ Loading error:", load_error)

def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

@app.route("/")
def home():
    return jsonify({
        "status": "Cyber Bullying Prediction API is running",
        "model_loaded": model is not None,
        "note": "Threshold is set low to detect more bullying cases"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded", "details": load_error}), 500
    
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        cleaned = preprocess(text)
        vector = vectorizer.transform([cleaned])
        
        bullying_prob = 0.5
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector)[0]
            # Take the probability of the positive (bullying) class
            bullying_prob = proba[1] if len(proba) > 1 else proba[0]
        else:
            pred = model.predict(vector)[0]
            bullying_prob = 0.9 if pred == 1 else 0.1
        
        # Very low threshold to make it more sensitive
        threshold = 0.22
        is_bullying = bullying_prob >= threshold
        
        result = "Bullying" if is_bullying else "Not Bullying"
        
        return jsonify({
            "text": text,
            "prediction": result,
            "confidence": round(float(bullying_prob) * 100, 1),
            "threshold_used": threshold,
            "cleaned_text": cleaned,
            "raw_probability": round(float(bullying_prob), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
