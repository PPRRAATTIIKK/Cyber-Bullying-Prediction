from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import joblib
import os

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "model.joblib"))
    vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.joblib"))
    print("Model loaded successfully")
except Exception as e:
    print("Model load failed:", str(e))

def is_bullying_text(text):
    bad_words = ['stupid', 'ugly', 'hate', 'die', 'kill', 'worthless', 'loser', 'idiot', 'fuck', 'bitch', 'shit']
    text_lower = text.lower()
    return any(word in text_lower for word in bad_words)

@app.route("/")
def home():
    return jsonify({
        "status": "API is running",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Simple rule-based fallback
        if is_bullying_text(text):
            return jsonify({
                "text": text,
                "prediction": "Bullying",
                "confidence": 75.0,
                "raw_probability": 0.75,
                "note": "Detected using keyword matching"
            })
        
        # Try ML model if available
        if model and vectorizer:
            # Add your ML prediction here later
            pass
            
        return jsonify({
            "text": text,
            "prediction": "Not Bullying",
            "confidence": 65.0,
            "raw_probability": 0.35
        })
        
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run()