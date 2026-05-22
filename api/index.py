import re
import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None

# Improved bullying keyword list
BULLYING_KEYWORDS = {
    'stupid', 'idiot', 'dumb', 'ugly', 'fat', 'loser', 'worthless', 'kill', 'die', 
    'hate', 'bitch', 'slut', 'whore', 'faggot', 'retard', 'retarded', 'cunt', 
    'fuck off', 'go die', 'kill yourself', 'worthless piece', 'nobody likes you',
    'you suck', 'you are nothing', 'i hate you', 'ugly bitch', 'stupid bitch'
}

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "model.joblib"))
    vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.joblib"))
    print("✅ ML Model loaded successfully!")
except Exception as e:
    print("⚠️ ML Model failed to load. Using keyword fallback.", str(e))

def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def contains_bullying_keywords(text):
    cleaned = preprocess(text)
    return any(keyword in cleaned for keyword in BULLYING_KEYWORDS)

@app.route("/")
def home():
    return jsonify({
        "status": "Cyber Bullying Prediction API is running",
        "model_loaded": model is not None,
        "using": "ML Model" if model else "Keyword Fallback"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        cleaned = preprocess(text)
        
        # Try ML model first
        if model and vectorizer:
            try:
                vector = vectorizer.transform([cleaned])
                proba = model.predict_proba(vector)[0]
                bullying_prob = float(proba[1])
                threshold = 0.22
                is_bullying = bullying_prob >= threshold
                confidence = round(bullying_prob * 100, 1)
                method = "ML Model"
            except:
                is_bullying = contains_bullying_keywords(text)
                confidence = 65.0 if is_bullying else 35.0
                method = "Keyword Fallback (ML failed)"
        else:
            is_bullying = contains_bullying_keywords(text)
            confidence = 70.0 if is_bullying else 30.0
            method = "Keyword Fallback"

        result = "Bullying" if is_bullying else "Not Bullying"
        
        return jsonify({
            "text": text,
            "prediction": result,
            "confidence": confidence,
            "raw_probability": round(confidence / 100, 3),
            "method": method,
            "cleaned_text": cleaned
        })
        
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run()