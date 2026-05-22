import re
import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

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
    model = joblib.load(os.path.join(base_dir, "model.joblib"))
    vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.joblib"))
    print("✅ Model loaded successfully!")
except Exception as e:
    load_error = str(e)
    print("❌ Error loading model:", load_error)

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
        "status": "API is running",
        "model_loaded": model is not None,
        "error": load_error
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded", "details": load_error}), 500
    
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    cleaned = preprocess(text)
    vector = vectorizer.transform([cleaned])
    
    proba = model.predict_proba(vector)[0]
    bullying_prob = float(proba[1])
    threshold = 0.22
    
    is_bullying = bullying_prob >= threshold
    result = "Bullying" if is_bullying else "Not Bullying"
    
    return jsonify({
        "text": text,
        "prediction": result,
        "confidence": round(bullying_prob * 100, 1),
        "raw_probability": round(bullying_prob, 4),
        "threshold_used": threshold,
        "cleaned_text": cleaned
    })

if __name__ == "__main__":
    app.run()