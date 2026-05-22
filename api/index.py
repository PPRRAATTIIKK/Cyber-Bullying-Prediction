import pickle
import re
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

# Load model at cold start
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    vectorizer = None
    print(f"❌ Model load failed: {e}")

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
        "status": "Cyber Bullying Prediction API is live on Vercel!",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    cleaned = preprocess(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    result = "Bullying" if prediction == 1 else "Not Bullying"
    
    return jsonify({
        "text": text,
        "prediction": result,
        "cleaned": cleaned
    })

# This is required for Vercel
app.debug = False