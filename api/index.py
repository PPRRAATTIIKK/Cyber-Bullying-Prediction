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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer
model = None
vectorizer = None

try:
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

@app.route("/", methods=["GET"])
def home():
    status = "✅ Model Loaded" if model and vectorizer else "❌ Model Not Loaded"
    return jsonify({
        "status": "Cyber Bullying Prediction API is running!",
        "model_status": status
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded properly"}), 500
    
    data = request.get_json()
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    result = "Bullying" if prediction == 1 else "Not Bullying"
    
    return jsonify({
        "text": text,
        "prediction": result,
        "cleaned_text": cleaned
    })


