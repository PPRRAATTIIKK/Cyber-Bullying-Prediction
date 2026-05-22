import pickle
import re
import os
import traceback
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
label_encoder = None
load_error = "No error"
file_status = {}

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_status['base_dir'] = base_dir
    
    for filename in ['model.pkl', 'vectorizer.pkl', 'label_encoder.pkl']:
        path = os.path.join(base_dir, filename)
        file_status[filename] = {
            "exists": os.path.exists(path),
            "size": os.path.getsize(path) if os.path.exists(path) else 0
        }
    
    with open(os.path.join(base_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    if os.path.exists(os.path.join(base_dir, "label_encoder.pkl")):
        with open(os.path.join(base_dir, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
    
    print("✅ All models loaded successfully!")
except Exception as e:
    load_error = str(e)
    print("❌ CRITICAL ERROR:", load_error)
    print(traceback.format_exc())

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
        "file_status": file_status,
        "error": load_error,
        "base_dir": os.path.dirname(os.path.abspath(__file__))
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({
            "error": "Model failed to load",
            "details": load_error,
            "file_status": file_status
        }), 500
    
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        cleaned = preprocess(text)
        vector = vectorizer.transform([cleaned])
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector)[0]
            bullying_prob = float(proba[1] if len(proba) > 1 else proba[0])
        else:
            pred = int(model.predict(vector)[0])
            bullying_prob = 0.9 if pred == 1 else 0.1
        
        threshold = 0.25
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
