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

# Fix for Vercel read-only filesystem
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
load_error = None

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(base_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(base_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
        
    print("✅ Model, vectorizer, and label encoder loaded successfully!")
    print("Classes:", label_encoder.classes_)
except Exception as e:
    load_error = str(e)
    print("❌ Loading failed:")
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
        "status": "Cyber Bullying Prediction API is running",
        "model_loaded": model is not None,
        "classes": list(label_encoder.classes_) if label_encoder else None,
        "error": load_error
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
        prediction = model.predict(vector)[0]
        
        # Use label encoder to get proper label
        if label_encoder:
            result = label_encoder.inverse_transform([prediction])[0]
            is_bullying = result.lower() == "yes" or result == 1
        else:
            is_bullying = prediction == 1
            result = "Yes" if is_bullying else "No"
        
        return jsonify({
            "text": text,
            "prediction": "Bullying" if is_bullying else "Not Bullying",
            "raw_prediction": int(prediction),
            "cleaned_text": cleaned,
            "label_used": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
