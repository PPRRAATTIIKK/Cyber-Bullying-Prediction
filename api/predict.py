from flask import Flask, request, jsonify
import joblib, re, os, nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

NLTK_DATA = '/tmp/nltk_data'
os.makedirs(NLTK_DATA, exist_ok=True)
nltk.data.path.append(NLTK_DATA)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir=NLTK_DATA, quiet=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model = _vectorizer = _le = None

def _load():
    global _model, _vectorizer, _le
    if _model is None:
        _model      = joblib.load(os.path.join(BASE, 'model.pkl'))
        _vectorizer = joblib.load(os.path.join(BASE, 'vectorizer.pkl'))
        _le         = joblib.load(os.path.join(BASE, 'label_encoder.pkl'))

_stop = None
_stem = SnowballStemmer('english')

def _clean(text):
    global _stop
    if _stop is None:
        _stop = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join([_stem.stem(w) for w in text.split() if w not in _stop and len(w) > 2])

app = Flask(__name__)

@app.after_request
def _cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return r

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        _load()
        vec   = _vectorizer.transform([_clean(text)])
        pred  = _model.predict(vec)[0]
        proba = _model.predict_proba(vec)[0]
        label = _le.inverse_transform([pred])[0]
        is_bully = str(label).lower() in ('yes', '1', 'true', 'bullying')
        return jsonify({
            'is_cyberbullying': is_bully,
            'label':      'Cyberbullying Detected' if is_bully else 'Not Cyberbullying',
            'confidence': round(float(max(proba)) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})
