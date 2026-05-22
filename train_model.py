"""
train_model.py
──────────────
Run this script in Google Colab (or locally) ONCE to train the model
and export model.pkl + vectorizer.pkl.

In Colab, add a cell at the end of your notebook and run:
    exec(open('train_model.py').read())
OR simply copy-paste the contents into a new Colab cell.
"""

import pandas as pd
import numpy as np
import re
import joblib
import os

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── 1. Load dataset ────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('Formspring.csv', encoding='latin-1')

# Keep only the two relevant columns (adjust names if yours differ)
# Common column names in this dataset: 'text'/'post'/'q' and 'answer'/'label'
text_col  = None
label_col = None

for c in df.columns:
    cl = c.lower().strip()
    if cl in ('text', 'post', 'q', 'question', 'content'):
        text_col = c
    if cl in ('answer', 'label', 'bullying', 'cyberbullying'):
        label_col = c

if text_col is None or label_col is None:
    # fallback: first two columns
    text_col, label_col = df.columns[0], df.columns[1]

print(f"Using columns: text='{text_col}', label='{label_col}'")
df = df[[text_col, label_col]].dropna()
df.columns = ['text', 'label']

# ── 2. Pre-process ─────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
stemmer    = SnowballStemmer('english')

def clean(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML
    text = re.sub(r'http\S+|www\S+', ' ', text)   # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)         # keep letters only
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

print("Cleaning text...")
df['clean'] = df['text'].apply(clean)

# ── 3. Encode labels ───────────────────────────────────────────────────────────
le = LabelEncoder()
df['encoded'] = le.fit_transform(df['label'].str.strip().str.lower()
                                   .map(lambda x: 'yes' if x in ('yes','1','true','bullying') else 'no'))

# ── 4. TF-IDF ─────────────────────────────────────────────────────────────────
print("Fitting TF-IDF...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean'])
y = df['encoded']

# ── 5. Train SVM (best model per README) ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("Training SVM...")
model = SVC(kernel='sigmoid', C=1, probability=True, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# ── 6. Save artefacts ─────────────────────────────────────────────────────────
joblib.dump(model,      'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(le,         'label_encoder.pkl')

print("\n✅  Saved: model.pkl, vectorizer.pkl, label_encoder.pkl")
print("   Download these 3 files from Colab and put them in your project root.")
