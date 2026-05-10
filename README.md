<div align="center">

# 🛡️ Cyber Bullying Prediction

**Detecting cyberbullying in social media posts using NLP & Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-154360?style=for-the-badge)](https://www.nltk.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<img src="https://img.shields.io/badge/Dataset-13%2C147%20Posts-6366f1?style=flat-square" />
<img src="https://img.shields.io/badge/Best%20Accuracy-95.08%25-22c55e?style=flat-square" />
<img src="https://img.shields.io/badge/Models-4%20Classifiers-f59e0b?style=flat-square" />

</div>

---

## 📌 Overview

Cyberbullying is a growing problem on social media platforms. This project builds an end-to-end **NLP pipeline** to automatically detect cyberbullying in Formspring Q&A posts — from raw messy text to trained classifiers with evaluation metrics and visualisations.

The pipeline covers:
- 🧹 **Text preprocessing** — HTML stripping, stopword removal, Snowball stemming
- 📊 **Feature extraction** — TF-IDF vectorisation (unigrams + bigrams)
- 🤖 **Model training** — 4 classifiers with GridSearchCV hyperparameter tuning
- 📈 **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrices

---

## 📂 Project Structure

```
Cyber-Bullying-Prediction/
│
├── 📓 cyber_bullying.ipynb     # Main notebook — full pipeline
├── 📄 Formspring.csv           # Dataset (place here before running)
└── 📋 README.md
```

---

## 🗂️ Dataset

**Formspring.csv** — 13,147 Q&A posts scraped from the Formspring social platform.

| Feature | Description |
|---|---|
| `text` | Raw Q&A post text |
| `answer` | Label — `Yes` (bullying) / `No` (not bullying) |

> ⚠️ **Class Imbalance:** ~93.5% of posts are labelled *Not Bullying* and ~6.5% *Bullying*. This inflates accuracy scores and suppresses recall — addressed in the Results section.

---

## ⚙️ Pipeline

```
Raw Text
   │
   ▼
┌─────────────────────────────┐
│   TEXT PREPROCESSING        │
│  • Lowercase                │
│  • Remove HTML / URLs       │
│  • Remove punctuation       │
│  • Remove stopwords (NLTK)  │
│  • Snowball Stemming        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   TF-IDF VECTORISATION      │
│  • Unigrams + Bigrams       │
│  • Top 10,000 features      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   LABEL ENCODING            │
│  No → 0   |   Yes → 1      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   TRAIN / TEST SPLIT        │
│  70% train  |  30% test     │
│  stratify=True              │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   MODEL TRAINING            │
│  SVM · NaiveBayes           │
│  DecisionTree · LogReg      │
│  + GridSearchCV tuning      │
└─────────────┬───────────────┘
              │
              ▼
     EVALUATION & PLOTS
```

---

## 🤖 Models & Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| 🥇 **SVM (Sigmoid kernel)** | **95.08%** | **80.52%** | 25.73% | 38.97% |
| 🥈 **Multinomial Naive Bayes** | 94.02% | 53.62% | 15.35% | 23.74% |
| 🥉 **Decision Tree** | 93.11% | 41.88% | **33.20%** | 37.03% |
| **Logistic Regression** | 94.35% | 90.24% | 14.45% | 24.92% |

> 💡 **Why is recall low?** The dataset is heavily imbalanced — 93.5% of samples are "Not Bullying". All models learn to predict the majority class. Improving recall requires techniques like **SMOTE oversampling**, `class_weight='balanced'`, or **transformer-based models** (BERT).

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` `numpy` | Data loading & manipulation |
| `NLTK` | Stopword removal, Snowball stemming |
| `scikit-learn` | TF-IDF, classifiers, GridSearchCV, metrics |
| `matplotlib` | Visualisations & plots |
| `Jupyter Notebook` | Interactive development |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/Cyber-Bullying-Prediction.git
cd Cyber-Bullying-Prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy nltk scikit-learn matplotlib jupyter
```

### 3. Download NLTK data
```python
import nltk
nltk.download('stopwords')
```

### 4. Add the dataset
Place `Formspring.csv` in the project root directory.

### 5. Run the notebook
```bash
jupyter notebook cyber_bullying.ipynb
```

> 🌐 **Running on Google Colab?** Upload `Formspring.csv` using:
> ```python
> from google.colab import files
> uploaded = files.upload()
> ```

---

## 📊 Visualisations

The notebook generates the following plots:

- 📉 **Class distribution** — bar + pie chart of label imbalance
- 📊 **Metric comparison** — grouped bar chart (Accuracy / Precision / Recall / F1) across all models
- 🟦 **Confusion matrices** — 2×2 heatmaps for each classifier
- 🏆 **F1-score ranking** — horizontal bar chart showing best model

---

## 🔮 Future Improvements

- [ ] Handle class imbalance with **SMOTE** or `class_weight='balanced'`
- [ ] Add **word embeddings** — Word2Vec, GloVe, or FastText
- [ ] Fine-tune a **BERT / DistilBERT** transformer model
- [ ] Build a **Streamlit web app** for live cyberbullying detection
- [ ] Add **cross-validation** for more robust evaluation

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ for safer online spaces

⭐ Star this repo if you found it useful!

</div>
