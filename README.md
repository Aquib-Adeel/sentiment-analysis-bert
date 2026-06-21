# 📊 Customer Review Sentiment Analysis
A Streamlit web app that performs sentiment analysis on customer reviews using:
- Baseline Model: TF‑IDF + Logistic Regression
- Transformer Model: DistilBERT (fine‑tuned)
  
## 📊 Dataset
- Source: [IMDb Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- Size: 50,000 reviews (balanced between positive and negative)
- Format: Plain text reviews with sentiment labels
- Preprocessing:
  - Text cleaning (lowercasing, punctuation removal)
  - Tokenization
  - TF-IDF vectorization for baseline model
  - Hugging Face tokenizer for DistilBERT
    
## 🔬 Analysis/Methodology:-
1. **Exploratory Data Analysis (EDA):**
   - Checked class distribution
   - Visualized word frequency and sentiment balance
2. **Baseline Model:**
   - TF-IDF features + Logistic Regression
   - Evaluated with accuracy, precision, recall, F1-score
3. **Transformer Model:**
   - Fine-tuned DistilBERT using Hugging Face
   - Used validation split and early stopping
   - Compared performance against baseline
4. **Deployment:**
   - Streamlit app for interactive predictions
   - Supports single review and batch CSV/Excel uploads

## 🚀 Features
- Single review prediction
- Batch CSV/Excel prediction
- Confidence scores
- Downloadable results

## 📘 Usage
- Enter a single review in the text box → get sentiment instantly.
- Upload a CSV/Excel file → batch predictions with downloadable results.
- Switch between baseline and transformer models via dropdown.

## 🌐 Live Demo
- Try the app here: Sentiment Analysis Demo (https://sentiment-analysis-bert-mlmodel.streamlit.app )-deployment-link

## 🧩 Models
- Baseline: TF‑IDF + Logistic Regression (baseline_model.pkl)
- Transformer: DistilBERT fine‑tuned (checkpoint-5000)

## 📂 Project Structure
├── streamlit_app.py  
├── baseline_model.pkl  
├── tfidf_vectorizer.pkl  
├── requirements.txt  
├── README.md  

## 🙌 Acknowledgements
- Hugging Face Transformers
- Scikit‑learn
- Streamlit

## ⚙️ Setup
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
