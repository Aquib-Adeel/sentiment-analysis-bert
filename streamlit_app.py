import streamlit as st
import pandas as pd
import pickle
from transformers import pipeline

# --- Load Baseline (pickle) ---
with open("baseline_model.pkl", "rb") as f:
    clf = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# --- Load Transformer (DistilBERT) ---
sentiment_pipeline = pipeline("sentiment-analysis", model="./results/checkpoint-5000")
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Positive"
}
st.title("Customer Review Sentiment Analysis")

# Dropdown to choose model
model_choice = st.selectbox("Choose model:", 
                            ["Baseline (TF-IDF + Logistic Regression)", 
                             "Transformer (DistilBERT)"])

# --- Single Review Input ---
st.subheader("Single Review Prediction")
user_input = st.text_area("Enter a customer review:")
if user_input:
    if model_choice == "Baseline (TF-IDF + Logistic Regression)":
        X_input = vectorizer.transform([user_input])
        prediction = clf.predict(X_input)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"Sentiment: {sentiment}")
    else:
        prediction = sentiment_pipeline(user_input)[0]
        sentiment = label_map[prediction['label']]   # <-- use mapping here
        st.write(f"Sentiment: {sentiment}, Confidence: {prediction['score']:.2f}")

# --- CSV Upload ---
st.subheader("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV/Excel file with a 'review' column", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, encoding="latin1")  # handles special chars
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None

    if df is not None:
        if model_choice == "Baseline (TF-IDF + Logistic Regression)":
            X_batch = vectorizer.transform(df["review"])
            df["prediction"] = clf.predict(X_batch)
            df["sentiment"] = df["prediction"].apply(lambda x: "Positive" if x == 1 else "Negative")
        else:
            df["raw_prediction"] = df["review"].apply(lambda x: sentiment_pipeline(x)[0])
            df["sentiment"] = df["raw_prediction"].apply(lambda x: "Positive" if x == 1 else "Negative")
            df["confidence"] = df["raw_prediction"].apply(lambda x: x["score"])
        st.write(df.head(10))
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
