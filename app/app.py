# app/app.py

import streamlit as st
import textract
import joblib
import numpy as np
from utils.preprocessing import preprocess_text
from sentence_transformers import SentenceTransformer

# Load saved models
model = joblib.load("models/bert_logistic_model.pkl")
bert_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Load locally or from path
label_encoder = joblib.load("models/label_encoder.pkl")

st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("🧠 AI Resume Screening System")
st.markdown("Upload a resume and get the predicted job category.")

# Upload file
uploaded_file = st.file_uploader("Upload Resume File (.pdf or .docx)", type=["pdf", "docx"])

def extract_text(file):
    text = textract.process(file).decode('utf-8')
    return text

# Skill extractor (optional)
custom_skills = ['python', 'sql', 'machine learning', 'deep learning', 'nlp', 'data analysis']

def extract_skills(text):
    return [skill for skill in custom_skills if skill in text.lower()]

if uploaded_file:
    with st.spinner("Processing..."):
        resume_text = extract_text(uploaded_file)
        clean_text = preprocess_text(resume_text)
        embedding = bert_encoder.encode([clean_text])
        prediction = model.predict(embedding)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        extracted_skills = extract_skills(clean_text)

    st.success(f"🎯 **Predicted Job Category:** {predicted_label}")
    
    if extracted_skills:
        st.info("💼 **Skills Detected:** " + ", ".join(extracted_skills))
    else:
        st.warning("No major skills detected from our list.")

    # Optionally show raw resume
    with st.expander("📄 Show Processed Resume Text"):
        st.write(clean_text)
