# app/app.py

import streamlit as st
import textract
import joblib
import numpy as np
from utils.preprocessing import preprocess_text
from sentence_transformers import SentenceTransformer
import os
import tempfile
from docx import Document

# Load saved models
model = joblib.load("models/bert_logistic_model.pkl")
bert_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Load locally or from path
label_encoder = joblib.load("models/label_encoder.pkl")

# Streamlit app config
st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("🧠 AI Resume Screening System")
st.markdown("Upload a resume and get the predicted job category.")

# Upload file
uploaded_file = st.file_uploader("Upload Resume File (.pdf or .docx)", type=["pdf", "docx"])

# Skill extractor (simple keyword matching)
custom_skills = ['python', 'sql', 'machine learning', 'deep learning', 'nlp', 'data analysis']

def extract_skills(text):
    return [skill for skill in custom_skills if skill in text.lower()]

# DOCX reader using python-docx
def extract_text_from_docx(path):
    doc = Document(path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Resume extraction handler
def extract_text_from_file(uploaded_file):
    if uploaded_file is not None:
        ext = uploaded_file.name.split('.')[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            if ext == 'docx':
                text = extract_text_from_docx(tmp_path)
            else:
                text = textract.process(tmp_path)
                text = text.decode("utf-8", errors="replace")
        except Exception as e:
            st.error(f"❌ Error extracting text: {e}")
            text = ""
        finally:
            os.remove(tmp_path)

        return text
    return None

# Main app logic
if uploaded_file:
    with st.spinner("Processing Resume..."):
        resume_text = extract_text_from_file(uploaded_file)

if not resume_text.strip():
    st.warning("⚠️ No text found in the uploaded file.")
    st.stop()
else:
    st.info("✅ Text successfully extracted from resume.")

    # Debug: Show raw text
    with st.expander("📄 Raw Resume Text (Before Preprocessing)"):
        st.write(resume_text)

    clean_text = preprocess_text(resume_text)

    if not clean_text.strip():
        st.warning("⚠️ Preprocessing removed all content.")
    else:
        with st.expander("📄 Show Processed Resume Text"):
            st.write(clean_text)
