import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_text
from utils.embeddings import get_bert_embeddings
import numpy as np
import textract
from sklearn.feature_extraction.text import TfidfVectorizer
import re

st.title("AI-Based Resume Screening")

# Load dataset
data_path = 'data/UpdatedResumeDataSet.csv'
@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    return df

df = load_data()

st.write(f"Loaded {len(df)} resumes.")

# Preprocess resumes
@st.cache_data
def preprocess_resumes(resume_texts):
    return [preprocess_text(text) for text in resume_texts]

st.subheader("Preprocessing resumes...")
preprocessed_resumes = preprocess_resumes(df['Resume'])

# Generate embeddings
@st.cache_resource
def embed_resumes(texts):
    embeddings, _ = get_bert_embeddings(texts)
    return embeddings

st.subheader("Generating embeddings...")
resume_embeddings = embed_resumes(preprocessed_resumes)

# User input for job description
st.header("Screen Resumes by Job Description")

# File uploader for PDF/DOCX
uploaded_file = st.file_uploader("Or upload a job description (PDF/DOCX):", type=["pdf", "docx"])
job_desc = st.text_area("Paste the job description here:")

def extract_text_from_file(uploaded_file):
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with open("temp_uploaded_file", "wb") as f:
            f.write(uploaded_file.read())
        try:
            text = textract.process("temp_uploaded_file")
            text = text.decode("utf-8", errors="ignore")
        except Exception as e:
            os.remove("temp_uploaded_file")
            st.error(f"Error extracting text: {e}")
            return None
        os.remove("temp_uploaded_file")
        return text
    return None

if uploaded_file is not None:
    job_desc = extract_text_from_file(uploaded_file)
    st.success("Text extracted from uploaded file.")
    st.text_area("Extracted job description:", job_desc, height=200)

if job_desc:
    # --- Extract key skills from job description ---
    def extract_skills(text, top_n=10):
        # Simple keyword extraction using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=50)
        X = vectorizer.fit_transform([text])
        feature_array = vectorizer.get_feature_names_out()
        tfidf_sorting = X.toarray().flatten().argsort()[::-1]
        top_skills = [feature_array[i] for i in tfidf_sorting[:top_n]]
        return top_skills

    key_skills = extract_skills(job_desc)
    st.markdown('**Key Skills Extracted from Job Description:**')
    st.write(', '.join(key_skills))

    preprocessed_job = preprocess_text(job_desc)
    job_emb, _ = get_bert_embeddings([preprocessed_job])
    # Compute cosine similarity
    def cosine_similarity(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b, a)
    sims = cosine_similarity(job_emb[0], resume_embeddings)
    top_n = st.slider("Number of top resumes to show:", 1, 20, 5)
    top_indices = np.argsort(sims)[::-1][:top_n]
    st.subheader(f"Top {top_n} Matching Resumes:")
    for idx in top_indices:
        st.markdown(f"**Candidate {idx+1}**")
        resume_text = df.iloc[idx]['Resume']
        # Highlight key skills in resume
        highlighted_resume = resume_text
        for skill in key_skills:
            # Use regex for case-insensitive highlighting
            highlighted_resume = re.sub(f"(?i)({re.escape(skill)})", r'<mark>\1</mark>', highlighted_resume)
        st.markdown(highlighted_resume, unsafe_allow_html=True)
        st.write(f"Similarity Score: {sims[idx]:.4f}")
        # Show the most suitable role (Category)
        st.write(f"**Suggested Role:** {df.iloc[idx]['Category']}")
        st.markdown("---") 