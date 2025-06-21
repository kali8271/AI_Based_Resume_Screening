import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_text
from utils.embeddings import get_bert_embeddings
import numpy as np

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
job_desc = st.text_area("Paste the job description here:")

if job_desc:
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
        st.write(df.iloc[idx]['Resume'])
        st.write(f"Similarity Score: {sims[idx]:.4f}")
        st.markdown("---") 