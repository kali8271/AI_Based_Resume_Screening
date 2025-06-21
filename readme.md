# AI-Based Resume Screening

This project provides an AI-powered tool for screening resumes based on a job description using NLP and embeddings.

## Features
- Preprocesses and embeds resumes using BERT
- Allows you to input a job description
- Ranks resumes by similarity to the job description
- Streamlit web interface

## Setup Instructions

### 1. Clone the repository and navigate to the project directory

```
git clone <repo-url>
cd AI_Based_Resume_Screening
```

### 2. (Recommended) Create and activate a virtual environment

```
python -m venv venv
# On Windows:
venv\Scripts\activate
```

### 3. Downgrade pip (required for textract compatibility)

```
pip install "pip<24.1"
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

### 5. Download NLTK data (first run only)

The app will attempt to download required NLTK data automatically. If you encounter errors, run:

```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 6. Run the Streamlit app

```
streamlit run app.py
```

Then open the provided local URL in your browser.

## Usage
- Upload or use the provided resume dataset in `data/UpdatedResumeDataSet.csv`.
- Paste a job description into the app.
- View the top matching resumes ranked by similarity.

## Notes
- If you encounter issues with `textract` installation, ensure you are using pip version <24.1.
- The app uses BERT embeddings via `sentence-transformers` for semantic matching.
