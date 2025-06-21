# AI-Based Resume Screening

This project provides an AI-powered tool for screening resumes based on a job description using NLP and BERT embeddings.

## Features
- Preprocesses and embeds resumes using BERT
- Allows you to input or upload a job description
- Ranks resumes by similarity to the job description
- Streamlit web interface

## Project Structure
- **app.py**: Main Streamlit app for screening resumes in batch against a job description.
- **data/**: Contains the resume dataset (`UpdatedResumeDataSet.csv`).
- **models/**: Pretrained models and vectorizers.
- **utils/**: Utility scripts for preprocessing, embeddings, and model evaluation.
- **notebooks/**: Jupyter notebooks for EDA and model training.

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
- The app loads resumes from `data/UpdatedResumeDataSet.csv`.
- Paste a job description or upload a PDF/DOCX file with the job description.
- The app will extract key skills, compute semantic similarity, and display the top matching resumes with highlighted skills and suggested roles.
- Use the slider to adjust the number of top resumes shown.

## Data Format
- The resume dataset should be a CSV file with at least two columns: `Category` (target job role) and `Resume` (resume text).
- Example:

  | Category      | Resume                        |
  |--------------|-------------------------------|
  | Data Science | Skills: Python, pandas, ...   |

## Model Training & Notebooks
- Notebooks in the `notebooks/` folder show how models were trained:
  - `eda.ipynb`: Exploratory data analysis.
  - `tfidf_baseline_models.ipynb`: Baseline models using TF-IDF features.
  - `bert_embeddings_models.ipynb`: Training and saving BERT-based models.
- Pretrained models are saved in the `models/` directory and loaded by the app.

## Troubleshooting & FAQ
- **Textract install issues**: Ensure you are using pip version <24.1.
- **NLTK errors**: Manually download required NLTK data as shown above.
- **Large model download**: The first run may download the BERT model (`all-MiniLM-L6-v2`).
- **File not found errors**: Ensure all required files in `data/` and `models/` are present.
- **Windows file locks**: If you get errors about temp files, ensure no other process is using them.

## Notes
- The app uses BERT embeddings via `sentence-transformers` for semantic matching.
- For best results, use clean, well-formatted resumes and job descriptions.
