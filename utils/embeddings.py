# utils/embeddings.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def get_tfidf_features(texts, max_features=1500):
    tfidf = TfidfVectorizer(max_features=max_features)
    features = tfidf.fit_transform(texts)
    return features, tfidf

def get_bert_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, model
