"""
utils/features.py
Feature engineering: TF-IDF, LDA topic modeling, and feature matrix assembly.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim import corpora
from gensim.models import LdaModel

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── TF-IDF ─────────────────────────────────────────────────────────────────

def build_tfidf(corpus: list, max_features: int = 1000, ngram_range=(1, 2)):
    """Fit TF-IDF vectorizer and return (matrix, vectorizer)."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,       # Apply log normalization to TF
        min_df=2                 # Ignore terms that appear in fewer than 2 docs
    )
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer


def transform_tfidf(corpus: list, vectorizer: TfidfVectorizer):
    """Transform new corpus using a fitted TF-IDF vectorizer."""
    return vectorizer.transform(corpus)


# ─── LDA Topic Modeling ─────────────────────────────────────────────────────

def build_lda(corpus: list, num_topics: int = 10, passes: int = 10):
    """
    Fit LDA topic model.
    Returns (lda_model, dictionary, topic_labels).
    """
    tokenized = [text.split() for text in corpus]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    lda = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha='auto',
        eta='auto'
    )

    # Print discovered topics
    print("\n[LDA] Discovered Topics:")
    for idx, topic in lda.print_topics(num_words=6):
        print(f"  Topic {idx}: {topic}")

    return lda, dictionary


def get_lda_features(corpus: list, lda: LdaModel, dictionary: corpora.Dictionary, num_topics: int):
    """
    Convert a corpus into an (n_docs x num_topics) matrix of topic probabilities.
    """
    features = np.zeros((len(corpus), num_topics))
    for i, text in enumerate(corpus):
        bow = dictionary.doc2bow(text.split())
        topic_dist = lda.get_document_topics(bow, minimum_probability=0)
        for topic_id, prob in topic_dist:
            features[i, topic_id] = prob
    return features


def get_dominant_topic(corpus: list, lda: LdaModel, dictionary: corpora.Dictionary) -> list:
    """Return the dominant topic ID for each document."""
    dominant = []
    for text in corpus:
        bow = dictionary.doc2bow(text.split())
        topic_dist = lda.get_document_topics(bow, minimum_probability=0)
        if topic_dist:
            dominant.append(max(topic_dist, key=lambda x: x[1])[0])
        else:
            dominant.append(0)
    return dominant


# ─── Full Feature Matrix Assembly ──────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, num_topics: int = 10):
    """
    Build complete feature matrix combining:
      - TF-IDF (sparse, converted to dense for combination)
      - LDA topic distribution
      - Word count
      - Year (ordinal)
      - Subject encoding

    Returns: (X, y, vectorizer, lda_model, dictionary, label_encoder)
    """
    corpus = df['clean_question'].fillna('').tolist()

    print("[FEATURES] Building TF-IDF features...")
    tfidf_matrix, vectorizer = build_tfidf(corpus, max_features=500)

    print("[FEATURES] Building LDA topic model...")
    lda, dictionary = build_lda(corpus, num_topics=num_topics)

    print("[FEATURES] Extracting LDA topic features...")
    lda_features = get_lda_features(corpus, lda, dictionary, num_topics)

    # Encode subject labels
    le = LabelEncoder()
    subject_encoded = le.fit_transform(df['subject'].fillna('UNKNOWN'))

    # Combine all features
    tfidf_dense = tfidf_matrix.toarray()
    year_norm = (df['year'].fillna(0) - df['year'].min()) / max(df['year'].max() - df['year'].min(), 1)
    word_count_norm = df['word_count'].fillna(0) / df['word_count'].max()

    X = np.hstack([
        tfidf_dense,
        lda_features,
        subject_encoded.reshape(-1, 1),
        year_norm.values.reshape(-1, 1),
        word_count_norm.values.reshape(-1, 1)
    ])

    # Target: dominant topic as classification label
    y = np.array(get_dominant_topic(corpus, lda, dictionary))

    print(f"[FEATURES] Feature matrix shape: {X.shape}")
    print(f"[FEATURES] Target classes: {np.unique(y)}")

    return X, y, vectorizer, lda, dictionary, le


# ─── Save / Load ─────────────────────────────────────────────────────────────

def save_artifacts(vectorizer, lda, dictionary, label_encoder):
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    lda.save(os.path.join(MODELS_DIR, 'lda_model'))
    dictionary.save(os.path.join(MODELS_DIR, 'lda_dictionary'))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    print("[FEATURES] All artifacts saved.")


def load_artifacts():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    lda = LdaModel.load(os.path.join(MODELS_DIR, 'lda_model'))
    dictionary = corpora.Dictionary.load(os.path.join(MODELS_DIR, 'lda_dictionary'))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    return vectorizer, lda, dictionary, label_encoder
