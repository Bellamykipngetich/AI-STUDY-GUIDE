"""
utils/predictor.py
Core prediction logic: topic frequency analysis, trend detection, and priority scoring.
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import os

PAPERS_DIR = Path(__file__).parent / "past_papers"
for file in glob.glob(str(PAPERS_DIR / "*.pdf")):
    # process each paper
    pass

# ─── Topic Frequency Analysis ────────────────────────────────────────────────

def get_topic_frequency(df: pd.DataFrame, subject: str = None) -> pd.DataFrame:
    """
    Count how often each topic (dominant_topic) appears per subject per year.
    Returns a DataFrame with columns: subject, year, topic, count, frequency_pct
    """
    data = df[df['subject'] == subject].copy() if subject else df.copy()

    if 'dominant_topic' not in data.columns:
        raise ValueError("DataFrame must have 'dominant_topic' column. Run feature engineering first.")

    freq = (
        data.groupby(['subject', 'year', 'dominant_topic'])
        .size()
        .reset_index(name='count')
    )
    total_per_year = data.groupby(['subject', 'year']).size().reset_index(name='total')
    freq = freq.merge(total_per_year, on=['subject', 'year'])
    freq['frequency_pct'] = round(freq['count'] / freq['total'] * 100, 2)
    return freq.sort_values(['subject', 'year', 'frequency_pct'], ascending=[True, True, False])


def get_keyword_frequency(df: pd.DataFrame, subject: str = None, top_n: int = 30) -> pd.DataFrame:
    """
    Extract and rank the most frequent meaningful keywords across questions.
    Uses TF-IDF to surface domain-relevant terms (not just raw counts).
    """
    data = df[df['subject'] == subject] if subject else df
    corpus = data['clean_question'].dropna().tolist()
    if len(corpus) < 2:
        return pd.DataFrame(columns=['keyword', 'tfidf_score'])

    vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(corpus)
    mean_scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    df_kw = pd.DataFrame({'keyword': terms, 'tfidf_score': mean_scores})
    return df_kw.sort_values('tfidf_score', ascending=False).head(top_n).reset_index(drop=True)


# ─── Trend Detection ─────────────────────────────────────────────────────────

def compute_topic_trends(freq_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (subject, topic), compute year-over-year trend.
    Trend = slope of frequency_pct over years (linear regression).

    Returns a DataFrame with columns:
        subject, topic, avg_frequency, trend_slope, priority
    """
    results = []
    for (subj, topic), group in freq_df.groupby(['subject', 'dominant_topic']):
        group = group.sort_values('year')
        avg_freq = group['frequency_pct'].mean()
        trend_slope = 0.0

        if len(group) >= 2:
            years = group['year'].values.astype(float)
            freqs = group['frequency_pct'].values
            # Simple linear slope
            years_norm = years - years.mean()
            if years_norm.std() > 0:
                trend_slope = float(np.polyfit(years_norm, freqs, 1)[0])

        # Assign priority based on frequency and trend
        if avg_freq >= 20 or (avg_freq >= 15 and trend_slope > 0):
            priority = "🔴 High Priority"
        elif avg_freq >= 10 or trend_slope > 0.5:
            priority = "🟡 Medium Priority"
        elif trend_slope > 0:
            priority = "🟢 Emerging Topic"
        else:
            priority = "⚪ Low Priority"

        results.append({
            'subject':       subj,
            'dominant_topic': topic,
            'avg_frequency': round(avg_freq, 2),
            'trend_slope':   round(trend_slope, 3),
            'priority':      priority,
            'years_seen':    len(group)
        })

    return pd.DataFrame(results).sort_values('avg_frequency', ascending=False)


# ─── Predict for a New Paper ─────────────────────────────────────────────────

def predict_topics_for_paper(
    questions: list,
    lda,
    dictionary,
    topic_trends_df: pd.DataFrame,
    subject: str = None
) -> pd.DataFrame:
    """
    Given a list of cleaned questions from a new (unseen) paper,
    predict the dominant topic for each question and return ranked predictions.

    Returns a DataFrame: question (truncated), predicted_topic, confidence, priority
    """
    from gensim import corpora as gcorpora

    results = []
    for q in questions:
        bow = dictionary.doc2bow(q.split())
        topic_dist = lda.get_document_topics(bow, minimum_probability=0)
        if topic_dist:
            dominant = max(topic_dist, key=lambda x: x[1])
            topic_id = dominant[0]
            confidence = round(dominant[1] * 100, 1)
        else:
            topic_id = 0
            confidence = 0.0

        # Look up priority from historical trends
        priority = "⚪ Unknown"
        if topic_trends_df is not None and len(topic_trends_df) > 0:
            match = topic_trends_df[topic_trends_df['dominant_topic'] == topic_id]
            if not match.empty:
                priority = match.iloc[0]['priority']

        results.append({
            'question_preview': q[:80] + '...' if len(q) > 80 else q,
            'predicted_topic':  f"Topic {topic_id}",
            'confidence_pct':   confidence,
            'priority':         priority
        })

    return pd.DataFrame(results).sort_values('confidence_pct', ascending=False)


# ─── Study Recommendation Summary ────────────────────────────────────────────

def generate_study_plan(topic_trends_df: pd.DataFrame, subject: str = None) -> dict:
    """
    Generate a structured study plan from topic trends.
    Returns dict with priority buckets and recommended time allocation.
    """
    df = topic_trends_df[topic_trends_df['subject'] == subject] if subject else topic_trends_df

    high    = df[df['priority'].str.startswith('🔴')]['dominant_topic'].tolist()
    medium  = df[df['priority'].str.startswith('🟡')]['dominant_topic'].tolist()
    emerging = df[df['priority'].str.startswith('🟢')]['dominant_topic'].tolist()
    low     = df[df['priority'].str.startswith('⚪')]['dominant_topic'].tolist()

    total = len(high) + len(medium) + len(emerging) + len(low)

    return {
        'high_priority':     high,
        'medium_priority':   medium,
        'emerging_topics':   emerging,
        'low_priority':      low,
        'time_allocation': {
            'High Priority':   '40-50% of study time',
            'Medium Priority': '30-35% of study time',
            'Emerging Topics': '10-15% of study time',
            'Low Priority':    '5% or less'
        },
        'total_topics': total
    }
