"""
utils/eda.py
Exploratory Data Analysis — visualizations and statistical summaries.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_question_count_per_subject(df: pd.DataFrame):
    """Bar chart: number of questions per subject."""
    counts = df['subject'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind='bar', color='steelblue', edgecolor='white', ax=ax)
    ax.set_title('Number of Questions per Subject', fontsize=14, fontweight='bold')
    ax.set_xlabel('Subject')
    ax.set_ylabel('Question Count')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'questions_per_subject.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[EDA] Saved: {path}")
    return path


def plot_question_length_distribution(df: pd.DataFrame):
    """Histogram: distribution of question word counts."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['word_count'], bins=30, color='teal', edgecolor='white', alpha=0.8)
    ax.set_title('Question Length Distribution (Word Count)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'question_length_dist.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[EDA] Saved: {path}")
    return path


def plot_question_type_distribution(df: pd.DataFrame):
    """Pie chart: proportion of question types."""
    type_counts = df['question_type'].value_counts()
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
           colors=colors[:len(type_counts)], startangle=140)
    ax.set_title('Question Type Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'question_type_pie.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[EDA] Saved: {path}")
    return path


def plot_year_over_year_trend(df: pd.DataFrame):
    """Line chart: number of questions per year (overall trend)."""
    if 'year' not in df.columns or df['year'].max() == 0:
        print("[EDA] Skipping year-over-year chart: no valid year data.")
        return None
    yearly = df.groupby(['year', 'subject']).size().reset_index(name='count')
    fig, ax = plt.subplots(figsize=(10, 5))
    for subj in yearly['subject'].unique():
        subset = yearly[yearly['subject'] == subj]
        ax.plot(subset['year'], subset['count'], marker='o', label=subj)
    ax.set_title('Questions per Year by Subject', fontsize=13, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Question Count')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'year_over_year.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[EDA] Saved: {path}")
    return path


def generate_wordcloud(df: pd.DataFrame, subject: str = None):
    """Word cloud for a specific subject (or all subjects if None)."""
    if subject:
        text = " ".join(df[df['subject'] == subject]['clean_question'].dropna())
        title = f"Word Cloud — {subject}"
        filename = f"wordcloud_{subject}.png"
    else:
        text = " ".join(df['clean_question'].dropna())
        title = "Word Cloud — All Subjects"
        filename = "wordcloud_all.png"

    if not text.strip():
        print(f"[EDA] Skipping word cloud: no text available.")
        return None

    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap='Blues', max_words=80).generate(text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[EDA] Saved: {path}")
    return path


def plot_top_tfidf_keywords(df: pd.DataFrame, subject: str = None, top_n: int = 20):
    """Bar chart: top N TF-IDF keywords for a subject or all data."""
    corpus = df[df['subject'] == subject]['clean_question'] if subject else df['clean_question']
    corpus = corpus.dropna().tolist()
    if len(corpus) < 2:
        print(f"[EDA] Not enough data for TF-IDF keyword chart.")
        return None

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    mean_scores = tfidf_matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    top_idx = mean_scores.argsort()[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([terms[i] for i in top_idx][::-1],
            [mean_scores[i] for i in top_idx][::-1],
            color='steelblue')
    label = subject if subject else "All Subjects"
    ax.set_title(f'Top {top_n} TF-IDF Keywords — {label}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean TF-IDF Score')
    plt.tight_layout()
    filename = f"tfidf_keywords_{subject or 'all'}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[EDA] Saved: {path}")
    return path


def print_statistical_summary(df: pd.DataFrame):
    """Print key statistics to console."""
    print("\n" + "="*50)
    print("DATASET STATISTICAL SUMMARY")
    print("="*50)
    print(f"Total Questions      : {len(df)}")
    print(f"Subjects             : {df['subject'].nunique()} ({', '.join(df['subject'].unique())})")
    print(f"Years Covered        : {sorted(df['year'].unique())}")
    print(f"Avg Question Length  : {df['word_count'].mean():.1f} words")
    print(f"Max Question Length  : {df['word_count'].max()} words")
    print(f"\nQuestion Type Breakdown:")
    print(df['question_type'].value_counts().to_string())
    print(f"\nQuestions per Subject:")
    print(df['subject'].value_counts().to_string())
    print("="*50 + "\n")


def run_full_eda(df: pd.DataFrame):
    """Run all EDA plots and print summary."""
    print_statistical_summary(df)
    paths = []
    paths.append(plot_question_count_per_subject(df))
    paths.append(plot_question_length_distribution(df))
    paths.append(plot_question_type_distribution(df))
    paths.append(plot_year_over_year_trend(df))
    paths.append(generate_wordcloud(df))
    paths.append(plot_top_tfidf_keywords(df))
    print(f"\n[EDA] All plots saved to: {OUTPUT_DIR}/")
    return [p for p in paths if p]
