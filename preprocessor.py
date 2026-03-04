"""
utils/preprocessor.py
Handles PDF extraction, text cleaning, and question segmentation.
"""

import os
import re
import pdfplumber
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first run
def download_nltk_data():
    for pkg in ['stopwords', 'wordnet', 'omw-1.4', 'punkt']:
        try:
            nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

download_nltk_data()

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# ─── Keywords that indicate question type ───
QUESTION_TYPE_RULES = {
    'definition': ['define', 'what is', 'what are', 'state', 'list', 'mention', 'identify'],
    'application': ['explain', 'describe', 'discuss', 'illustrate', 'demonstrate', 'apply', 'use'],
    'calculation': ['calculate', 'compute', 'find', 'determine', 'solve', 'evaluate', 'derive'],
    'analysis':   ['compare', 'contrast', 'analyze', 'analyse', 'differentiate', 'justify', 'critically']
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[WARNING] Could not extract {pdf_path}: {e}")
    return text


def segment_questions(text: str) -> list:
    """
    Split exam paper text into individual questions using regex patterns.
    Handles common formats: Q1., 1., (a), (i), QUESTION 1, etc.
    """
    # Normalize line breaks
    text = re.sub(r'\r\n|\r', '\n', text)

    # Split on common question numbering patterns
    patterns = [
        r'\n(?=(?:QUESTION|Question)\s+\d+)',     # QUESTION 1
        r'\n(?=\d+\.\s+[A-Z])',                   # 1. Define...
        r'\n(?=[a-z]\)\s)',                        # a) ...
        r'\n(?=\([a-z]\)\s)',                      # (a) ...
        r'\n(?=\([ivx]+\)\s)',                     # (i), (ii) ...
    ]
    combined = '|'.join(patterns)
    segments = re.split(combined, text)

    questions = []
    for seg in segments:
        seg = seg.strip()
        # Filter out very short segments (headers, page numbers, etc.)
        if len(seg.split()) >= 5:
            questions.append(seg)

    return questions


def classify_question_type(text: str) -> str:
    """Rule-based question type classifier."""
    text_lower = text.lower()
    for q_type, keywords in QUESTION_TYPE_RULES.items():
        if any(kw in text_lower for kw in keywords):
            return q_type
    return 'other'


def clean_text(text: str) -> str:
    """
    Full NLP cleaning pipeline:
    lowercase → remove non-alpha → tokenize → stop word removal → lemmatize
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def parse_filename(filename: str) -> dict:
    """
    Extract metadata from filename.
    Expected format: SubjectCode_Year_Semester.pdf
    Example: COMP322_2022_S1.pdf
    Falls back to defaults if format does not match.
    """
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    return {
        'subject': parts[0] if len(parts) > 0 else 'UNKNOWN',
        'year':    int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
        'semester': parts[2] if len(parts) > 2 else 'S1'
    }


def process_pdf_folder(folder_path: str) -> pd.DataFrame:
    """
    Process all PDFs in a folder.
    Returns a DataFrame with one row per extracted question.

    Columns:
        filename, subject, year, semester,
        raw_question, clean_question, question_type, word_count
    """
    records = []

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"[INFO] No PDF files found in {folder_path}")
        return pd.DataFrame()

    for filename in pdf_files:
        print(f"[INFO] Processing: {filename}")
        pdf_path = os.path.join(folder_path, filename)
        metadata = parse_filename(filename)

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            print(f"[WARNING] Empty text extracted from {filename}, skipping.")
            continue

        questions = segment_questions(raw_text)
        for q in questions:
            cleaned = clean_text(q)
            if len(cleaned.split()) < 3:
                continue  # Skip trivial fragments
            records.append({
                'filename':      filename,
                'subject':       metadata['subject'],
                'year':          metadata['year'],
                'semester':      metadata['semester'],
                'raw_question':  q,
                'clean_question': cleaned,
                'question_type': classify_question_type(q),
                'word_count':    len(q.split())
            })

    df = pd.DataFrame(records)
    print(f"[INFO] Total questions extracted: {len(df)}")
    return df


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved processed data to: {output_path}")


def load_processed_data(csv_path: str) -> pd.DataFrame:
    """Load previously processed data from CSV."""
    return pd.read_csv(csv_path)
