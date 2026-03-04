"""
train.py
Main pipeline: run this script to process data, train models, and save artifacts.

Usage:
    python train.py --data_dir data/raw --output_dir data/processed
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessor import process_pdf_folder, save_processed_data, load_processed_data
from utils.eda import run_full_eda
from utils.features import build_feature_matrix, save_artifacts, get_dominant_topic
from utils.model import (
    split_data, train_all_models, evaluate_all_models,
    plot_confusion_matrix, plot_model_comparison, plot_feature_importance,
    save_model, load_model
)


def run_pipeline(data_dir: str, processed_csv: str = "data/processed/questions.csv", skip_eda: bool = False):
    print("\n" + "="*60)
    print("  EXAM QUESTION PREDICTOR — TRAINING PIPELINE")
    print("="*60)

    # ── STEP 1: Data Extraction ──────────────────────────────────
    print("\n[STEP 1] Extracting questions from PDFs...")
    if os.path.exists(processed_csv):
        print(f"  Found existing processed data at {processed_csv}, loading...")
        df = load_processed_data(processed_csv)
    else:
        df = process_pdf_folder(data_dir)
        if df.empty:
            print("[ERROR] No data extracted. Please add PDF files to the data/raw directory.")
            print("  File naming convention: SubjectCode_Year_Semester.pdf")
            print("  Example: COMP322_2022_S1.pdf")
            sys.exit(1)
        os.makedirs(os.path.dirname(processed_csv), exist_ok=True)
        save_processed_data(df, processed_csv)

    print(f"  Dataset: {len(df)} questions across {df['subject'].nunique()} subjects")

    # ── STEP 2: EDA ──────────────────────────────────────────────
    if not skip_eda:
        print("\n[STEP 2] Running Exploratory Data Analysis...")
        run_full_eda(df)
    else:
        print("\n[STEP 2] Skipping EDA (--skip_eda flag set)")

    # ── STEP 3: Feature Engineering ──────────────────────────────
    print("\n[STEP 3] Building feature matrix...")
    X, y, vectorizer, lda, dictionary, label_encoder = build_feature_matrix(df, num_topics=10)

    # Add dominant topic back to dataframe for predictor use
    df['dominant_topic'] = get_dominant_topic(df['clean_question'].fillna('').tolist(), lda, dictionary)
    save_processed_data(df, processed_csv)  # Re-save with dominant_topic column
    save_artifacts(vectorizer, lda, dictionary, label_encoder)

    # ── STEP 4: Train / Test Split ────────────────────────────────
    print("\n[STEP 4] Splitting data...")
    if len(np.unique(y)) < 2:
        print("[ERROR] Need at least 2 topic classes. Add more varied data.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ── STEP 5: Model Training ────────────────────────────────────
    print("\n[STEP 5] Training models...")
    trained_models, scaler = train_all_models(X_train, y_train)
    joblib.dump(scaler, "models/scaler.pkl")

    # ── STEP 6: Evaluation ────────────────────────────────────────
    print("\n[STEP 6] Evaluating models...")
    results_df = evaluate_all_models(trained_models, X_test, y_test, scaler)
    print("\n[RESULTS] Model Comparison:")
    print(results_df.to_string())

    # Save comparison chart
    plot_model_comparison(results_df)

    # ── STEP 7: Select & Save Best Model ─────────────────────────
    best_model_name = results_df['f1_score'].idxmax()
    best_model = trained_models[best_model_name]
    print(f"\n[STEP 7] Best model: {best_model_name} (F1={results_df.loc[best_model_name,'f1_score']})")

    # Confusion matrix for best model
    plot_confusion_matrix(best_model, X_test, y_test, model_name=best_model_name, scaler=scaler)

    # Feature importance (if supported)
    feature_names = vectorizer.get_feature_names_out().tolist()
    plot_feature_importance(best_model, feature_names, model_name=best_model_name)

    # Save best model
    save_model(best_model, "best_model.pkl")
    results_df.to_csv("models/evaluation_results.csv")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  Best Model : {best_model_name}")
    print(f"  F1 Score   : {results_df.loc[best_model_name,'f1_score']}")
    print(f"  Artifacts  : models/")
    print(f"  EDA Plots  : eda_outputs/")
    print("="*60 + "\n")

    return trained_models, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exam Question Predictor — Training Pipeline")
    parser.add_argument('--data_dir',  default='data/raw',             help='Folder containing PDF files')
    parser.add_argument('--processed', default='data/processed/questions.csv', help='Path to save/load processed CSV')
    parser.add_argument('--skip_eda',  action='store_true',            help='Skip EDA visualizations')
    args = parser.parse_args()

    run_pipeline(args.data_dir, args.processed, args.skip_eda)
