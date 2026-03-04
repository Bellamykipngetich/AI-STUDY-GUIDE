"""
utils/model.py
Model training, hyperparameter tuning, evaluation, and persistence.
"""

import numpy as np
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

MODELS_DIR = "models"
EDA_DIR    = "eda_outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)


# ─── Train / Test Split ──────────────────────────────────────────────────────

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ─── Model Definitions ───────────────────────────────────────────────────────

def get_models():
    """Return dict of model name → (estimator, param_grid) tuples."""
    return {
        "Naive Bayes": (
            MultinomialNB(),
            {'alpha': [0.1, 0.5, 1.0, 2.0]}
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'saga']}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
        ),
        "XGBoost": (
            xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [4, 6]}
        )
    }


# ─── Training ────────────────────────────────────────────────────────────────

def train_all_models(X_train, y_train):
    """
    Train all models with GridSearchCV.
    Returns dict of model_name → best_estimator.
    """
    # Scale features to [0,1] for Naive Bayes compatibility
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    trained_models = {}
    for name, (estimator, param_grid) in get_models().items():
        print(f"\n[TRAIN] Training {name}...")
        grid = GridSearchCV(
            estimator, param_grid,
            cv=5, scoring='f1_macro',
            n_jobs=-1, verbose=0
        )
        grid.fit(X_train_scaled, y_train)
        trained_models[name] = grid.best_estimator_
        print(f"  Best params : {grid.best_params_}")
        print(f"  CV F1 Score : {grid.best_score_:.4f}")

    return trained_models, scaler


def train_best_model(X_train, y_train, model_name="XGBoost"):
    """Train only the best model (for quick runs)."""
    models, scaler = train_all_models(X_train, y_train)
    return models.get(model_name, list(models.values())[-1]), scaler, models


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model", scaler=None):
    """Evaluate a single model and return metrics dict."""
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)

    metrics = {
        'model':     model_name,
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'f1_score':  round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
    }

    # ROC-AUC (requires probability predictions)
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X)
            if len(np.unique(y_test)) > 1:
                metrics['roc_auc'] = round(
                    roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'), 4
                )
        except Exception:
            metrics['roc_auc'] = 'N/A'
    else:
        metrics['roc_auc'] = 'N/A'

    print(f"\n[EVAL] {model_name}")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1 Score  : {metrics['f1_score']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    return metrics


def evaluate_all_models(trained_models, X_test, y_test, scaler):
    """Evaluate all models and return a comparison DataFrame."""
    results = []
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name, scaler=scaler)
        results.append(metrics)
    return pd.DataFrame(results).set_index('model')


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, model_name="Model", scaler=None):
    """Save confusion matrix heatmap."""
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.5, linecolor='white')
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Topic')
    ax.set_ylabel('Actual Topic')
    plt.tight_layout()
    path = os.path.join(EDA_DIR, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[EVAL] Confusion matrix saved: {path}")
    return path


def plot_model_comparison(results_df: pd.DataFrame):
    """Bar chart comparing all models across key metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    available = [m for m in metrics if m in results_df.columns]
    df_plot = results_df[available].astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_plot))
    width = 0.2
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

    for i, metric in enumerate(available):
        ax.bar(x + i * width, df_plot[metric], width, label=metric.capitalize(), color=colors[i])

    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(df_plot.index, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(EDA_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[EVAL] Model comparison chart saved: {path}")
    return path


def plot_feature_importance(model, feature_names: list, top_n: int = 20, model_name="Model"):
    """Bar chart of top N feature importances (Random Forest / XGBoost only)."""
    if not hasattr(model, 'feature_importances_'):
        print(f"[EVAL] Feature importance not available for {model_name}")
        return None

    importances = model.feature_importances_
    if len(feature_names) > len(importances):
        feature_names = feature_names[:len(importances)]
    elif len(feature_names) < len(importances):
        feature_names = feature_names + [f"feature_{i}" for i in range(len(importances) - len(feature_names))]

    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in idx][::-1],
            [importances[i] for i in idx][::-1], color='steelblue')
    ax.set_title(f'Top {top_n} Feature Importances — {model_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    path = os.path.join(EDA_DIR, f'feature_importance_{model_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[EVAL] Feature importance chart saved: {path}")
    return path


# ─── Save / Load Best Model ──────────────────────────────────────────────────

def save_model(model, filename="best_model.pkl"):
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"[MODEL] Saved: {path}")


def load_model(filename="best_model.pkl"):
    return joblib.load(os.path.join(MODELS_DIR, filename))
