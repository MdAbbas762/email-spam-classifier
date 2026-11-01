"""Testable helpers for the interactive classifier dashboard.

Inference and evaluation always use real persisted model parameters and input data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from .data import load_dataset

MAX_BATCH_ROWS = 500
MAX_MESSAGE_LENGTH = 20_000


def validate_message(message: str) -> str:
    if not isinstance(message, str) or not message.strip():
        raise ValueError('Enter a non-empty email message.')
    message = message.strip()
    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(f'Message is too long (maximum {MAX_MESSAGE_LENGTH:,} characters).')
    return message


def predict_one(pipeline, message: str) -> dict:
    """Return the saved classifier's label and positive-class model score."""
    message = validate_message(message)
    prediction = int(pipeline.predict([message])[0])
    score = float(pipeline.predict_proba([message])[0, 1])
    return {'label': 'Spam' if prediction == 1 else 'Not Spam',
            'spam_score': score, 'prediction': prediction}


def explain_linear_prediction(pipeline, message: str, n: int = 8) -> pd.DataFrame:
    """TF-IDF × linear weights: useful model attribution, not causal explanation."""
    message = validate_message(message)
    if n <= 0:
        raise ValueError('n must be positive')
    columns = ['Term', 'Contribution', 'Direction']
    if not hasattr(pipeline, 'named_steps'):
        return pd.DataFrame(columns=columns)
    vectorizer = pipeline.named_steps.get('tfidf')
    classifier = pipeline.named_steps.get('classifier')
    if vectorizer is None or classifier is None or not hasattr(classifier, 'coef_'):
        return pd.DataFrame(columns=columns)
    vector = vectorizer.transform([message]).tocsr()
    coefs = np.asarray(classifier.coef_)[0]
    terms = vectorizer.get_feature_names_out()
    features = []
    for idx, feature_value in zip(vector.indices, vector.data):
        contribution = float(feature_value * coefs[idx])
        if contribution:
            features.append((str(terms[idx]), contribution))
    features = sorted(features, key=lambda pair: abs(pair[1]), reverse=True)[:n]
    return pd.DataFrame([{'Term': term, 'Contribution': value,
                          'Direction': 'Toward spam' if value > 0 else 'Toward not spam'}
                         for term, value in features], columns=columns)


def classify_batch(pipeline, dataset: pd.DataFrame) -> pd.DataFrame:
    """Classify supplied CSV messages with the same saved model as single prediction."""
    if 'text' not in dataset.columns:
        raise ValueError('Your CSV must include a column named text.')
    if dataset.empty or len(dataset) > MAX_BATCH_ROWS:
        raise ValueError(f'Upload between 1 and {MAX_BATCH_ROWS} messages.')
    texts = [validate_message(value) for value in dataset['text'].tolist()]
    labels = np.asarray(pipeline.predict(texts), dtype=int)
    scores = np.asarray(pipeline.predict_proba(texts)[:, 1], dtype=float)
    return pd.DataFrame({'text': texts,
                         'prediction': np.where(labels == 1, 'Spam', 'Not Spam'),
                         'spam_score_uncalibrated': np.round(scores, 5)})


def load_metrics(path: str | Path) -> dict:
    """Read saved training experiment results; never manufacture metrics."""
    with Path(path).open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    for required in ('selected_model', 'holdout_metrics', 'holdout_confusion_matrix', 'cv_comparison'):
        if required not in data:
            raise ValueError(f'Metrics file is missing {required}. Re-run training.')
    if np.asarray(data['holdout_confusion_matrix']).shape != (2, 2):
        raise ValueError('Expected a 2×2 holdout confusion matrix.')
    return data


def evaluate_saved_model(pipeline, data_path: str | Path, *, seed: int = 42,
                         test_size: float = 0.2) -> dict:
    """Re-score the original held-out partition with the currently loaded model.

    Uses the identical deduplication and stratified split used during training.
    This is an evaluation of a saved fitted model; it neither retrains nor tunes it.
    """
    data, stats = load_dataset(data_path)
    _, holdout_idx = train_test_split(np.arange(len(data)),
                                      test_size=test_size, random_state=seed,
                                      stratify=data['spam'])
    holdout = data.iloc[holdout_idx]
    y_true = holdout['spam'].to_numpy(dtype=int)
    texts = holdout['text'].astype(str).tolist()
    y_pred = np.asarray(pipeline.predict(texts), dtype=int)
    y_score = np.asarray(pipeline.predict_proba(texts)[:, 1], dtype=float)
    if y_score.shape != (len(texts),) or y_pred.shape != y_true.shape:
        raise ValueError('Unexpected model output shape.')
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'average_precision': float(average_precision_score(y_true, y_score)),
        'roc_auc': float(roc_auc_score(y_true, y_score)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    errors = pd.DataFrame({
        'Dataset row': holdout_idx[y_pred != y_true].astype(int),
        'Actual': np.where(y_true[y_pred != y_true] == 1, 'Spam', 'Not Spam'),
        'Predicted': np.where(y_pred[y_pred != y_true] == 1, 'Spam', 'Not Spam'),
        'Model score': np.round(y_score[y_pred != y_true], 4),
    })
    return {'metrics': metrics, 'confusion_matrix': cm,
            'pr_curve': {'recall': recall.tolist(), 'precision': precision.tolist()},
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'errors': errors, 'sample_count': len(y_true),
            'dataset_rows': stats['remaining_rows'], 'seed': seed}
