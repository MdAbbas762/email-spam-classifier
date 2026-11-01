import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from spam_classifier.dashboard_utils import evaluate_saved_model
from spam_classifier.models import build_pipeline
from spam_classifier.data import load_dataset


def test_live_holdout_matches_deterministic_split(tmp_path):
    rows = []
    for i in range(25):
        rows.append({'text': f'project update document number {i}', 'spam': 0})
        rows.append({'text': f'claim bonus prize number {i}', 'spam': 1})
    path = tmp_path / 'emails.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    data, _ = load_dataset(path)
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=.2,
                                           random_state=42, stratify=data['spam'])
    model = build_pipeline('logistic_regression', max_features=200)
    model.fit(data.iloc[train_idx]['text'].tolist(), data.iloc[train_idx]['spam'].to_numpy())
    report = evaluate_saved_model(model, path)
    assert report['sample_count'] == len(test_idx)
    assert sum(map(sum, report['confusion_matrix'])) == len(test_idx)
    assert set(report['metrics']) == {'accuracy', 'precision', 'recall', 'f1', 'average_precision', 'roc_auc'}
    assert report['errors'].shape[0] == sum(report['confusion_matrix'][0][1:]) + report['confusion_matrix'][1][0]
    assert 0 <= report['metrics']['f1'] <= 1
