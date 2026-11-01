import pandas as pd
import pytest
from spam_classifier.dashboard_utils import (
    validate_message, predict_one, classify_batch,
    explain_linear_prediction, load_metrics,
)
from spam_classifier.models import build_pipeline

@pytest.fixture(scope='module')
def trained_model():
    x = [
        'team meeting agenda', 'project review meeting', 'please check report',
        'weekly meeting update', 'status document review', 'share project notes',
        'win free cash reward', 'claim a free prize', 'click bonus promotion',
        'money prizes winner', 'claim bonus money', 'exclusive gift award',
    ]
    y = [0]*6 + [1]*6
    return build_pipeline('logistic_regression', max_features=100).fit(x,y)


def test_empty_message_rejected():
    with pytest.raises(ValueError, match='non-empty'):
        validate_message('   ')


def test_single_prediction_with_score(trained_model):
    result = predict_one(trained_model, 'claim your cash reward')
    assert result['label'] in ('Spam','Not Spam')
    assert 0 <= result['spam_score'] <= 1


def test_batch_produces_labels_and_scores(trained_model):
    result = classify_batch(trained_model, pd.DataFrame({'text': ['meeting agenda', 'free prize']}))
    assert list(result.columns) == ['text','prediction','spam_score_uncalibrated']
    assert len(result)==2
    assert result['spam_score_uncalibrated'].between(0,1).all()


def test_bad_batch_rejected(trained_model):
    with pytest.raises(ValueError, match='text'):
        classify_batch(trained_model, pd.DataFrame({'body': ['hello']}))
    with pytest.raises(ValueError, match='non-empty'):
        classify_batch(trained_model, pd.DataFrame({'text': [' ']}))
    with pytest.raises(ValueError, match='between 1 and'):
        classify_batch(trained_model, pd.DataFrame({'text': ['hi']*501}))


def test_explanations_are_nonempty_for_logistic_regression(trained_model):
    features=explain_linear_prediction(trained_model, 'free cash bonus money')
    assert 0 < len(features) <= 8
    assert set(features.columns) == {'Term','Contribution','Direction'}
    assert all(x in {'Toward spam','Toward not spam'} for x in features['Direction'])


def test_metrics_require_confusion_matrix(tmp_path):
    file=tmp_path/'metrics.json'
    file.write_text('{"selected_model":"logistic_regression", "holdout_metrics":{}, "holdout_confusion_matrix": [[1,2,3]], "cv_comparison":[]}',encoding='utf-8')
    with pytest.raises(ValueError,match='2×2'):
        load_metrics(file)
