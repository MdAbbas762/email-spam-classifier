import pandas as pd
import pytest
from spam_classifier.data import load_dataset
from spam_classifier.models import build_pipeline
from spam_classifier.text import tokenize
from app import predict
import joblib

def test_tokenizer_deterministic_and_stems():
    assert tokenize('RUNNING! running and RUNS') == tokenize('running running and runs')
    assert 'run' in tokenize('running')

def test_dataset_removes_duplicate_text_before_split(tmp_path):
    p=tmp_path/'emails.csv'
    pd.DataFrame({'text':['hello','hello','win money','please call'],
                  'spam':[0,0,1,0]}).to_csv(p,index=False)
    df,stats=load_dataset(p)
    assert len(df)==3 and stats['removed_duplicate_texts']==1

def test_conflicting_duplicate_labels_rejected(tmp_path):
    p=tmp_path/'emails.csv'
    pd.DataFrame({'text':['same','same','other'], 'spam':[0,1,1]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='Conflicting'):
        load_dataset(p)

@pytest.mark.parametrize('model_name',['multinomial_nb','bernoulli_nb','logistic_regression'])
def test_pipeline_fits_and_predicts(model_name):
    X=['team meeting tomorrow','please review document',
       'free prize win money','claim your free reward',
       'let us discuss plans','send updated notes',
       'bonus money reward','win gift prizes']
    y=[0,0,1,1,0,0,1,1]
    p=build_pipeline(model_name,seed=2,max_features=80).fit(X,y)
    assert len(p.predict(['free money']))==1
    assert p.predict_proba(['free money']).shape==(1,2)

def test_saved_pipeline_predicts_without_refitting(tmp_path):
    X=['regular team meeting','project status update','free cash reward','win free money']
    y=[0,0,1,1]
    p=build_pipeline('multinomial_nb',max_features=50).fit(X,y)
    path=tmp_path/'spam_pipeline.joblib'
    joblib.dump(p,path)
    out=predict('win free cash',path)
    assert out['label'] in ('Spam','Not Spam')
    assert 0 <= out['predicted_spam_probability'] <= 1

def test_empty_or_missing_label_rejected(tmp_path):
    p=tmp_path/'emails.csv'
    pd.DataFrame({'text':['x','y'],'spam':[0,9]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='either 0'):
        load_dataset(p)


def test_fractional_label_rejected(tmp_path):
    p=tmp_path/'emails.csv'
    pd.DataFrame({'text':['x','y'],'spam':[0.5,1]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='either 0'):
        load_dataset(p)


def test_training_pipeline_excludes_test_only_vocabulary(tmp_path):
    from sklearn.model_selection import train_test_split
    from spam_classifier.train import train_and_evaluate
    import numpy as np
    # Construct distinct documents and insert an unseen token in one guaranteed holdout row.
    rows=[]
    for i in range(24):
        is_spam=i%2
        text=(('free prize bonus claim '+str(i)) if is_spam else
              ('weekly meeting document schedule '+str(i)))
        rows.append({'text':text,'spam':is_spam})
    train_idx,test_idx=train_test_split(np.arange(24),test_size=.25,random_state=42,
                                       stratify=[r['spam'] for r in rows])
    idx=int(test_idx[0]);rows[idx]['text'] += ' qwertyzxyz'
    csv=tmp_path/'emails.csv'
    pd.DataFrame(rows).to_csv(csv,index=False)
    report=train_and_evaluate(csv,tmp_path/'artifacts',folds=2,test_size=.25,
                              max_features=100,seed=42)
    assert report['split']['test_size_rows'] == 6
    fitted=joblib.load(tmp_path/'artifacts'/'spam_pipeline.joblib')
    vocab=fitted.named_steps['tfidf'].vocabulary_
    assert 'qwertyzxyz' not in vocab
    assert (tmp_path/'artifacts/reports/confusion_matrix.png').exists()
    assert (tmp_path/'artifacts/reports/holdout_errors.csv').exists()
    assert 'text' not in pd.read_csv(tmp_path/'artifacts/reports/holdout_errors.csv').columns
