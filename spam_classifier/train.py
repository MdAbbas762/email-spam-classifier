"""Reproducible, leakage-safe training, CV model selection, and holdout evaluation.

Usage: python -m spam_classifier.train --data emails.csv --out artifacts
Keep holdout rows completely outside CV and model selection.
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, roc_curve)

from .data import load_dataset
from .models import MODEL_NAMES, build_pipeline

SCORING = {'accuracy':'accuracy','precision':'precision','recall':'recall',
           'f1':'f1','average_precision':'average_precision'}

def classification_metrics(y_true, y_pred, scores):
    return {
        'accuracy':float(accuracy_score(y_true,y_pred)),
        'precision':float(precision_score(y_true,y_pred,zero_division=0)),
        'recall':float(recall_score(y_true,y_pred,zero_division=0)),
        'f1':float(f1_score(y_true,y_pred,zero_division=0)),
        'average_precision':float(average_precision_score(y_true,scores)),
        'roc_auc':float(roc_auc_score(y_true,scores)),
    }

def save_plots(y_true, predictions, scores, reports: Path):
    cm=confusion_matrix(y_true,predictions,labels=[0,1])
    fig,ax=plt.subplots(figsize=(5,4))
    im=ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Ham','Spam'],
           yticklabels=['Ham','Spam'],xlabel='Predicted',ylabel='Actual',
           title='Holdout confusion matrix')
    fig.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(reports/'confusion_matrix.png',dpi=160);plt.close(fig)

    p,r,_=precision_recall_curve(y_true,scores)
    fig,ax=plt.subplots(figsize=(5,4));ax.plot(r,p)
    ax.set(xlabel='Recall (spam)',ylabel='Precision (spam)',
           title='Holdout precision–recall curve',xlim=(0,1),ylim=(0,1))
    fig.tight_layout();fig.savefig(reports/'precision_recall_curve.png',dpi=160);plt.close(fig)

    fpr,tpr,_=roc_curve(y_true,scores)
    fig,ax=plt.subplots(figsize=(5,4));ax.plot(fpr,tpr,label='Classifier')
    ax.plot([0,1],[0,1],'--',label='Random baseline')
    ax.set(xlabel='False-positive rate',ylabel='True-positive rate',
           title='Holdout ROC curve',xlim=(0,1),ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(reports/'roc_curve.png',dpi=160);plt.close(fig)
    return cm

def top_features(pipeline, model_name, out: Path):
    features=pipeline.named_steps['tfidf'].get_feature_names_out()
    classifier=pipeline.named_steps['classifier']
    if model_name == 'logistic_regression':
        weights=classifier.coef_[0]
    else:
        weights=classifier.feature_log_prob_[1]-classifier.feature_log_prob_[0]
    n=min(15,len(weights))
    ranked=np.argsort(weights)
    records=[{'class':'spam','feature':str(features[i]),'weight':float(weights[i])}
             for i in ranked[-n:][::-1]]
    records +=[{'class':'ham','feature':str(features[i]),'weight':float(weights[i])}
              for i in ranked[:n]]
    pd.DataFrame(records).to_csv(out/'top_features.csv',index=False)

def train_and_evaluate(data_path: str | Path, out: str | Path,
                       *, seed: int=42, test_size: float=.20,
                       folds: int=3, max_features: int=5000) -> dict:
    if not (0.05 < test_size < 0.5):
        raise ValueError('test_size must be between 0.05 and 0.5')
    if folds < 2:
        raise ValueError('folds must be at least 2')
    out=Path(out);reports=out/'reports'; reports.mkdir(parents=True,exist_ok=True)
    data,stats=load_dataset(data_path)
    # The row indices are preserved for local-only error auditing.
    train_idx, test_idx=train_test_split(np.arange(len(data)),test_size=test_size,
                                        random_state=seed,stratify=data['spam'])
    X_train=data.loc[train_idx,'text'].astype(str).to_numpy()
    X_test=data.loc[test_idx,'text'].astype(str).to_numpy()
    y_train=data.loc[train_idx,'spam'].to_numpy()
    y_test=data.loc[test_idx,'spam'].to_numpy()
    if min(np.bincount(y_train))<folds:
        raise ValueError('Too few examples per class for the chosen CV folds')
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    leaderboard=[]
    print(f'Dataset: {stats["remaining_rows"]} unique emails; '
          f'train={len(train_idx)}, held-out test={len(test_idx)}')
    for model_name in MODEL_NAMES:
        pipe=build_pipeline(model_name,seed,max_features)
        result=cross_validate(pipe,X_train,y_train,cv=cv,scoring=SCORING,
                              n_jobs=1,error_score='raise')
        row={'model':model_name}
        for metric in SCORING:
            scores=result['test_'+metric]
            row[f'cv_{metric}_mean']=float(np.mean(scores))
            row[f'cv_{metric}_std']=float(np.std(scores))
        leaderboard.append(row)
        print(f'CV {model_name}: F1={row["cv_f1_mean"]:.4f} '
              f'± {row["cv_f1_std"]:.4f}')
    # Select using training CV only; the held-out test is evaluated once.
    best=max(leaderboard,key=lambda r:(r['cv_f1_mean'],r['cv_average_precision_mean']))
    name=best['model']
    model=build_pipeline(name,seed,max_features)
    model.fit(X_train,y_train)
    train_pred=model.predict(X_train)
    train_score=model.predict_proba(X_train)[:,1]
    pred=model.predict(X_test)
    score=model.predict_proba(X_test)[:,1]
    train_metrics=classification_metrics(y_train,train_pred,train_score)
    test_metrics=classification_metrics(y_test,pred,score)
    cm=save_plots(y_test,pred,score,reports)
    top_features(model,name,reports)
    pd.DataFrame(leaderboard).sort_values('cv_f1_mean',ascending=False).to_csv(
        reports/'cv_model_comparison.csv',index=False)
    # No raw email text is written to generated reports by default.
    errors=[]
    for idx,true,guess,prob in zip(test_idx,y_test,pred,score):
        if int(true)!=int(guess):
            errors.append({'cleaned_row_index':int(idx),'actual_label':int(true),
                           'predicted_label':int(guess),
                           'predicted_spam_probability':round(float(prob),6),
                           'error_type':'false_positive' if guess==1 else 'false_negative'})
    pd.DataFrame(errors,columns=['cleaned_row_index','actual_label','predicted_label',
                                 'predicted_spam_probability','error_type']).to_csv(
        reports/'holdout_errors.csv',index=False)
    summary={
        'experiment':'TF-IDF + classification (stratified train-only CV)',
        'run_at_utc':datetime.now(timezone.utc).isoformat(),
        'dataset':stats,
        'split':{'seed':seed,'test_size':test_size,'cv_folds':folds,
                 'train_size':len(train_idx),'test_size_rows':len(test_idx),
                 'train_spam':int(y_train.sum()),'test_spam':int(y_test.sum())},
        'selection_metric':'mean train-fold cross-validation F1 (spam class)',
        'selected_model':name,
        'cv_comparison':leaderboard,
        'training_metrics':train_metrics,
        'holdout_metrics':test_metrics,
        'holdout_confusion_matrix':cm.tolist(),
        'holdout_false_positives':int(cm[0,1]),
        'holdout_false_negatives':int(cm[1,0]),
        'software':{'python':platform.python_version(),'scikit_learn':sklearn.__version__,
                    'pandas':pd.__version__},
        'notes':['TF-IDF is fitted only within training folds and final training split.',
                 'Test set is never used to tune hyperparameters or select a model.',
                 'Random stratification is appropriate for this static email snapshot; '
                 'no claim about temporal generalization or real inbox deployment.',
                 'Predicted probabilities are uncalibrated; do not treat as confidence guarantees.']
    }
    (reports/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    model_path=out/'spam_pipeline.joblib'
    joblib.dump(model,model_path)
    (out/'model_card.md').write_text(
        f'# Model card — Email spam classifier\n\n'
        f'- Selected: `{name}` via {folds}-fold stratified CV on training data only\n'
        f'- Holdout accuracy: {test_metrics["accuracy"]:.4f}\n'
        f'- Holdout precision (spam): {test_metrics["precision"]:.4f}\n'
        f'- Holdout recall (spam): {test_metrics["recall"]:.4f}\n'
        f'- Holdout F1 (spam): {test_metrics["f1"]:.4f}\n'
        f'- False positives: {int(cm[0,1])}; false negatives: {int(cm[1,0])}\n\n'
        '- **Scope:** Dataset-based classification experiment; not connected to a live inbox.\n'
        '- **Limits:** Static corpus, possible dataset artifacts, no temporal/out-of-domain test, '
        'uncalibrated probabilities.\n'
        '- **Safety:** Never load `.joblib` files from untrusted sources.\n',
        encoding='utf-8')
    print(f'Selected: {name}; holdout F1={test_metrics["f1"]:.4f}; '
          f'precision={test_metrics["precision"]:.4f}; recall={test_metrics["recall"]:.4f}')
    print(f'Artifacts saved under {out.resolve()}')
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='emails.csv')
    parser.add_argument('--out',default='artifacts')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--test-size',type=float,default=.2)
    parser.add_argument('--max-features',type=int,default=5000)
    args=parser.parse_args()
    train_and_evaluate(args.data,args.out,seed=args.seed,folds=args.folds,
                       test_size=args.test_size,max_features=args.max_features)
if __name__=='__main__':
    main()
