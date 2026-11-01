"""Inspect misclassifications locally; never exports raw emails to reports.

python -m spam_classifier.inspect_errors --kind false_negative --limit 3
"""
import argparse
from pathlib import Path
import pandas as pd
from .data import load_dataset


def inspect(data_path: str | Path, errors_path: str | Path, kind: str='all',
            limit: int=5, chars: int=180) -> list[dict]:
    if limit < 1 or chars < 1:
        raise ValueError('limit and chars must be positive')
    dataset, _ = load_dataset(data_path)
    errors = pd.read_csv(errors_path)
    if kind != 'all':
        errors = errors.loc[errors['error_type'] == kind]
    records=[]
    for r in errors.head(limit).to_dict('records'):
        index = int(r['cleaned_row_index'])
        if index not in dataset.index:
            raise ValueError(f'Invalid cleaned row index: {index}')
        records.append({**r, 'local_text_preview': str(dataset.loc[index,'text'])[:chars]})
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='emails.csv')
    ap.add_argument('--errors',default='artifacts/reports/holdout_errors.csv')
    ap.add_argument('--kind',choices=['all','false_positive','false_negative'],default='all')
    ap.add_argument('--limit',type=int,default=5)
    ap.add_argument('--chars',type=int,default=180)
    args=ap.parse_args()
    for r in inspect(args.data,args.errors,args.kind,args.limit,args.chars):
        print(f"\n{r['error_type']} | cleaned row {r['cleaned_row_index']} | "
              f"spam probability {float(r['predicted_spam_probability']):.3f}\n"
              f"{r['local_text_preview']}")


if __name__ == '__main__':
    main()
