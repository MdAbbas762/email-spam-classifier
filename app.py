"""Email spam classifier CLI — the original one-message interaction still works.

python app.py                     # interactive, as in the original project
python app.py --text "Subject: Free prize"  # one-off prediction
python app.py --text "hello" --json        # structured result
"""
import argparse
import json
from pathlib import Path
import joblib

def predict(text: str, model_path='artifacts/spam_pipeline.joblib') -> dict:
    path=Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f'New trained model not found at {path}. '
            'Run: python -m spam_classifier.train --data emails.csv --out artifacts')
    # joblib/pickle deserialization is only safe for trusted models.
    pipeline=joblib.load(path)
    pred=int(pipeline.predict([text])[0])
    prob=float(pipeline.predict_proba([text])[0,1])
    return {'label':'Spam' if pred==1 else 'Not Spam',
            'predicted_spam_probability':round(prob,4)}

def main():
    parser=argparse.ArgumentParser(description='Classify an email as Spam or Not Spam')
    parser.add_argument('--text',help='Email content to classify; prompts if omitted')
    parser.add_argument('--model',default='artifacts/spam_pipeline.joblib')
    parser.add_argument('--json',action='store_true',help='Print machine-readable JSON')
    args=parser.parse_args()
    message=args.text if args.text is not None else input('Enter the email: ')
    if not message.strip():
        parser.error('Email text must not be empty')
    try:
        result=predict(message,args.model)
    except (FileNotFoundError,ValueError) as exc:
        parser.exit(2,f'Error: {exc}\n')
    if args.json:
        print(json.dumps(result))
    else:
        print(result['label'])
        print(f'Estimated spam probability: {result["predicted_spam_probability"]:.1%} '
              '(uncalibrated)')

if __name__=='__main__':
    main()
