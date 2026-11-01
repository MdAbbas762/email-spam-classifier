# Model Card — Email Spam Classifier

**Model:** Class-balanced Logistic Regression selected using three-fold stratified cross-validation on training data only.  
**Features:** TF-IDF unigrams and bigrams with deterministic text preprocessing.  
**Held-out evaluation:** 1,139 distinct messages from the supplied labeled dataset (seed 42).

| Held-out metric | Verified local run |
|---|---:|
| Accuracy | 99.03% |
| Precision (spam) | 96.47% |
| Recall (spam) | 99.64% |
| F1 (spam) | 98.03% |
| Average precision | 0.99826 |
| ROC-AUC | 0.99946 |

**Confusion matrix:** 855 true negatives; 10 false positives; 1 false negative; 273 true positives.

## Scope and responsible usage

- Text-only classifier evaluated on a static labeled corpus, not connected to a live email inbox.
- No phishing detection or automated email blocking functionality.
- Uncalibrated model scores are not verified probabilities.
- No temporal or independent external dataset evaluation; results may not generalize to other populations.
- Only load locally generated or otherwise trusted `.joblib` model files.

These values represent the user's November 1, 2026 local Python 3.11 run. Re-training on different software versions can change values; use the new `metrics.json` and regenerate this card if results differ.
