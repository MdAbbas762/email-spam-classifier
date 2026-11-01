"""Comparable models that each fit their *own* TF-IDF in each CV fold."""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from .text import tokenize

MODEL_NAMES = ('multinomial_nb', 'bernoulli_nb', 'logistic_regression')

def build_pipeline(name: str, seed: int = 42, max_features: int = 5000) -> Pipeline:
    choices = {
        'multinomial_nb': lambda: MultinomialNB(alpha=1.0),
        'bernoulli_nb': lambda: BernoulliNB(alpha=1.0),
        'logistic_regression': lambda: LogisticRegression(
            max_iter=1200, class_weight='balanced', random_state=seed),
    }
    if name not in choices:
        raise ValueError(f'Unknown model {name}; valid choices: {MODEL_NAMES}')
    return Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                   lowercase=False, max_features=max_features,
                                   ngram_range=(1,2), sublinear_tf=True)),
        ('classifier', choices[name]()),
    ])
