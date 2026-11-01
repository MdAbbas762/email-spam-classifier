"""Deterministic, corpus-free English text processing used at training AND inference.

The original notebook used NLTK data downloads and fit TF-IDF before the split.
The new pipeline puts this tokenizer inside cross-validated TF-IDF, preventing leakage.
"""
import re
from functools import lru_cache
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STEMMER = PorterStemmer()
_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)

@lru_cache(maxsize=40000)
def _stem(token: str) -> str:
    return _STEMMER.stem(token)

def tokenize(text: str) -> list[str]:
    """Lowercase, tokenize, remove common stopwords, and Porter-stem tokens.

    Does not require nltk.download or access to an external model service.
    The exact same callable is persisted within the fitted sklearn pipeline.
    """
    if not isinstance(text, str):
        raise TypeError("Expected text to be a string")
    return [_stem(w) for w in _TOKEN_RE.findall(text.lower())
            if len(w) > 1 and w not in _STOPWORDS]
