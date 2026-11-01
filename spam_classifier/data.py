"""Load and validate the original two-column Kaggle dataset."""
from pathlib import Path
import pandas as pd

def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    missing = {'text', 'spam'} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    raw_rows = len(df)
    df = df[['text', 'spam']].copy()
    df['text'] = df['text'].astype('string').str.strip()
    missing_rows = df['text'].isna() | df['text'].eq('') | df['spam'].isna()
    removed_missing = int(missing_rows.sum())
    df = df.loc[~missing_rows].copy()
    try:
        df['spam'] = pd.to_numeric(df['spam'], errors='raise')
    except Exception as exc:
        raise ValueError('Labels must be numeric 0 (ham) or 1 (spam)') from exc
    if not df['spam'].isin([0, 1]).all():
        raise ValueError('Labels must be either 0 (ham) or 1 (spam)')
    if (df.groupby('text')['spam'].nunique() > 1).any():
        raise ValueError('Conflicting labels for identical email text')
    # Drop duplicate content before splitting, so exact copies cannot enter both splits.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    if df['spam'].nunique() != 2:
        raise ValueError('Both ham and spam classes must be present')
    return df, {'source_rows':raw_rows, 'removed_missing':removed_missing,
                'removed_duplicate_texts':before_dedup-len(df),
                'remaining_rows':len(df),
                'ham':int((df['spam']==0).sum()),'spam':int((df['spam']==1).sum())}
