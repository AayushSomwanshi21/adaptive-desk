from datasets import load_dataset
from collections import Counter
from pathlib import Path
import pandas as pd


def load_dataset(
        file_path: str,
        text_column: str = 'text',
        label_column: str = 'category') -> pd.DataFrame:
    """Load a labelled text-classification dataset from a CSV file"""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required_cols = [text_column, label_column]
    missing_columns = [
        column for column in required_cols
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[text_column, label_column]].copy()
    df.columns = ['text', 'label']

    return df
