import pandas as pd


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a standardized text-classification dataset"""

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns"
        )

    df = df.copy()

    # Remove rows with missing text or labels
    df = df.dropna(subset=["text", "label"])

    # Convert values to strings and remove surrounding whitespace
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    # Remove empty text or labels
    df = df[(df["text"] != "") & (df["label"] != "")]

    # Remove duplicate text-label pairs
    df = df.drop_duplicates(subset=["text", "label"])

    if df.empty:
        raise ValueError("No valid examples remain after cleaning")

    if df["label"].nunique() < 2:
        raise ValueError(
            "At least two distinct categories are required"
        )

    return df.reset_index(drop=True)
