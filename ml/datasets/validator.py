import pandas as pd

# Remove exact duplicate text-label pairs only when requested.
# By default preserve all rows.


def validate_dataset(
    df: pd.DataFrame,
    remove_duplicates: bool = False,
) -> pd.DataFrame:
    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns."
        )

    df = df.copy()
    df = df.dropna(subset=["text", "label"])

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[(df["text"] != "") & (df["label"] != "")]

    if remove_duplicates:
        df = df.drop_duplicates(subset=["text", "label"])

    if df.empty:
        raise ValueError("No valid examples remain after cleaning.")

    if df["label"].nunique() < 2:
        raise ValueError("At least two distinct categories are required.")

    return df.reset_index(drop=True)
