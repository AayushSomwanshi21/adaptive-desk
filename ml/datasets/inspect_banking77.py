from datasets import load_dataset
from collections import Counter
from pathlib import Path
import pandas as pd


def main():

    print("Loading Banking77 Dataset")

    dataset_dir = Path(__file__).parent
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target_col = "category" if "category" in train.columns else "label"
    required_cols = {'text', target_col}
    if not required_cols.issubset(train.columns):
        raise ValueError(
            f"Expected columns {required_cols}, "
            f"but found {list(train.columns)}"
        )

    print("Dataset Spilts")
    print(f"Training Examples:{len(train)}")
    print(f"Test Examples:{len(test)}")

    print(f"\nColumns: {train.columns.to_list()}")
    print(f"\nNo of Categories : {train[target_col].nunique()}")

    print("\nTraining Examples per category:\n")
    counts = Counter(train[target_col])
    for label, count in counts.items():
        print(f"Label : {label} Count: {count}")

    print("\nSample training examples:\n")

    print(train.head())


if __name__ == "__main__":
    main()
