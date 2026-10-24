from loader import load_dataset
from validator import validate_dataset

TRAIN_PATH = "ml/datasets/train.csv"
TEST_PATH = "ml/datasets/test.csv"


def main():
    # Load the original datasets
    train_raw = load_dataset(
        TRAIN_PATH,
        text_column="text",
        label_column="category",
    )

    test_raw = load_dataset(
        TEST_PATH,
        text_column="text",
        label_column="category",
    )

    print(f"Training rows: {len(train_raw)}")
    print(f"Test rows: {len(test_raw)}")
    print(f"Training columns: {train_raw.columns.tolist()}")
    print(f"Test columns: {test_raw.columns.tolist()}")

    # Validate training data and remove exact duplicate pairs
    train_df = validate_dataset(
        train_raw,
        remove_duplicates=True,
    )

    # Preserve all valid test examples, including repeated pairs
    test_df = validate_dataset(
        test_raw,
        remove_duplicates=False,
    )

    print("\nValidated Dataset Summary:\n")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Training categories: {train_df['label'].nunique()}")
    print(f"Test categories: {test_df['label'].nunique()}")

    print("\nData Quality Checks")
    print(
        "Missing training text:",
        train_df["text"].isna().sum(),
    )
    print(
        "Missing training labels:",
        train_df["label"].isna().sum(),
    )
    print(
        "Missing test text:",
        test_df["text"].isna().sum(),
    )
    print(
        "Missing test labels:",
        test_df["label"].isna().sum(),
    )
    print(
        "Duplicate training pairs:",
        train_df.duplicated(subset=["text", "label"]).sum(),
    )
    print(
        "Duplicate test pairs:",
        test_df.duplicated(subset=["text", "label"]).sum(),
    )

    print("\nSample Training Examples")
    print(train_df.head().to_string(index=False))

    print("\nTraining Category Counts")
    print(train_df["label"].value_counts().sort_index().to_string())

    # Basic checks for the expected BANKING77 dataset
    assert train_df["label"].nunique() == 77, (
        "Expected 77 training categories."
    )
    assert test_df["label"].nunique() == 77, (
        "Expected 77 test categories."
    )
    assert len(test_df) == len(test_raw), (
        "Validation unexpectedly changed the test row count."
    )

    print("\nAll dataset checks passed.")


if __name__ == "__main__":
    main()
