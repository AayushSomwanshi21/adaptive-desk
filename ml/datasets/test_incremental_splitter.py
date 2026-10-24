from loader import load_dataset
from validator import validate_dataset
from incremental_splitter import create_incremental_splits

raw_test_df = load_dataset("ml/datasets/test.csv")
print("After loading:", len(raw_test_df))

test_df = validate_dataset(raw_test_df)
print("After validation:", len(test_df))

print("Rows with missing text:", raw_test_df["text"].isna().sum())
print("Rows with missing labels:", raw_test_df["label"].isna().sum())
print(
    "Duplicate pairs before validation:",
    raw_test_df.duplicated(subset=["text", "label"]).sum()
)

train_df = validate_dataset(
    load_dataset("ml/datasets/train.csv"),
    remove_duplicates=True
)

stages = create_incremental_splits(
    train_df,
    test_df,
    classes_per_stage=10,
    seed=42,
)

print("Final stage test rows:", len(stages[-1]["test"]))
