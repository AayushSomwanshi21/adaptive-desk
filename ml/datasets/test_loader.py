from loader import load_dataset
from validator import validate_dataset

DATA_DIR = "ml/datasets"

train_df = load_dataset(
    f"{DATA_DIR}/train.csv",
    text_column="text",
    label_column="category"
)

clean_df = validate_dataset(train_df)

print("Original rows:", len(train_df))
print("Valid rows:", len(clean_df))
print("Categories:", clean_df["label"].nunique())
print("\nSample rows:")
print(clean_df.head())
