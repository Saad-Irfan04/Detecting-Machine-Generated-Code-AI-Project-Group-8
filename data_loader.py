import os
import pandas as pd
from datasets import load_dataset

def main():
    ds = load_dataset("DaniilOr/SemEval-2026-Task13","A")

    train = ds["train"]
    valid = ds["validation"]
    test  = ds["test"]

    print("\n--- Dataset Loaded ---")
    print("Train size:", len(train))
    print("Validation size:", len(valid))
    print("Test size:", len(test))

    train_df = train.to_pandas()
    valid_df = valid.to_pandas()
    test_df  = test.to_pandas()

    data_dir = os.path.join("data")
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train_full.parquet")
    valid_path = os.path.join(data_dir, "valid_full.parquet")
    test_path  = os.path.join(data_dir, "test_full.parquet")

    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("\nFiles saved:")
    print(train_path)
    print(valid_path)
    print(test_path)

    print("\n--- Example Row ---")
    print(train_df.iloc[0])


if __name__ == "__main__":
    main()
