import pandas as pd
from typing import Tuple

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a parquet dataset file.
    """
    return pd.read_parquet(path)


def load_train_valid(
    train_path: str = "data/train_full.parquet",
    valid_path: str = "data/valid_full.parquet"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation parquet files.
    """
    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    return train_df, valid_df
