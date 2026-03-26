import pandas as pd
import numpy as np

from config import TRAIN_DATA_PATH, PROCESSED_TRAIN_PATH

def load_data():
    df = pd.read_csv(TRAIN_DATA_PATH)
    return df

def basic_cleaning(df):
    df = df.copy()
    # drop Id column
    if "Id" in df.columns:
        df.drop("Id", axis=1, inplace=True)
    return df

def handle_missing_values(df):
    # Categorical columns where NaN means "feature not present"
    cols_fill_none = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2",
        "MasVnrType"
    ]
    for col in cols_fill_none:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Numerical columns where NaN means feature not present
    zero_fill_cols = [
        "GarageYrBlt", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
        "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
        "GarageCars", "GarageArea"
    ]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage special case
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Safety check: replace any remaining categorical NaN with "None"
    cat_cols = df.select_dtypes(include=["object","string"]).columns
    df[cat_cols] = df[cat_cols].fillna("None")

    return df

def save_processed_data(df):
    # ensure processed folder exists
    PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_TRAIN_PATH, index=False)

def main():
    df = load_data()
    df = basic_cleaning(df)
    df = handle_missing_values(df)
    save_processed_data(df)
    print("Preprocessing completed successfully")

if __name__ == "__main__":
    main()