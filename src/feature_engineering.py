import pandas as pd

from config import PROCESSED_TRAIN_PATH, FEATURES_TRAIN_PATH

def load_processed_data():
    df = pd.read_csv(PROCESSED_TRAIN_PATH, keep_default_na=False)
    return df

def create_features(df):
    df = df.copy()
    # Total house area
    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )
    # Total bathrooms
    df["TotalBathrooms"] = (
        df["FullBath"]
        + (0.5 * df["HalfBath"])
        + df["BsmtFullBath"]
        + (0.5 * df["BsmtHalfBath"])
    )
    # Age features
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    # Porch area
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["3SsnPorch"]
        + df["EnclosedPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )
    
    return df

def save_features_data(df):
    FEATURES_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_TRAIN_PATH, index=False)

def main():
    df = load_processed_data()
    df = create_features(df)
    save_features_data(df)
    print("Feature engineering completed successfully")

if __name__ == "__main__":
    main()