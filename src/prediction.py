import pandas as pd
import numpy as np
import joblib

from config import TEST_DATA_PATH, MODEL_PATH, PREDICTIONS_PATH

from preprocess import basic_cleaning, handle_missing_values
from feature_engineering import create_features


def load_test_data():
    df = pd.read_csv(TEST_DATA_PATH)
    return df

def preprocess_data(df):
    df = basic_cleaning(df)
    df = handle_missing_values(df)
    df = create_features(df)
    return df

def make_predictions(model, df, ids):
    preds_log = model.predict(df)
    preds = np.expm1(preds_log)
    predictions = pd.DataFrame({"Id": ids, "SalePrice": preds})
    return predictions

def save_predictions(predictions):
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

def main():
    model = joblib.load(MODEL_PATH)
    test_df = load_test_data()
    ids = test_df["Id"]
    test_df = preprocess_data(test_df)
    predictions = make_predictions(model, test_df, ids)
    save_predictions(predictions)
    print("Predictions saved successfully")

if __name__ == "__main__":
    main()