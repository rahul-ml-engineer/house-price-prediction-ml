import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from config import FEATURES_TRAIN_PATH, MODEL_PATH

def load_data():
    df = pd.read_csv(FEATURES_TRAIN_PATH)
    return df

def split_data(df):
    y = np.log1p(df["SalePrice"])
    X = df.drop("SalePrice", axis=1)
    

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def encode_data(X_train, X_val):
    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)
    # align columns so both datasets match
    X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)
    print("Train shape after encoding:", X_train.shape)
    print("Validation shape after encoding:", X_val.shape)
    return X_train, X_val

def train_model(X_train, y_train, X_val, y_val):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions_log = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, predictions_log)
    print("Validation RMSE:", rmse)
    return model

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def main():
    df = load_data()
    X_train, X_val, y_train, y_val = split_data(df)
    X_train, X_val = encode_data(X_train, X_val)
    model = train_model(X_train, y_train, X_val, y_val)
    save_model(model)
    print("Model training completed successfully")
    print(df.shape)
    print("SalePrice min:", df["SalePrice"].min())
    print("SalePrice max:", df["SalePrice"].max())

if __name__ == "__main__":
    main()