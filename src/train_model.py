import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from config import FEATURES_TRAIN_PATH, MODEL_PATH

def load_data():
    df = pd.read_csv(FEATURES_TRAIN_PATH)
    return df

def split_data(df):
    Y = np.log1p(df["SalePrice"])
    X = df.drop("SalePrice", axis=1)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_val, Y_train, Y_val

def build_pipeline(X_train):

    categorical_cols = X_train.select_dtypes(include=["object","string"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object","string"]).columns

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline

def train_model(pipeline, X_train, Y_train):
    pipeline.fit(X_train, Y_train)
    return pipeline

def evaluate_model(pipeline, X_val, Y_val):
    predictions_log = pipeline.predict(X_val)
    predictions = np.expm1(predictions_log)
    Y_val_real = np.expm1(Y_val)

    rmse = root_mean_squared_error(Y_val_real, predictions)
    print("Validation RMSE:", rmse)

def save_model(pipeline):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

def main():
    df = load_data()
    X_train, X_val, Y_train, Y_val = split_data(df)
    pipeline = build_pipeline(X_train)
    pipeline = train_model(pipeline, X_train, Y_train)
    evaluate_model(pipeline, X_val, Y_val)
    save_model(pipeline)
    print("Model training completed successfully")
    print(df.shape)
    
if __name__ == "__main__":
    main()