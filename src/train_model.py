import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from config import FEATURES_TRAIN_PATH, MODEL_PATH

def load_data():
    df = pd.read_csv(FEATURES_TRAIN_PATH)
    return df

def build_pipeline(X_train):
    categorical_cols = X_train.select_dtypes(include=["object","string"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object","string"]).columns

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )

    model = GradientBoostingRegressor(random_state=42, n_estimators=300, learning_rate=0.05)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline

def train_model(pipeline, X, Y):
    pipeline.fit(X, Y)
    return pipeline

def save_model(pipeline):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

def main():
    df = load_data()
    X = df.drop("SalePrice", axis=1)
    Y = np.log1p(df["SalePrice"])
    pipeline = build_pipeline(X)
    pipeline = train_model(pipeline, X, Y)
    save_model(pipeline)
    print("Final model training completed successfully")

if __name__ == "__main__":
    main()