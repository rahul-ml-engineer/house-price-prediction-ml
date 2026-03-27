import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from config import FEATURES_TRAIN_PATH

def load_data():
    df = pd.read_csv(FEATURES_TRAIN_PATH, keep_default_na=False)
    return df

def split_data(df):
    X = df.drop("SalePrice", axis=1)
    Y = np.log1p(df["SalePrice"])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def build_preprocessor(X_train):
    categorical_cols = X_train.select_dtypes(include=["object","string"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object","string"]).columns

    preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )
    return preprocessor

def compare_models(X_train, Y_train, preprocessor):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }
    results = []
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        scores = cross_val_score(pipeline, X_train, Y_train, cv=5, scoring="neg_mean_squared_error",n_jobs=-1)
        rmse_scores = np.sqrt(-scores)
        results.append({"Model": name, "Mean RMSE": round(rmse_scores.mean(), 2), "Std RMSE": round(rmse_scores.std(), 2)})
    results_df = pd.DataFrame(results).sort_values("Mean RMSE")
    
    print("\nModel Comparison Results:\n")
    print(results_df)

def main():
    df = load_data()
    X_train, X_test, Y_train, Y_test = split_data(df)
    preprocessor = build_preprocessor(X_train)
    compare_models(X_train, Y_train, preprocessor)

if __name__ == "__main__":
    main()