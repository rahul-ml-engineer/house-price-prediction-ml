import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

from config import FEATURES_TRAIN_PATH, MODEL_COMPARISON_RESULTS_PATH

def load_data():
    df = pd.read_csv(FEATURES_TRAIN_PATH)
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
def cross_validate_model(pipeline, X, Y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        Y_train_fold = Y.iloc[train_idx]
        Y_val_fold = Y.iloc[val_idx]

        pipeline.fit(X_train_fold, Y_train_fold)
        preds_log = pipeline.predict(X_val_fold)
        preds = np.expm1(preds_log)
        Y_true = np.expm1(Y_val_fold)
        rmse = root_mean_squared_error(Y_true, preds)
        rmse_scores.append(rmse)
    return np.mean(rmse_scores), np.std(rmse_scores)


def evaluate_test_set(pipeline, X_train, Y_train, X_test, Y_test):
    pipeline.fit(X_train, Y_train)
    preds_log = pipeline.predict(X_test)
    preds = np.expm1(preds_log)
    y_true = np.expm1(Y_test)
    rmse = root_mean_squared_error(y_true, preds)
    return rmse

def compare_models(X_train, X_test, Y_train, Y_test):

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42,n_estimators=300, learning_rate=0.05)
    }
    results = []

    for name, model in models.items():
        # fresh preprocessor per model
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        cv_mean, cv_std = cross_validate_model(pipeline, X_train, Y_train)
        test_rmse = evaluate_test_set(pipeline, X_train, Y_train, X_test, Y_test)

        results.append({"Model": name, "CV Mean RMSE": round(cv_mean, 2), "CV Std RMSE": round(cv_std, 2), "Test RMSE": round(test_rmse, 2)})

    results_df = pd.DataFrame(results).sort_values("CV Mean RMSE")
    results_df["Best"] = ""
    results_df.iloc[0, results_df.columns.get_loc("Best")] = "✓"

    print("\nModel Comparison Results:\n")
    print(results_df)

    MODEL_COMPARISON_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(MODEL_COMPARISON_RESULTS_PATH, index=False)

def main():
    df = load_data()
    X_train, X_test, Y_train, Y_test = split_data(df)
    compare_models(X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":
    main()