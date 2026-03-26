import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from config import FEATURES_TRAIN_PATH, MODEL_PATH

def load_data():
    df = pd.read_csv(FEATURES_TRAIN_PATH, keep_default_na=False)
    return df

def prepare_data(df):
    df = df.copy()
    # target
    y = df["SalePrice"]
    # features
    X = df.drop("SalePrice", axis=1)
    # encode categorical variables
    X = pd.get_dummies(X)
    return X, y

def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, predictions)
    print("Validation RMSE:", rmse)
    return model

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def main():
    df = load_data()
    X, y = prepare_data(df)
    model = train_model(X, y)
    save_model(model)
    print("Model training completed successfully")

if __name__ == "__main__":
    main()