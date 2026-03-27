import sys
from pathlib import Path

# Add project root and src folder to python path
project_root = Path(__file__).resolve().parents[1]
src_path = project_root/"src"

sys.path.append(str(project_root))
sys.path.append(str(src_path))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.preprocess import basic_cleaning, handle_missing_values
from src.feature_engineering import create_features


# Cache model loading
@st.cache_resource
def load_model():
    return joblib.load("models/house_price_model.pkl")


# Cache template loading
@st.cache_data
def load_template():
    df = pd.read_csv("data/raw/train.csv")
    df = df.drop("SalePrice", axis=1)
    return df.iloc[[0]].copy()


model = load_model()
template = load_template()

st.title("🏠 House Price Predictor")

st.write("Enter house features in the sidebar to estimate the price.")


# Sidebar inputs
st.sidebar.header("House Features")

overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)

gr_liv_area = st.sidebar.number_input(
    "Living Area (sq ft)",
    min_value=300,
    max_value=6000,
    value=1500
)

garage_cars = st.sidebar.slider("Garage Capacity", 0, 4, 2)

total_bsmt_sf = st.sidebar.number_input(
    "Basement Area (sq ft)",
    min_value=0,
    max_value=3000,
    value=800
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1800,
    max_value=2024,
    value=2000
)


# Update template
template["OverallQual"] = overall_qual
template["GrLivArea"] = gr_liv_area
template["GarageCars"] = garage_cars
template["TotalBsmtSF"] = total_bsmt_sf
template["YearBuilt"] = year_built


if st.button("Predict Price"):

    df = template.copy()

    # Apply preprocessing
    df = basic_cleaning(df)
    df = handle_missing_values(df)

    # Apply feature engineering
    df = create_features(df)

    pred_log = model.predict(df)
    price = np.expm1(pred_log)

    st.metric("Predicted House Price", f"${price[0]:,.0f}")


st.markdown("---")

st.markdown("### About This Model")

st.write(
"""
This application predicts house prices using a Linear Regression model
trained on the Ames Housing Dataset.

Pipeline includes:
• Data preprocessing
• Feature engineering
• Outlier removal
• Cross-validation model comparison
• Final model training
"""
)