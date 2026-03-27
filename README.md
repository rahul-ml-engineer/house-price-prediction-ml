![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 🏠 House Price Prediction – End-to-End Machine Learning Project

This project builds a **complete machine learning pipeline** to predict house prices using the **Ames Housing Dataset**.

It demonstrates a full **end-to-end ML workflow**, including:

* Data exploration
* Data preprocessing
* Feature engineering
* Model comparison
* Final model training
* Prediction pipeline
* Deployment using a **Streamlit web application**

The goal is to estimate the **sale price of a house based on property features** such as living area, overall quality, basement size, garage capacity, and more.

---

## 🚀 Live Demo

Try the deployed application here:

https://house-price-prediction-ml-cgjpukm8yb78fpyrsgpz94.streamlit.app/

## 📷 App Preview

![App Screenshot](images/app_preview.png)

# 📊 Project Overview

House price prediction is a classic **regression problem in machine learning**.

This project follows real-world ML development practices:

* Exploratory Data Analysis (EDA)
* Data cleaning and preprocessing
* Feature engineering
* Model comparison using cross-validation
* Final model training
* Building a prediction pipeline
* Deploying the model with Streamlit

The final result is an **interactive application that predicts house prices instantly**.

---

# ⚙️ Machine Learning Pipeline

The project follows this workflow:

EDA
↓
Data Preprocessing
↓
Feature Engineering
↓
Model Comparison
↓
Final Model Training
↓
Prediction Pipeline
↓
Streamlit Web Application

---

# 🧠 Models Evaluated

The following models were evaluated using **5-fold cross-validation**:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Model performance was compared using **Root Mean Squared Error (RMSE)**.

**Final Model Used:** Gradient Boosting Regressor

---

## 📈 Model Performance

The models were evaluated using **5-fold cross-validation** and **Root Mean Squared Error (RMSE)**.

| Model              | Test RMSE | MAE    | R²   |
|--------------------|-----------|--------|------|
| Gradient Boosting  | 19,279    | 13,886 | 0.93 |
| Linear Regression  | 21,765    | 15,441 | 0.91 |
| Random Forest      | 23,658    | 16,462 | 0.89 |

**Final Model Selected:** Gradient Boosting Regressor

The Gradient Boosting model achieved the best performance with the lowest Test RMSE and MAE, and the highest R² score. Therefore, it was selected as the final model.

## 📊 Model Comparison

The models were compared using Test RMSE to determine the best performing algorithm.

![Model Comparison](reports/figures/model_comparison_rmse.png)

# 📈 Features Used

The model uses several important house characteristics such as:

* Overall quality of the house
* Living area (square feet)
* Basement size
* Garage capacity
* Year built

Additional **engineered features** were created to improve performance:

* HouseAge
* GarageAge
* RemodAge
* TotalSF
* TotalBathrooms
* TotalPorchSF

Feature engineering significantly improved the model’s predictive power.

---

## 🔎 Feature Importance

Feature importance analysis was performed using the final Gradient Boosting model to understand which features most influence house price predictions.

Key influential features include:

- OverallQual
- GrLivArea
- TotalSF
- GarageCars
- TotalBsmtSF

![Feature Importance](reports/figures/feature_importance.png)

# ⚡ Quick Start

Follow these steps to run the project locally.

## 1️⃣ Clone the repository

Repository:
https://github.com/rahul-ml-engineer/house-price-prediction-ml

Clone Command:
git clone https://github.com/rahul-ml-engineer/house-price-prediction-ml.git

cd house-price-prediction-ml

## 2️⃣ Create a virtual environment

python -m venv .venv

## 3️⃣ Activate the environment

Windows:

.venv\Scripts\activate

Mac / Linux:

source .venv/bin/activate

## 4️⃣ Install dependencies

pip install -r requirements.txt

## 5️⃣ Train the model

python src/train_model.py

## 6️⃣ Run the Streamlit application

streamlit run app/app.py

The application will open in your browser at:

http://localhost:8501

---

# 🖥 Streamlit Web Application

The Streamlit interface allows users to **input house features and get an estimated price instantly**.

Example inputs include:

* Overall Quality
* Living Area
* Basement Area
* Garage Capacity
* Year Built

The application then predicts the **estimated market value of the property**.

---

# 🗂 Project Structure

```
house-price-prediction-ml
│
├── app
│   └── app.py
│        Streamlit web application
│
├── data
│   └── raw
│       ├── train.csv
│       └── test.csv
│
├── models
│   └── house_price_model.pkl
│
├── notebooks
│   └── eda.ipynb
|
├── reports
│   └── figures
│       ├── model_comparison_rmse.png
│       └── feature_importance.png
│
├── src
│   ├── config.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── model_comparison.py
│   ├── train_model.py
│   ├── feature_importance.py
│   └── predict.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 📚 Dataset

Dataset used:

**Ames Housing Dataset**

This dataset contains detailed information about residential homes in **Ames, Iowa**, and is widely used for machine learning regression problems.

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib
* Matplotlib / Seaborn

---

# 📌 Key Learnings

This project demonstrates:

* Building an end-to-end machine learning pipeline
* Handling missing values and outliers
* Creating engineered features for better predictions
* Comparing models using cross-validation
* Deploying machine learning models with Streamlit

---

# 👤 Author

Machine learning project built for **portfolio and freelancing demonstration**.

---

# ⭐ Future Improvements

Possible enhancements include:

* Hyperparameter tuning
* Adding advanced models (XGBoost / LightGBM)
* Cloud deployment (Streamlit Cloud or AWS)
* Expanding the web interface with more features
