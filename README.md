# House Price Prediction (Linear Regression)

A simple end-to-end machine learning project that predicts house prices using scikit-learn's `LinearRegression`.

## Project Steps
1. Load and preprocess the housing data (impute missing values, scale numbers, one-hot encode city)
2. Split data into training and testing sets (80/20)
3. Train a `LinearRegression` model
4. Evaluate using **RMSE** and **R²**
5. Save the trained model to disk (`models/house_price_model.pkl`)

## Structure
```
house-price-prediction/
├── data/housing.csv        # dataset (synthetic, 1000 rows)
├── models/                 # saved model goes here
├── generate_data.py        # recreates the dataset
├── train.py                # steps 1-5
├── predict.py              # use the saved model
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python generate_data.py   # optional - data/housing.csv is already included
python train.py
python predict.py
```

## Using Your Own Data
Replace `data/housing.csv` with your CSV. Keep the target column named `price` (or edit `train.py`).

## Metrics
- **RMSE** - average prediction error in price units (lower is better)
- **R²** - how much variance the model explains (closer to 1 is better)
