"""House Price Prediction - end-to-end with LinearRegression."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load and preprocess the data
df = pd.read_csv("data/housing.csv")
X = df.drop(columns="price")
y = df["price"]

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train a LinearRegression model
model = Pipeline([("preprocess", preprocess), ("regressor", LinearRegression())])
model.fit(X_train, y_train)

# 4. Evaluate using RMSE and R^2
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f"RMSE: {rmse:,.2f}")
print(f"R2  : {r2:.4f}")

# 5. Save the trained model to disk
joblib.dump(model, "models/house_price_model.pkl")
print("Model saved to models/house_price_model.pkl")
