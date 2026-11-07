"""Load the saved model and predict one house price."""
import joblib
import pandas as pd

model = joblib.load("models/house_price_model.pkl")
house = pd.DataFrame([{
    "area_sqft": 1500, "bedrooms": 3, "bathrooms": 2,
    "age_years": 5, "distance_to_city_km": 8.0, "city": "Pune",
}])
print(f"Predicted price: Rs {model.predict(house)[0]:,.0f}")
