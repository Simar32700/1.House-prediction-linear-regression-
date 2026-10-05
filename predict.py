"""Load the saved model and predict a new house's price."""

from datetime import date
from pathlib import Path

import joblib
import pandas as pd


HOUSE_FIELDS = [
    ("OverallQual", "Overall quality (1 to 10)", 1, 10),
    ("GrLivArea", "Above-ground living area (dataset area units)", 1, None),
    ("GarageCars", "Garage capacity (number of cars)", 0, None),
    ("TotalBsmtSF", "Basement area (dataset area units; 0 for none)", 0, None),
    ("YearBuilt", "Year built", 1, date.today().year),
    ("FullBath", "Number of full bathrooms", 0, None),
    ("BedroomAbvGr", "Number of above-ground bedrooms", 0, None),
    ("LotArea", "Lot area (dataset area units)", 1, None),
]
MODEL_PATH = Path(__file__).resolve().parent / "models" / "house_price_model.joblib"


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Saved model not found. Run train_model.py first.")
    return joblib.load(MODEL_PATH)


def predict_price(house_details, model=None):
    """Return the predicted price in the dataset's price units."""
    validated_details = {}
    for name, label, minimum, maximum in HOUSE_FIELDS:
        if name not in house_details:
            raise ValueError(f"Missing house detail: {name}")
        try:
            value = float(house_details[name])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label} must be a whole number.") from error
        if not value.is_integer():
            raise ValueError(f"{label} must be a whole number.")
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"{label} is outside the allowed range.")
        validated_details[name] = int(value)

    if model is None:
        model = load_model()
    house = pd.DataFrame([validated_details])
    price = float(model.predict(house)[0])
    if price <= 0 or not pd.notna(price) or price == float("inf"):
        raise ValueError("The model returned an invalid price for these details.")
    return price
