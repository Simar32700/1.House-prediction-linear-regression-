# House Price Prediction

How much could a house cost? Enter its details, and this Python project gives
you an estimated price using a model called **Linear Regression**.

## What Each File Does

| File | Its job |
| --- | --- |
| `clean_data.py` | Checks the data for missing values, duplicates, and unusual values. |
| `train_model.py` | Teaches the model, tests it, and saves the model and charts. |
| `predict.py` | Uses the saved model to estimate a house price. |
| `main.py` | Asks you for house details and shows the price. |
| `requirements.txt` | Lists the Python libraries the project needs. |

## Run It

Tested with Python 3.11. Download or clone this repository, open its folder in
VS Code, and run these commands in the terminal:

```powershell
python -m pip install -r requirements.txt
python clean_data.py
python train_model.py
python main.py
```

Enter the quality, living area, garage capacity, basement area, year built,
bathrooms, bedrooms, and lot area when asked. Use whole numbers and the same
area units as the dataset. Quality must be between 1 and 10.

Once the model is saved, you only need `python main.py` to make more predictions.
Enter `y` for another house or `n` to finish.

## Data and Results

We use `data/house_prices_practice.csv`, which has **300 houses**. Our cleaning
checks found no issues that needed changes.

The model learns from **240 houses (80%)** and is tested on **60 houses (20%)**.
`Id` is excluded, and `SalePrice` is the price it learns to predict.

- **90% within 10% of the actual price:** 54 of 60 test predictions passed.
  For a house priced at 100,000, a prediction from 90,000 to 110,000 passes.
- **Average price error (MAE): 10,431.66.** Lower is better.
- **R-squared: 0.9724.** Closer to 1 is better; this is not an accuracy percentage.

These results describe this practice dataset, not guaranteed real-world prices.
Its source, currency, and area units have not been verified.

## Where Are the Pictures?

Open the `results` folder in VS Code and click:

- `confusion_matrix.png` to see correct and incorrect low/high-price labels.
- `roc_curve.png` to see how well predicted prices separate those two groups.

These charts use **251,578** as the cutoff: prices at or above it are high-price.
This separate category check scored **96.67%**, with **ROC AUC of 0.9989**.
It is different from the 90% price-tolerance result above.

The saved model is in `models/house_price_model.joblib`. If it is missing, or you
change the dataset, run the checks and training again before predicting.
