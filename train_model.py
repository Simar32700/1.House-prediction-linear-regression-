"""Prepare house price data for model training."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

data_path = Path(__file__).resolve().parent / "data" / "house_prices_practice.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Input columns:", X.columns.tolist())
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
numeric_columns = X.select_dtypes(include="number").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numbers", "passthrough", numeric_columns),
    ]
)

# Learn categories only from training data, then apply them to both sets.
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("\nEncoded input columns:", preprocessor.get_feature_names_out().tolist())
print("Encoded training shape:", X_train_encoded.shape)
print("Encoded test shape:", X_test_encoded.shape)

model = LinearRegression()
model.fit(X_train_encoded, y_train)
predictions = model.predict(X_test_encoded)

model_directory = data_path.parent.parent / "models"
model_directory.mkdir(exist_ok=True)
model_path = model_directory / "house_price_model.joblib"
prediction_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
joblib.dump(prediction_pipeline, model_path)
print(f"\nModel and preprocessing saved to: {model_path}")

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train_encoded, y_train)
baseline_predictions = baseline.predict(X_test_encoded)

print("\nTest results:")
print(f"Model mean absolute error: {mean_absolute_error(y_test, predictions):,.2f}")
print(f"Average-price baseline error: {mean_absolute_error(y_test, baseline_predictions):,.2f}")
print(f"Model R-squared: {r2_score(y_test, predictions):.4f}")

# Custom tolerance score, not standard classification accuracy.
tolerance = 0.10
valid_prices = y_test > 0
if valid_prices.any():
    relative_errors = (
        (y_test[valid_prices] - predictions[valid_prices]).abs() / y_test[valid_prices]
    )
    within_tolerance = relative_errors <= tolerance
    tolerance_score = within_tolerance.mean() * 100
    print(f"Accuracy within {tolerance:.0%} tolerance: {tolerance_score:.2f}%")
    print(f"Houses within tolerance: {within_tolerance.sum()} / {valid_prices.sum()}")
else:
    print("Tolerance score unavailable: no positive test prices.")
if (~valid_prices).any():
    print(f"Excluded nonpositive prices from tolerance score: {(~valid_prices).sum()}")

print("\nFirst five test predictions:")
comparison = pd.DataFrame({"Actual price": y_test, "Predicted price": predictions})
print(comparison.head().round(2).to_string(index=False))

# These charts evaluate a separate binary task, not price regression accuracy.
price_cutoff = y_train.median()
actual_classes = (y_test >= price_cutoff).astype(int)
predicted_classes = (predictions >= price_cutoff).astype(int)
chart_directory = data_path.parent.parent / "results"
chart_directory.mkdir(exist_ok=True)

print("\nHigh-price vs. low-price classification (separate diagnostic):")
print(f"High-price cutoff from training median: {price_cutoff:,.2f}")
print(f"Class accuracy: {accuracy_score(actual_classes, predicted_classes):.2%}")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    actual_classes,
    predicted_classes,
    labels=[0, 1],
    display_labels=["Low price", "High price"],
    cmap="Blues",
    colorbar=False,
    ax=ax,
)
ax.set_title(f"Price category confusion matrix\nCutoff: {price_cutoff:,.2f}")
fig.tight_layout()
matrix_path = chart_directory / "confusion_matrix.png"
fig.savefig(matrix_path, dpi=150)
plt.close(fig)
print(f"Confusion matrix saved to: {matrix_path}")

if actual_classes.nunique() == 2:
    # Predicted prices are continuous ranking scores, not probabilities.
    auc = roc_auc_score(actual_classes, predictions)
    print(f"Price category ROC AUC: {auc:.4f}")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        actual_classes, predictions, name="Predicted price score", ax=ax
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random ranking")
    ax.set_title("ROC curve: high-price vs. low-price")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = chart_directory / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved to: {roc_path}")
else:
    print("ROC AUC unavailable: the test set needs both price categories.")
