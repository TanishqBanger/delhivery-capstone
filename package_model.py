
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import FEATURES, clean_shipments


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Read final Part C configuration
# --------------------------------------------------

config = pd.read_csv(OUTPUT_DIR / "part_c_final_config.csv")
config = config.iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

max_depth_value = str(config["max_depth"]).strip().lower()

max_depth = (
    None if max_depth_value == "unlimited"
    else int(max_depth_value)
)


# --------------------------------------------------
# 3. Rebuild the same Part C dataset and split
# --------------------------------------------------

df = pd.read_csv(DATA_DIR / "modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# --------------------------------------------------
# 4. Build and train final classifier
# --------------------------------------------------

final_model = Pipeline([
    ("selector", SelectKBest(
        score_func=f_classif,
        k=selector_k
    )),
    ("classifier", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    ))
])

final_model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Save model and supporting files
# --------------------------------------------------

joblib.dump(
    final_model,
    MODEL_DIR / "delay_model.pkl"
)

pd.DataFrame({
    "threshold": [threshold]
}).to_csv(
    MODEL_DIR / "delay_threshold.csv",
    index=False
)

pd.DataFrame({
    "feature": FEATURES
}).to_csv(
    MODEL_DIR / "feature_order.csv",
    index=False
)


# --------------------------------------------------
# 6. Calculate and save training imputation medians
# --------------------------------------------------

raw_shipments = pd.read_csv(DATA_DIR / "raw_shipments.csv")

cleaned_shipments, medians = clean_shipments(raw_shipments)

impute_values = pd.DataFrame([
    {"column": column, "median": value}
    for column, value in medians.items()
])

impute_values.to_csv(
    MODEL_DIR / "impute_values.csv",
    index=False
)


# --------------------------------------------------
# 7. Reload model and verify predictions
# --------------------------------------------------

reloaded_model = joblib.load(
    MODEL_DIR / "delay_model.pkl"
)

original_probabilities = final_model.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

max_probability_diff = np.max(
    np.abs(original_probabilities - reloaded_probabilities)
)

reload_check = pd.DataFrame([
    {"metric": "test_rows_scored", "value": len(X_test)},
    {
        "metric": "reload_max_abs_probability_diff",
        "value": max_probability_diff
    },
    {"metric": "threshold", "value": threshold}
])

reload_check.to_csv(
    OUTPUT_DIR / "part_d_reload_check.csv",
    index=False
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\nModel packaging completed!")

print("\nConfiguration:")
print(config)

print("\nSaved model files:")
print("delay_model.pkl")
print("delay_threshold.csv")
print("feature_order.csv")
print("impute_values.csv")

print("\nReload verification:")
print(reload_check.to_string(index=False))
