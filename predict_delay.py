
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from src.shipment_prep import clean_shipments, build_features


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


# --------------------------------------------------
# 2. Load new shipments and supporting data
# --------------------------------------------------

new_shipments = pd.read_csv(DATA_DIR / "new_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")
hub_segments = pd.read_csv(OUTPUT_DIR / "hub_segments.csv")


# --------------------------------------------------
# 3. Load saved model and configuration
# --------------------------------------------------

model = joblib.load(MODEL_DIR / "delay_model.pkl")

threshold_df = pd.read_csv(MODEL_DIR / "delay_threshold.csv")
threshold = float(threshold_df.loc[0, "threshold"])

feature_order_df = pd.read_csv(MODEL_DIR / "feature_order.csv")
feature_order = feature_order_df["feature"].tolist()

impute_df = pd.read_csv(MODEL_DIR / "impute_values.csv")
impute_values = dict(
    zip(impute_df["column"], impute_df["median"])
)


# --------------------------------------------------
# 4. Clean new shipments using training medians
# --------------------------------------------------

cleaned_shipments, _ = clean_shipments(
    new_shipments,
    medians=impute_values
)


# --------------------------------------------------
# 5. Generate features using shared function
# --------------------------------------------------

featured_shipments = build_features(
    cleaned_shipments,
    hubs
)

X_new = featured_shipments[feature_order]


# --------------------------------------------------
# 6. Predict delay probabilities
# --------------------------------------------------

probabilities = model.predict_proba(X_new)[:, 1]

flagged = (probabilities >= threshold).astype(int)


# --------------------------------------------------
# 7. Join hub segment information
# --------------------------------------------------

results = featured_shipments[
    ["shipment_id", "hub_id"]
].copy()

results["delay_probability"] = np.round(probabilities, 4)
results["flagged"] = flagged

results = results.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    validate="many_to_one",
    sort=False
)


# --------------------------------------------------
# 8. Assign business actions
# --------------------------------------------------

results["action"] = np.select(
    [
        (results["flagged"] == 1) &
        (results["segment"] == "Escalate"),

        results["flagged"] == 1
    ],
    [
        "Escalate to hub manager",
        "Proactive customer call"
    ],
    default="No action"
)


# --------------------------------------------------
# 9. Arrange required output columns
# --------------------------------------------------

results = results[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
]


# --------------------------------------------------
# 10. Save prediction results
# --------------------------------------------------

results.to_csv(
    OUTPUT_DIR / "new_shipment_actions.csv",
    index=False
)


# --------------------------------------------------
# 11. Display summary
# --------------------------------------------------

print("\nNew Shipment Predictions:")
print(results.to_string(index=False))

print("\nPrediction Summary:")
print("Total shipments:", len(results))
print("Flagged shipments:", int(results["flagged"].sum()))

print("\nAction Counts:")
print(results["action"].value_counts().to_string())

print("\nMissing values:")
print(results.isna().sum().to_string())

print("\nSaved: outputs/new_shipment_actions.csv")
