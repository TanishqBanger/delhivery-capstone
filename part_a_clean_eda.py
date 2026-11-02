from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shipment_prep import clean_shipments, build_features, FEATURES
from shipment_prep import (
    clean_shipments,
    build_features,
    FEATURES,
)

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

raw = pd.read_csv(DATA_DIR / "raw_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

raw_rows = len(raw)

# Keep a deduplicated copy for pre-cleaning checks
raw_dedup = raw.drop_duplicates().copy()

duplicate_rows_removed = raw_rows - len(raw_dedup)

# --------------------------------------------------
# 3. Calculate pre-cleaning checks
# --------------------------------------------------

missing_declared_value = (
    raw_dedup["declared_value"]
    .astype("string")
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

missing_declared_value = pd.to_numeric(
    missing_declared_value, errors="coerce"
).isna().sum()

checks = {
    "raw_rows": raw_rows,
    "duplicate_rows_removed": duplicate_rows_removed,
    "rows_after_dedup": len(raw_dedup),

    "missing_distance_km_before": raw_dedup["distance_km"].isna().sum(),

    "missing_parcel_weight_before": raw_dedup["parcel_weight_kg"].isna().sum(),

    "sentinel_weight_rows": (
        raw_dedup["parcel_weight_kg"] > 100
    ).sum(),

    "missing_declared_value_before": missing_declared_value,

    "missing_payment_mode_before": raw_dedup["payment_mode"].isna().sum(),

    "service_type_spellings_before": raw_dedup["service_type"].nunique(),
}

# --------------------------------------------------
# 4. Clean shipments
# --------------------------------------------------

clean, medians = clean_shipments(raw)

checks["service_type_spellings_after"] = clean["service_type"].nunique()

checks["median_distance_km"] = medians["distance_km"]
checks["median_parcel_weight_kg"] = medians["parcel_weight_kg"]
checks["median_declared_value_inr"] = medians["declared_value_inr"]

# --------------------------------------------------
# 5. Build features and merge hubs
# --------------------------------------------------

modelling = build_features(clean, hubs)

checks["rows_after_hub_merge"] = len(modelling)

# Create target variable
modelling["is_delayed"] = (
    modelling["delivery_hours"] > modelling["promised_hours"]
).astype(int)

# --------------------------------------------------
# 6. Feature and target checks
# --------------------------------------------------

checks["nulls_in_features_and_target"] = (
    modelling[FEATURES + ["delivery_hours", "is_delayed"]]
    .isna()
    .sum()
    .sum()
)

checks["delayed_count"] = modelling["is_delayed"].sum()
checks["delay_rate"] = modelling["is_delayed"].mean()

# --------------------------------------------------
# 7. Delivery-hours outlier checks using IQR
# --------------------------------------------------

q1 = modelling["delivery_hours"].quantile(0.25)
q3 = modelling["delivery_hours"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_count = (
    (modelling["delivery_hours"] < lower_bound)
    | (modelling["delivery_hours"] > upper_bound)
).sum()

checks["hours_q1"] = q1
checks["hours_q3"] = q3
checks["hours_lower_bound"] = lower_bound
checks["hours_upper_bound"] = upper_bound
checks["hours_outlier_count"] = outlier_count

# --------------------------------------------------
# 8. Declared-value statistics
# --------------------------------------------------

checks["declared_value_mean_inr"] = (
    modelling["declared_value_inr"].mean()
)

checks["declared_value_median_inr"] = (
    modelling["declared_value_inr"].median()
)

checks["log_declared_value_mean"] = (
    modelling["log_declared_value"].mean()
)

checks["log_declared_value_median"] = (
    modelling["log_declared_value"].median()
)

# --------------------------------------------------
# 9. Save modelling table
# --------------------------------------------------

modelling.to_csv(DATA_DIR / "modelling_table.csv", index=False)

# --------------------------------------------------
# 10. Delay analysis by group
# --------------------------------------------------

group_frames = []

# Service type
for dimension, column in [
    ("service_type", "service_type"),
    ("hub_tier", "hub_tier"),
]:
    grouped = (
        modelling.groupby(column, observed=True)
        .agg(
            shipments=("shipment_id", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean"),
        )
        .reset_index()
    )

    grouped = grouped.rename(columns={column: "group"})
    grouped.insert(0, "dimension", dimension)

    group_frames.append(grouped)

# Distance bands
modelling["distance_band"] = pd.cut(
    modelling["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

distance_grouped = (
    modelling.groupby("distance_band", observed=True)
    .agg(
        shipments=("shipment_id", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    )
    .reset_index()
)

distance_grouped = distance_grouped.rename(
    columns={"distance_band": "group"}
)

distance_grouped.insert(0, "dimension", "distance_band")

group_frames.append(distance_grouped)

delay_by_group = pd.concat(group_frames, ignore_index=True)

delay_by_group.to_csv(
    OUTPUT_DIR / "part_a_delay_by_group.csv",
    index=False,
)

# --------------------------------------------------
# 11. Correlation matrix
# --------------------------------------------------

numeric_data = modelling.select_dtypes(include="number")

correlations = numeric_data.corr()

correlations.to_csv(
    OUTPUT_DIR / "part_a_correlations.csv",
    index_label="feature",
)

# --------------------------------------------------
# 12. Save Part A checks
# --------------------------------------------------

checks_df = pd.DataFrame(
    list(checks.items()),
    columns=["metric", "value"],
)

checks_df.to_csv(
    OUTPUT_DIR / "part_a_checks.csv",
    index=False,
)

# --------------------------------------------------
# 13. Final message
# --------------------------------------------------

print("PART A CLEANING AND EDA COMPLETED")
print("---------------------------------")
print("Raw rows:", raw_rows)
print("Rows after cleaning:", len(clean))
print("Rows after hub merge:", len(modelling))
print("Delayed shipments:", checks["delayed_count"])
print("Delay rate:", round(checks["delay_rate"], 4))
print("Missing values in features and target:",
      checks["nulls_in_features_and_target"])

print("\nFiles generated:")
print("data/modelling_table.csv")
print("outputs/part_a_checks.csv")
print("outputs/part_a_delay_by_group.csv")
print("outputs/part_a_correlations.csv")