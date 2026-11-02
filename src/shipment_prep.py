import numpy as np
import pandas as pd


# Expected delivery time based on service type
PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}


# Features required for machine learning
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


# Columns considered operational noise
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


def clean_shipments(raw, medians=None):

    # Step 1: Remove exact duplicate rows
    df = raw.drop_duplicates().copy()

    # Step 2: Standardize service_type
    df["service_type"] = (
        df["service_type"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # Step 3: Handle missing payment_mode
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Step 4: Convert declared_value into numeric values
    df["declared_value_inr"] = (
        df["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value_inr"],
        errors="coerce"
    ).astype(float)

    # Remove original text column
    df = df.drop(columns=["declared_value"])

    # Step 5: Convert dispatch_time into datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # Step 6: Replace scanner-fault sentinel weights with missing values
    df.loc[
        df["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan

    # Step 7: Calculate or reuse median values
    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]

    if medians is None:

        medians = {
            col: float(df[col].median())
            for col in median_columns
        }

    else:

        medians = {
            col: float(medians[col])
            for col in median_columns
        }

    # Fill missing values using the calculated or supplied medians
    for col, value in medians.items():
        df[col] = df[col].fillna(value)

    return df, medians


def build_features(clean, hubs):

    # Step 1: Merge hub information into shipment data
    df = clean.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity"
            ]
        ],
        on="hub_id",
        how="left",
        validate="many_to_one"
    )

    # Step 2: Create promised_hours
    df["promised_hours"] = (
        df["service_type"]
        .map(PROMISED_HOURS)
        .astype(int)
    )

    # Step 3: Create is_cod
    df["is_cod"] = (
        df["payment_mode"] == "COD"
    ).astype(int)

    # Step 4: Identify weekends
    df["is_weekend"] = (
        df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # Step 5: Extract dispatch hour
    df["dispatch_hour"] = (
        df["dispatch_time"].dt.hour
    ).astype(int)

    # Step 6: Apply natural logarithm to declared value
    df["log_declared_value"] = np.log(
        df["declared_value_inr"]
    )

    return df