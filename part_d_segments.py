
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load modelling dataset
df = pd.read_csv(DATA_DIR / "modelling_table.csv")

print("Dataset shape:", df.shape)

# Task 1: Create hub-level profile

hub_profile = (
    df.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    )
    .reset_index()
    .sort_values("hub_id")
    .reset_index(drop=True)
)

# Save hub profile
hub_profile.to_csv(
    OUTPUT_DIR / "hub_profile.csv",
    index=False
)

print("\nHub Profile:")
print(hub_profile.to_string(index=False))

print("\nHub Profile Shape:", hub_profile.shape)
print("\nShipments per hub:")
print(hub_profile["shipments"].value_counts())


# K-Means segmentation

# Task 2: K-Means segmentation

# Select the four hub profile features
cluster_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

# Scale the features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    hub_profile[cluster_features]
)

# Fit K-Means with exactly 3 clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means Cluster Results:")
print(
    hub_profile[
        ["hub_id", "delay_rate", "cluster"]
    ].to_string(index=False)
)

print("\nCluster Sizes:")
print(
    hub_profile["cluster"]
    .value_counts()
    .sort_values(ascending=False)
)

# Task 3: Cluster separation evidence

from itertools import combinations
from scipy.spatial.distance import euclidean

separation_results = []

for k in [2, 3, 4, 5]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    within_distances = []
    across_distances = []

    # Compare every unordered pair of hubs
    for i, j in combinations(range(len(X_scaled)), 2):

        distance = euclidean(X_scaled[i], X_scaled[j])

        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    across_over_within = (
        np.mean(across_distances) /
        np.mean(within_distances)
    )

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_values(ascending=False)
    )

    size_string = "-".join(
        cluster_sizes.astype(str)
    )

    separation_results.append({
        "K": k,
        "across_over_within": across_over_within,
        "cluster_sizes": size_string
    })

separation_df = pd.DataFrame(separation_results)

separation_df.to_csv(
    OUTPUT_DIR / "part_d_separation.csv",
    index=False
)

print("\nCluster Separation Results:")
print(separation_df.to_string(index=False))


# Task 4: Assign business segment names

# Calculate average delay rate for each cluster
cluster_delay_rates = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

# Map clusters to business segment names
segment_names = {
    cluster_delay_rates.index[0]: "Standard",
    cluster_delay_rates.index[1]: "Watch",
    cluster_delay_rates.index[2]: "Escalate"
}

hub_profile["segment"] = (
    hub_profile["cluster"].map(segment_names)
)

# Load hub details
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

# Create hub segments file
hub_segments = hub_profile[
    ["hub_id", "segment"]
].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left"
)

hub_segments.to_csv(
    OUTPUT_DIR / "hub_segments.csv",
    index=False
)

print("\nHub Segments:")
print(hub_segments.to_string(index=False))

# Create segment profile
segment_profile = (
    hub_profile.groupby("segment")
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
    .reset_index()
)

# Ensure business order
segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = (
    segment_profile.sort_values("segment")
    .reset_index(drop=True)
)

segment_profile.to_csv(
    OUTPUT_DIR / "segment_profile.csv",
    index=False
)

print("\nSegment Profiles:")
print(segment_profile.to_string(index=False))


# Tier versus segment mismatch

expected_segment_by_tier = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(expected_segment_by_tier)
)

tier_mismatch = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
].copy()

tier_mismatch = tier_mismatch[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]

tier_mismatch.to_csv(
    OUTPUT_DIR / "tier_vs_segment_mismatch.csv",
    index=False
)

print("\nTier vs Segment Mismatches:")
print(tier_mismatch.to_string(index=False))

print("\nTotal mismatches:", len(tier_mismatch))
