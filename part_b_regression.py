import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.model_selection import KFold, cross_val_score

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project paths
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load modelling dataset
df = pd.read_csv(DATA_DIR / "modelling_table.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

from src.shipment_prep import FEATURES

# Input features
X = df[FEATURES]

# Target variable
y = df["delivery_hours"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget name:", y.name)


# Step 4 — Train / Validation / Test split
# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Second split: 15% validation, 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

# Display split sizes
print("\nDataset split sizes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# Step 5 — Scaling without data leakage
# Initialize scaler
scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the fitted scaler
X_test_scaled = scaler.transform(X_test)

# Calculate maximum absolute column means
train_scaled_max_abs_mean = np.abs(
    X_train_scaled.mean(axis=0)
).max()

test_scaled_max_abs_mean = np.abs(
    X_test_scaled.mean(axis=0)
).max()

print("\nScaling checks:")
print("Training scaled max abs mean:", train_scaled_max_abs_mean)
print("Testing scaled max abs mean:", test_scaled_max_abs_mean)


# Step 6 — Save Part B checks
# Save Part B validation checks
checks = {
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "train_scaled_max_abs_mean": train_scaled_max_abs_mean,
    "test_scaled_max_abs_mean": test_scaled_max_abs_mean,
}

checks_df = pd.DataFrame(
    list(checks.items()),
    columns=["metric", "value"]
)

checks_df.to_csv(
    OUTPUT_DIR / "part_b_checks.csv",
    index=False
)

print("\nPart B checks saved successfully!")
print(checks_df)

# Step 7 — Mean Baseline Regression
# Mean baseline prediction
baseline_prediction = np.full(
    len(y_val),
    y_train.mean()
)

# Calculate validation metrics
baseline_mae = mean_absolute_error(
    y_val,
    baseline_prediction
)

baseline_mse = mean_squared_error(
    y_val,
    baseline_prediction
)

baseline_rmse = np.sqrt(baseline_mse)

baseline_r2 = r2_score(
    y_val,
    baseline_prediction
)

print("\nMean Baseline Results")
print("---------------------")
print("Training target mean:", y_train.mean())
print("MAE:", baseline_mae)
print("MSE:", baseline_mse)
print("RMSE:", baseline_rmse)
print("R2:", baseline_r2)


# Step 8 — Decision Tree depth diagnostic
# Decision Tree depth experiment
depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_results = []

for depth in depths:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_prediction)
    )

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_prediction)
    )

    tree_results.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

# Create results DataFrame
tree_results_df = pd.DataFrame(tree_results)

# Save results
tree_results_df.to_csv(
    OUTPUT_DIR / "part_b_tree_depth.csv",
    index=False
)

print("\nDecision Tree Depth Results")
print(tree_results_df.to_string(index=False))


# Step 9 — Build the four candidate regression models
# Select the best decision tree depth
best_depth = int(
    tree_results_df.loc[
        tree_results_df["val_rmse"].idxmin(),
        "max_depth"
    ]
)

# Define candidate models
models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "decision_tree": DecisionTreeRegressor(
        max_depth=best_depth,
        random_state=42
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

# Store validation metrics
model_results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predictions)

    model_results.append({
        "model": name,
        "split": "val",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

# Add baseline metrics
model_results.append({
    "model": "mean_baseline",
    "split": "val",
    "MAE": baseline_mae,
    "MSE": baseline_mse,
    "RMSE": baseline_rmse,
    "R2": baseline_r2
})

# Create and save metrics
metrics_df = pd.DataFrame(model_results)

metrics_df.to_csv(
    OUTPUT_DIR / "part_b_metrics.csv",
    index=False
)

print("\nRegression Model Comparison")
print(metrics_df.to_string(index=False))


# Step 10 — Gradient Boosting learning-rate experiment
# Gradient Boosting learning-rate experiment

learning_rates = [0.01, 0.1, 1.0]

gb_results = []

for lr in learning_rates:

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    predictions = gb_model.predict(X_val)

    val_rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    val_r2 = r2_score(y_val, predictions)

    gb_results.append({
        "learning_rate": lr,
        "val_rmse": val_rmse,
        "val_r2": val_r2
    })

# Create results DataFrame
gb_results_df = pd.DataFrame(gb_results)

# Save results
gb_results_df.to_csv(
    OUTPUT_DIR / "part_b_gb_learning_rate.csv",
    index=False
)

print("\nGradient Boosting Learning Rate Results")
print(gb_results_df.to_string(index=False))


# Combine training and validation datasets
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Create one shared KFold object
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Select the two best models
top_models = {
    "linear_regression": models["linear_regression"],
    "gradient_boosting": models["gradient_boosting"]
}

cv_results = []

for name, model in top_models.items():

    scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=kf
    )

    cv_results.append({
        "model": name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

# Create and save results
cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    OUTPUT_DIR / "part_b_cv.csv",
    index=False
)

print("\n5-Fold Cross-Validation Results")
print(cv_df.to_string(index=False))


# Step 12 — Final test evaluation
# Select final model using validation RMSE
best_model_name = (
    metrics_df[
        metrics_df["model"] != "mean_baseline"
    ]
    .sort_values("RMSE")
    .iloc[0]["model"]
)

final_model = models[best_model_name]

# Predict on test data exactly once
test_predictions = final_model.predict(X_test)

# Calculate final test metrics
test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_mse = mean_squared_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    y_test,
    test_predictions
)

# Add final test result to metrics
test_result = pd.DataFrame([{
    "model": best_model_name,
    "split": "test",
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2
}])

metrics_df = pd.concat(
    [metrics_df, test_result],
    ignore_index=True
)

metrics_df.to_csv(
    OUTPUT_DIR / "part_b_metrics.csv",
    index=False
)

print("\nFinal Model:", best_model_name)
print("\nFinal Test Results")
print(test_result.to_string(index=False))