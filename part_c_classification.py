import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.shipment_prep import FEATURES, NOISE_COLUMNS


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "modelling_table.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df["is_delayed"].value_counts().sort_index())

print("\nFeature count:", len(FEATURES))
print("Noise columns:", NOISE_COLUMNS)

# Stratified train-validation-test split
# Define features and target
X = df[FEATURES]
y = df["is_delayed"]

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Second split: 50% validation, 50% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# Check split sizes
print("\nSplit Shapes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# Check delayed parcel counts
print("\nDelayed Parcel Counts:")
print("Training:", y_train.sum())
print("Validation:", y_val.sum())
print("Test:", y_test.sum())

# Check total rows
print("\nTotal split rows:",
      len(X_train) + len(X_val) + len(X_test))

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        )
    }
    
# Blind baseline: predict every parcel is on time
blind_predictions = np.zeros(len(y_val), dtype=int)

blind_metrics = calculate_metrics(
    y_val,
    blind_predictions
)

print("\nBlind Baseline Results:")
print(blind_metrics)    


# logistic regression 

# Logistic Regression with StandardScaler
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

logistic_metrics = calculate_metrics(
    y_val,
    logistic_predictions
)

print("\nLogistic Regression Results:")
print(logistic_metrics)

tree_results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:

    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree_model.fit(X_train, y_train)

    tree_predictions = tree_model.predict(X_val)

    tree_metrics = calculate_metrics(
        y_val,
        tree_predictions
    )

    tree_results.append({
        "depth": depth,
        "f1": tree_metrics["f1"],
        "model": tree_model,
        "metrics": tree_metrics
    })

# Highest F1; ties go to smaller depth
best_tree = sorted(
    tree_results,
    key=lambda item: (-item["f1"], item["depth"])
)[0]

best_tree_model = best_tree["model"]
best_tree_depth = best_tree["depth"]
best_tree_metrics = best_tree["metrics"]

print("\nDecision Tree Depth Results:")

for result in tree_results:
    print(
        f"Depth: {result['depth']}, "
        f"F1: {result['f1']:.4f}"
    )

print("\nBest Tree Depth:", best_tree_depth)
print("Best Tree Metrics:", best_tree_metrics)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_val)

rf_metrics = calculate_metrics(
    y_val,
    rf_predictions
)

print("\nRandom Forest Results:")
print(rf_metrics)


# Combine all validation model results

model_rows = []

# Blind baseline
model_rows.append({
    "model": "blind_all_on_time",
    **blind_metrics
})

# Logistic Regression
model_rows.append({
    "model": "logistic_regression",
    **logistic_metrics
})

# Best Decision Tree
model_rows.append({
    "model": f"decision_tree_depth_{best_tree_depth}",
    **best_tree_metrics
})

# Random Forest
model_rows.append({
    "model": "random_forest_default",
    **rf_metrics
})

# Create DataFrame
models_df = pd.DataFrame(model_rows)

# Save CSV
models_df.to_csv(
    OUTPUT_DIR / "part_c_models.csv",
    index=False
)

print("\nPart C Model Comparison:")
print(models_df.to_string(index=False))

# Fit feature scoring models
# Random Forest feature importance using all 17 features

importance_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

importance_model.fit(X_train, y_train)

rf_importances = importance_model.feature_importances_


# ANOVA F-scores using SelectKBest

selector_all = SelectKBest(
    score_func=f_classif,
    k="all"
)

selector_all.fit(X_train, y_train)

f_scores = selector_all.scores_


# Create feature scores DataFrame

feature_scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances
})

# Rank features: 1 = highest F-score
feature_scores_df["f_rank"] = (
    feature_scores_df["f_score"]
    .rank(ascending=False, method="min")
    .astype(int)
)

# Identify noise columns
feature_scores_df["is_noise"] = (
    feature_scores_df["feature"].isin(NOISE_COLUMNS)
)

# Arrange columns in required order
feature_scores_df = feature_scores_df[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]

# Sort by F-score ranking
feature_scores_df = feature_scores_df.sort_values(
    "f_rank"
).reset_index(drop=True)

# Save output
feature_scores_df.to_csv(
    OUTPUT_DIR / "part_c_feature_scores.csv",
    index=False
)

print("\nFeature Scores:")
print(feature_scores_df.to_string(index=False))


# Combined noise importance
noise_importance = feature_scores_df.loc[
    feature_scores_df["is_noise"],
    "rf_importance"
].sum()

print("\nCombined Noise Feature Importance:", noise_importance)

# Test different feature counts (k)
k_results = []

for k in [5, 7, 10, 12, 17]:

    selection_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    selection_model.fit(X_train, y_train)

    predictions = selection_model.predict(X_val)

    val_f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    k_results.append({
        "k": k,
        "val_f1": val_f1
    })

# Save results
selectk_df = pd.DataFrame(k_results)

selectk_df.to_csv(
    OUTPUT_DIR / "part_c_selectk.csv",
    index=False
)

print("\nSelectKBest Results:")
print(selectk_df.to_string(index=False))

# Hyperparameter tuning
# Step 8: Hyperparameter tuning using top 7 features

selector = SelectKBest(f_classif, k=7)

X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

print("Selected training shape:", X_train_selected.shape)
print("Selected validation shape:", X_val_selected.shape)
print("Selected features:", selector.get_feature_names_out(FEATURES))

# Configure Grid Search and Randomized Search
# Hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)

grid_search.fit(X_train_selected, y_train)

print("\nGrid Search Results:")
print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random_search.fit(X_train_selected, y_train)

print("\nRandomized Search Results:")
print("Best parameters:", random_search.best_params_)
print("Best CV accuracy:", random_search.best_score_)

# Evaluate the default Random Forest
# Default Random Forest using the same 7 selected features

default_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_rf.fit(X_train_selected, y_train)

# Validation predictions
grid_predictions = grid_search.best_estimator_.predict(X_val_selected)
random_predictions = random_search.best_estimator_.predict(X_val_selected)
default_predictions = default_rf.predict(X_val_selected)

# Calculate validation F1 scores
grid_val_f1 = f1_score(
    y_val, grid_predictions, zero_division=0
)

random_val_f1 = f1_score(
    y_val, random_predictions, zero_division=0
)

default_val_f1 = f1_score(
    y_val, default_predictions, zero_division=0
)

print("\nValidation F1 Comparison:")
print("Grid Search:", grid_val_f1)
print("Randomized Search:", random_val_f1)
print("Default Random Forest:", default_val_f1)

# Save hyperparameter tuning results

tuning_results = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_val_f1
    }
]

tuning_df = pd.DataFrame(tuning_results)

tuning_df.to_csv(
    OUTPUT_DIR / "part_c_tuning.csv",
    index=False
)

print("\nTuning Results:")
print(tuning_df.to_string(index=False))

# Build the final model
# Step 9: Build the final classification model

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Fit using training data only
final_model.fit(X_train, y_train)

print("\nFinal Model:")
print("Selected features:", final_model.named_steps["selector"].get_feature_names_out(FEATURES))
print("Model parameters:", final_model.named_steps["model"].get_params())

# Calculate validation thresholds
# Step 10: Threshold evaluation on validation data

val_probabilities = final_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(y_val, val_predictions)

    threshold_results.append({
        "threshold": threshold,
        **metrics
    })

thresholds_df = pd.DataFrame(threshold_results)

thresholds_df.to_csv(
    OUTPUT_DIR / "part_c_thresholds.csv",
    index=False
)

print("\nThreshold Results:")
print(thresholds_df.to_string(index=False))

# Save the final configuration
# Step 11: Select operating threshold

qualifying_thresholds = thresholds_df[
    thresholds_df["recall"] >= 0.80
]

if len(qualifying_thresholds) > 0:
    chosen_threshold = qualifying_thresholds["threshold"].max()
else:
    chosen_threshold = 0.1

print("\nChosen operating threshold:", chosen_threshold)

# Save final model configuration

final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": 100,
    "max_depth": "unlimited",
    "min_samples_leaf": 1,
    "threshold": chosen_threshold,
    "chosen_from": "default_forest_same_7_features"
}])

final_config.to_csv(
    OUTPUT_DIR / "part_c_final_config.csv",
    index=False
)

print("\nFinal Configuration:")
print(final_config.to_string(index=False))

# Final test evaluation
# Step 12: Final test evaluation (one-time evaluation)

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

test_metrics = calculate_metrics(y_test, test_predictions)

final_test_df = pd.DataFrame([{
    "threshold": chosen_threshold,
    **test_metrics
}])

final_test_df.to_csv(
    OUTPUT_DIR / "part_c_final_test.csv",
    index=False
)

print("\nFinal Test Results:")
print(final_test_df.to_string(index=False))