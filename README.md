# Delhivery Last-Mile Delay Intelligence Pipeline

## Project Overview

This project develops a machine learning pipeline to analyze and predict last-mile shipment delays using shipment, hub, and operational data.

The project covers:
- Data cleaning and exploratory data analysis
- Delivery-time regression
- Shipment-delay classification
- Operational segmentation
- Model packaging and prediction

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/TanishqBanger/delhivery-capstone.git
cd delhivery-capstone
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project pins pandas, NumPy, scikit-learn and joblib to specific versions to support reproducibility.

### 4. Generate the datasets

```bash
python generate_data.py
```

### 5. Run the analysis and model development scripts

```bash
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
```

### 6. Package the final classification model

```bash
python package_model.py
```

This script rebuilds and trains the final classifier using the saved Part C configuration. It regenerates `models/delay_model.pkl` and the supporting model configuration files.

### 7. Generate predictions for new shipments

```bash
python predict_delay.py
```

The prediction script loads the packaged model, saved threshold, feature order, imputation medians and hub segments. It generates `outputs/new_shipment_actions.csv`.

**Reproducibility note:** The model pickle should be loaded using the pinned scikit-learn version. If the installed version differs, rerun `package_model.py` in the environment created from `requirements.txt` to regenerate the model.

## Project Structure

```text
delhivery-capstone/
├── README.md
├── requirements.txt
├── generate_data.py
├── data/
├── src/
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── models/
└── outputs/
```

## Part A: Data Cleaning and Exploratory Data Analysis

### Data Cleaning

- Removed duplicate shipment records.
- Standardized service type values.
- Replaced missing payment modes with `Unknown`.
- Converted declared values into numeric INR values.
- Converted dispatch timestamps into datetime values.
- Treated parcel weights above 100 kg as missing scanner-fault values.
- Imputed missing distance, parcel weight, and declared value using their respective medians.
- Merged shipment data with hub information.

### Dataset Summary

| Metric | Result |
|---|---:|
| Original rows | 1,224 |
| Duplicate rows removed | 24 |
| Rows after cleaning | 1,200 |
| Delayed shipments | 278 |
| Overall delay rate | 23.17% |
| Missing values in features and target | 0 |

### Data Findings

#### 1. Delay patterns by service type

Express shipments have the highest delay rate at 48.97% (119 out of 243 shipments), followed by Standard at 20.56% and Economy at 8.57%.

#### 2. Delay patterns by hub tier

The delay rate increases from 4.00% for Tier 1 hubs to 21.50% for Tier 2 and 44.00% for Tier 3 hubs.

#### 3. Impact of delivery distance

Delay rates increase with distance:

- 0–200 km: 2.19%
- 201–400 km: 13.68%
- 401–600 km: 43.64%
- 601+ km: 85.23%

#### 4. Strongest correlations with delivery time

The three strongest correlations with `delivery_hours`, ranked by absolute correlation, are:

| Feature | Correlation |
|---|---:|
| `promised_hours` | +0.8372 |
| `distance_km` | +0.4013 |
| `n_stops` | +0.3247 |

These correlations indicate associations with actual delivery time, not necessarily causal relationships.

#### 5. Declared-value log transformation

The median declared value is ₹1,121.50, while the mean is ₹1,533.05, indicating a right-skewed distribution.

After applying the natural logarithm:

- Log mean: 7.0277
- Log median: 7.0224

The log transformation reduces the influence of large declared values and makes the distribution more balanced.

## Generated Part A Outputs

- `data/modelling_table.csv`
- `outputs/part_a_checks.csv`
- `outputs/part_a_delay_by_group.csv`
- `outputs/part_a_correlations.csv`

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib


# Part B
## Model results (regression)

### 1. Baseline performance

The mean baseline achieved an R² score of -0.0212 on the validation set. This indicates that simply predicting the average delivery time provides poor predictive performance.

### 2. Model comparison

Linear Regression achieved the lowest validation RMSE of **4.8912 hours**, narrowly outperforming Gradient Boosting (4.9165 hours).

| Model | Validation RMSE | Validation R² |
|---|---:|---:|
| Linear Regression | 4.8912 | 0.9313 |
| Decision Tree | 6.2796 | 0.8867 |
| Random Forest | 5.5095 | 0.9128 |
| Gradient Boosting | 4.9165 | 0.9306 |

Linear Regression was selected as the final model based on its lowest validation RMSE.

### 3. Decision Tree depth analysis

A depth-1 tree showed high bias, with training RMSE of 11.6334 and validation RMSE of 12.1324.

Increasing the depth improved performance up to depth 6, which achieved the lowest validation RMSE of 6.2796.

At greater depths, training error continued to decrease while validation error increased. The unlimited-depth tree achieved zero training RMSE but validation RMSE of 7.8368, indicating overfitting.

### 4. Gradient Boosting learning rate

| Learning rate | Validation RMSE | Validation R² |
|---|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3701 | 0.8440 |

A learning rate of 0.1 achieved the best validation performance. A learning rate of 0.01 learned too slowly with 100 estimators, while 1.0 was too aggressive and produced weaker validation performance.

### 5. Cross-validation

Five-fold cross-validation produced these results:

| Model | Mean R² | Std R² |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9353 | 0.0045 |

Although Gradient Boosting performed better on average during cross-validation, Linear Regression was retained because it achieved the lowest RMSE on the designated validation set.

### 6. Final test performance

The selected Linear Regression model achieved the following results on the held-out test set:

- MAE: **3.5671 hours**
- MSE: **20.4624**
- RMSE: **4.5235 hours**
- R²: **0.9493**

The final model explains approximately 94.9% of the variation in delivery time on the test set.

# Part C
## Model results (classification)

### 1. Baseline and model comparison

The blind baseline predicts every shipment as on-time. It achieves 76.67% accuracy because 76.67% of validation shipments are on-time, but its recall and F1-score for delayed shipments are both zero. Therefore, accuracy alone is not sufficient for evaluating this classification task.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Blind baseline | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree (depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest (default) | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

All three trained models outperform the blind baseline. Logistic Regression achieves the highest F1 among the initial model comparisons.

### 2. Feature importance and noise features

The top seven features ranked by ANOVA F-score are:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

All five noise columns have F-scores below 10 and ranks of 8 or lower in importance (rank 8 or worse). Their combined Random Forest importance is 0.2244, approximately 22.44%.

### 3. Feature selection

| Selected features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

Seven selected features achieved the highest validation F1-score. Therefore, k=7 was used for subsequent tuning and final model development.

### 4. Hyperparameter tuning

| Search | Candidates | Total fits | Best CV accuracy | Validation F1 |
|---|---:|---:|---:|---:|
| Grid Search | 27 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 45 | 0.8905 | 0.7848 |
| Default Random Forest | 1 | 1 | — | 0.8333 |

The default Random Forest achieved the highest validation F1-score. Although Grid Search achieved slightly higher cross-validation accuracy than Randomized Search, neither tuned model outperformed the default model on validation F1.

This demonstrates that hyperparameter tuning does not necessarily improve validation performance, particularly when model selection is based on a relatively small validation set.

### 5. Operating threshold selection

The operating threshold was selected using validation recall. The rule was to choose the highest threshold with recall of at least 0.80.

The selected threshold was **0.5**, with validation precision of 0.7955, recall of 0.8333 and F1-score of 0.8140.

Increasing the threshold generally reduced recall while increasing precision, illustrating the trade-off between identifying delayed shipments and avoiding false delay predictions.

### 6. Final test evaluation

The final model used SelectKBest with k=7 and a default Random Forest with 100 estimators, unlimited maximum depth and minimum leaf size of 1.

The final test evaluation was performed once using the selected operating threshold of 0.5.

| Metric | Test result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1-score | 0.7209 |

The final model achieved a test F1-score of 0.7209, with recall of 0.7561 and precision of 0.6889. These results satisfy the specified evaluation requirements.

## Operations recommendation

### 1. Why three hub segments were selected

K-Means separation increased as the number of clusters increased:

| K | Across-to-within ratio | Cluster sizes |
|---:|---:|---|
| 2 | 2.894 | 15-9 |
| 3 | 5.393 | 9-8-7 |
| 4 | 6.305 | 8-7-5-4 |
| 5 | 6.887 | 8-7-4-3-2 |

Although K=4 and K=5 achieved higher separation ratios, K=3 was retained because it provides a more practical operational structure.

The K=4 and K=5 solutions contain clusters with fewer than 6 hubs. Such small groups may not have enough hubs to justify a separate operational playbook. Three segments provide a better balance between cluster separation and operational simplicity.

### 2. Hub segment recommendations

The final hub segments are named according to their average delay rates:

| Segment | Hubs | Average delay rate | Operational approach |
|---|---:|---:|---|
| Standard | 7 | 2.57% | Maintain existing operations and routine monitoring |
| Watch | 8 | 15.50% | Monitor shipment delays and investigate emerging issues |
| Escalate | 9 | 46.00% | Prioritize capacity, dispatch and delivery-process reviews |

The Escalate segment contains HUB13, HUB15, HUB17, HUB18, HUB19, HUB20, HUB21, HUB23 and HUB24.

These hubs should receive priority operational attention because their segment has the highest average delay rate.

### 3. Tier-versus-segment mismatches

Four hubs have a segment classification that differs from the expected segment associated with their hub tier:

| Hub | City | Hub tier | Actual segment |
|---|---|---:|---|
| HUB07 | Kolkata | 1 | Watch |
| HUB13 | Nagpur | 2 | Escalate |
| HUB15 | Kochi | 2 | Escalate |
| HUB22 | Dehradun | 3 | Watch |

These mismatches demonstrate that hub tier alone does not fully describe operational delay risk.

HUB07 should be investigated despite being a Tier 1 hub. HUB13 and HUB15 deserve additional attention because their delay performance is worse than the expected Tier 2 profile. HUB22 performs better than the expected Tier 3 profile and may provide useful operational lessons for other hubs.

### 4. Classification threshold and business impact

The selected operating threshold is 0.5. A shipment is flagged when its predicted delay probability is greater than or equal to this threshold.

On the held-out test set:

- Precision: 0.6889
- Recall: 0.7561
- F1-score: 0.7209

In operational terms, approximately 6.9 out of every 10 flagged shipments were actually delayed, while approximately 7.6 out of every 10 actual delayed shipments were identified by the model.

This threshold balances identifying delayed shipments with avoiding unnecessary interventions.

### 5. New shipment action summary

The prediction pipeline evaluated 48 new shipments:

| Action | Shipments |
|---|---:|
| Escalate to hub manager | 6 |
| Proactive customer call | 2 |
| No action | 40 |
| Total | 48 |

The action policy is:

- Flagged shipment in an Escalate hub: Escalate to hub manager.
- Flagged shipment in a Standard or Watch hub: Proactive customer call.
- Shipment below the threshold: No action.

This approach combines shipment-level delay probability with hub-level operational risk to prioritize interventions.

## Production plan

### 1. Production readiness

| Readiness characteristic | Current implementation | Production improvement |
|---|---|---|
| Data quality | Shared cleaning and feature-generation functions in `src/shipment_prep.py` | Add schema validation and automated data-quality alerts |
| Model reproducibility | Pinned dependencies and `package_model.py` | Maintain versioned model artifacts and training metadata |
| Prediction consistency | Saved feature order, imputation medians and threshold | Validate incoming data and monitor prediction consistency |
| Operational integration | `predict_delay.py` generates shipment-level actions | Integrate predictions with the shipment management system |
| Model evaluation | Held-out test metrics and reload verification | Track precision, recall and F1 on newly completed shipments |
| Hub monitoring | K-Means hub segments and mismatch analysis | Refresh hub profiles and investigate changes in segment membership |

### 2. Monitoring quantities and proposed triggers

The following are proposed operational monitoring thresholds, not measured results from the current experiment.

| Quantity | Monitoring frequency | Proposed trigger | Recommended response |
|---|---|---|---|
| Missing feature values | Daily | More than 2% of incoming records | Investigate upstream data pipelines |
| Flagged shipment rate | Daily | More than 30% of shipments | Review hub conditions and incoming shipment mix |
| Model recall | Monthly, using completed shipments | Below 0.70 | Investigate missed delays and review the operating threshold |

### 3. Retraining schedule

Retrain the classification model monthly using the latest completed shipment records.

Before deployment, evaluate the updated model on a separate validation period and compare its precision, recall and F1-score with the currently deployed model. Replace the existing model only when the updated version meets the agreed operational performance requirements.

### 4. Sale-event monitoring

The dataset includes `sale_event_flag` to identify sale-event shipments.

For the sale window from 15–28 September, monitor delay rates, flagged shipment rates and model performance separately from ordinary operating periods.

Compare sale-event performance with previous comparable sale-event periods rather than directly comparing it with ordinary weeks. This helps account for the additional shipment volume and operational pressure associated with sale events.

### 5. Data and model drift

Monitor changes in shipment distance, service type, hub capacity, shipment volume and delay rates.

A change in the distribution of incoming features may indicate data drift. A decline in prediction performance after actual delivery outcomes become available may indicate concept drift.

Investigate these changes before deciding whether to retrain the model or update operational procedures.