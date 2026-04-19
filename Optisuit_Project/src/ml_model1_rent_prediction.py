# ============================================
# OPTISUIT - RENT PREDICTION MODEL (v4)
# ✅ Log transform on target (fixes skew)
# ✅ Stratified split by house_type
# ✅ Target encoding for location
# ✅ Cross Validation (before & after tuning)
# ✅ Hyperparameter Tuning (RandomizedSearchCV)
# ✅ No feature leakage
# ============================================

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV
)
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = r'C:\Users\harsh\Desktop\OptiSuit_Project\Optisuit_Project'
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'optisuitmodel.xlsx')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rent_model.pkl')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'optisuitmodel.xlsx')
print("=" * 55)
print("  OPTISUIT — RENT PREDICTION MODEL (v4)")
print("=" * 55)

# ================= LOAD =================
df = pd.read_excel(DATA_PATH, engine='openpyxl')
print(f"\nDataset shape: {df.shape}")

# ================= LOG TRANSFORM TARGET =================
# Rent is right-skewed (skew ~0.72). Log makes it normal,
# which is much easier for tree models to fit accurately.
df['log_rent'] = np.log1p(df['actual_rent'])
print(f"Rent skew       : {df['actual_rent'].skew():.3f}")
print(f"Log(rent) skew  : {df['log_rent'].skew():.3f}  (closer to 0 = better)")

# ================= FEATURES =================
base_features = [
    'size_sqft',
    'house_type',               # label encoded: PG=1, 1BHK=0, 2BHK=2
    'furnishing',               # label encoded
    'safety_score',
    'risk_index',
    'traffic_score',
    'overall_score',
    'accessibility_score',
    'police_station_count',
    'congestion_index',
    'distance_km',
    'traffic_level',
]
base_features = [f for f in base_features if f in df.columns]

X     = df[base_features].copy()
y     = df['log_rent']          # predict log(rent)
y_raw = df['actual_rent']       # keep original for MAE in rupees

# ================= STRATIFIED SPLIT by house_type =================
# Ensures PG / 1BHK / 2BHK are evenly represented in test set
X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X, y, y_raw,
    test_size=0.2,
    random_state=42,
    stratify=df['house_type']
)

print(f"\nTrain: {len(X_train)} rows  |  Test: {len(X_test)} rows")
print(f"Test house_type distribution:")
for ht, count in df.loc[X_test.index, 'house_type'].value_counts().sort_index().items():
    print(f"  house_type={ht}: {count} rows")

# ================= TARGET ENCODING (train only) =================
# Replaces area/city with mean rent from training data only.
# Prevents leakage while capturing location signal.
train_df    = df.loc[X_train.index]
area_mean   = train_df.groupby('area')['actual_rent'].mean()
city_mean   = train_df.groupby('city')['actual_rent'].mean()
global_mean = y_raw_train.mean()

X_train = X_train.copy()
X_test  = X_test.copy()

X_train['area_rent_enc'] = df.loc[X_train.index, 'area'].map(area_mean).fillna(global_mean)
X_train['city_rent_enc'] = df.loc[X_train.index, 'city'].map(city_mean).fillna(global_mean)
X_test['area_rent_enc']  = df.loc[X_test.index,  'area'].map(area_mean).fillna(global_mean)
X_test['city_rent_enc']  = df.loc[X_test.index,  'city'].map(city_mean).fillna(global_mean)

# ================= INTERACTION FEATURES =================
# Combining strong predictors creates richer signals
# without needing more data. These interactions reflect
# real-world relationships (e.g. a large 2BHK in a premium
# area costs disproportionately more than a small PG).
for X in [X_train, X_test]:
    X['size_x_housetype']    = X['size_sqft']    * X['house_type']
    X['size_x_area']         = X['size_sqft']    * X['area_rent_enc']
    X['housetype_x_area']    = X['house_type']   * X['area_rent_enc']
    X['safety_x_area']       = X['safety_score'] * X['area_rent_enc']
    X['size_sqft_sq']        = X['size_sqft']    ** 2   # captures non-linear size effect

print(f"\nTotal features after interactions: {X_train.shape[1]}")

# ================= CROSS VALIDATION — BASELINE =================
# Run 5-fold CV before tuning to get a baseline score.
# This tells us how stable the model is with default params.
# Target: CV Mean > 0.60, CV Std < 0.08
print("\n--- CROSS VALIDATION (Baseline — before tuning) ---")
baseline_models = {
    'XGBoost'      : XGBRegressor(random_state=42),
    'RandomForest' : RandomForestRegressor(random_state=42),
}
for name, m in baseline_models.items():
    cv = cross_val_score(m, X_train, y_train, cv=5, scoring='r2')
    print(f"{name:>15} | CV Scores: {[round(s,4) for s in cv]} | Mean: {cv.mean():.4f} | Std: {cv.std():.4f}")

# ================= HYPERPARAMETER TUNING =================
# RandomizedSearchCV tries 30 random parameter combinations
# per model using 5-fold CV on training data only.
# Automatically finds better params than manual guessing.
print("\n--- HYPERPARAMETER TUNING (RandomizedSearchCV, 30 iters, 5-fold CV) ---")

# XGBoost search space
xgb_param_grid = {
    'n_estimators'     : [100, 150, 200],
    'learning_rate'    : [0.02, 0.03, 0.05],
    'max_depth'        : [3, 4],                 # middle ground
    'subsample'        : [0.6, 0.7],
    'colsample_bytree' : [0.6, 0.7],
    'reg_alpha'        : [5, 8, 12],             # moderate L1
    'reg_lambda'       : [10, 15, 20],           # moderate L2
    'min_child_weight' : [10, 15, 20],           # moderate leaf size
    'gamma'            : [0, 0.5, 1],            # mild pruning
}

print("\nTuning XGBoost...")
xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_distributions=xgb_param_grid,
    n_iter=30, cv=5, scoring='r2',
    random_state=42, n_jobs=1
)
xgb_search.fit(X_train, y_train)
print(f"  Best CV R2 : {xgb_search.best_score_:.4f}")
print(f"  Best params: {xgb_search.best_params_}")

# RandomForest search space
rf_param_grid = {
    'n_estimators'    : [200, 300, 400, 500],
    'max_depth'       : [6, 7, 8, 10, None],
    'min_samples_leaf': [2, 3, 4],
    'max_features'    : [0.6, 0.7, 0.8, 'sqrt'],
    'min_samples_split': [2, 4, 6],
}

print("\nTuning RandomForest...")
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=30, cv=5, scoring='r2',
    random_state=42, n_jobs=1
)
rf_search.fit(X_train, y_train)
print(f"  Best CV R2 : {rf_search.best_score_:.4f}")
print(f"  Best params: {rf_search.best_params_}")

# ================= CROSS VALIDATION — AFTER TUNING =================
# Run CV again with tuned params to confirm improvement.
# Compare these scores with the baseline CV above.
print("\n--- CROSS VALIDATION (After Tuning) ---")
tuned_models = {
    'XGBoost'      : xgb_search.best_estimator_,
    'RandomForest' : rf_search.best_estimator_,
}
for name, m in tuned_models.items():
    cv = cross_val_score(m, X_train, y_train, cv=5, scoring='r2')
    print(f"{name:>15} | CV Scores: {[round(s,4) for s in cv]} | Mean: {cv.mean():.4f} | Std: {cv.std():.4f}")

# ================= EVALUATION ON TEST SET =================
# Evaluate tuned models on the held-out test set.
# R2 in log space. MAE converted back to rupees using expm1().
# Gap = Train R2 - Test R2  (target: below 0.10)
print(f"\n--- TEST SET EVALUATION ---")
print(f"\n{'Model':<15} {'Train R2':>10} {'Test R2':>10} {'Gap':>8} {'MAE (Rs)':>12}")
print("-" * 58)

results    = {}
best_model = None
best_score = -999

for name, m in tuned_models.items():
    m.fit(X_train, y_train)
    tr        = r2_score(y_train, m.predict(X_train))
    te        = r2_score(y_test,  m.predict(X_test))
    pred_rent = np.expm1(m.predict(X_test))
    mae       = mean_absolute_error(y_raw_test, pred_rent)

    results[name] = {'train': tr, 'test': te, 'gap': tr - te, 'mae': mae}
    print(f"{name:<15} {tr:>10.4f} {te:>10.4f} {tr-te:>8.4f} {mae:>12,.0f}")

    # Penalized selection: rewards high test R2, penalizes overfitting
    penalized_score = te - abs(tr - te) * 0.5
    if penalized_score > best_score:
        best_score = penalized_score
        best_model = m
        best_name  = name

model = best_model
print(f"\n  Best model (penalized selection): {best_name}")
print(f"  Test R2 = {results[best_name]['test']:.4f}  |  Gap = {results[best_name]['gap']:.4f}")

# ================= DIAGNOSIS =================
gap = results[best_name]['gap']
print("\nDIAGNOSIS")
if gap >= 0.10:
    print(f"  Overfitting   : Yes — gap is {gap:.4f}  (target < 0.10)")
else:
    print(f"  Generalization: Good — gap is {gap:.4f}")

if results[best_name]['test'] >= 0.80:
    print("  Accuracy      : Strong (above 80%)")
elif results[best_name]['test'] >= 0.70:
    print("  Accuracy      : Good (above 70% target met)")
elif results[best_name]['test'] >= 0.60:
    print("  Accuracy      : Acceptable (below 70% target)")
else:
    print("  Accuracy      : Needs improvement")

# ================= FEATURE IMPORTANCE =================
# Confirms the model is using sensible signals, not noise.
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
print("\nFEATURE IMPORTANCES")
print(feat_imp.sort_values(ascending=False).to_string())

# ================= SAMPLE PREDICTIONS =================
# Sanity check: compare actual vs predicted rent on test rows.
sample_preds = np.expm1(model.predict(X_test))
sample_df = pd.DataFrame({
    'Actual Rent (Rs)'   : y_raw_test.values[:20],
    'Predicted Rent (Rs)': sample_preds[:20].round(0),
    'Error (Rs)'         : (sample_preds[:20] - y_raw_test.values[:20]).round(0)
})
print("\nSAMPLE PREDICTIONS (first 8 test rows)")
print(sample_df.to_string(index=False))

# ================= FULL-DATA PREDICTIONS =================
# Create predicted_rent for every row so the dashboard can use
# model output instead of the ground-truth rent column.
X_full = df[base_features].copy()
X_full['area_rent_enc'] = df['area'].map(area_mean).fillna(global_mean)
X_full['city_rent_enc'] = df['city'].map(city_mean).fillna(global_mean)

X_full['size_x_housetype'] = X_full['size_sqft'] * X_full['house_type']
X_full['size_x_area'] = X_full['size_sqft'] * X_full['area_rent_enc']
X_full['housetype_x_area'] = X_full['house_type'] * X_full['area_rent_enc']
X_full['safety_x_area'] = X_full['safety_score'] * X_full['area_rent_enc']
X_full['size_sqft_sq'] = X_full['size_sqft'] ** 2

X_full = X_full[X_train.columns]
df['predicted_rent'] = np.expm1(model.predict(X_full)).round(0)
df.to_excel(OUTPUT_PATH, index=False, engine='openpyxl')
print(f"\nUpdated dataset saved with predicted_rent -> {OUTPUT_PATH}")

# ================= SAVE (model + encoding maps) =================
# Saved as a bundle so the dashboard has everything it needs
# to transform user inputs and make predictions.
model_bundle = {
    'model'      : model,
    'area_mean'  : area_mean,
    'city_mean'  : city_mean,
    'global_mean': global_mean,
    'features'   : X_train.columns.tolist(),
}
pickle.dump(model_bundle, open(MODEL_PATH, 'wb'))
print(f"\nModel bundle saved -> {MODEL_PATH}")
print("  (includes model + target encoding maps for dashboard use)")
print("\nDONE")
