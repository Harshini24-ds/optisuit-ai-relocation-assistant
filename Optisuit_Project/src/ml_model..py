"""
OptiSuit - ML Models
======================
Model 1 : Rent Prediction (Random Forest + XGBoost)
Model 2 : Suitability Classification (XGBoost + SMOTE)

Input  : optisuit_preprocessed_main_dataset.xlsx
Output : rent_model.pkl
         suitability_model.pkl
         optisuit_preprocessed_main_dataset.xlsx (updated)

Feature cols for ML (scaled):
  city, area, house_type, size_sqft, furnishing,
  avg_meal_price, accident_count, crime_count,
  police_station_count, traffic_level, congestion_index,
  distance_km, risk_index, traffic_score

Real value cols (NOT used for ML training):
  monthly_food_cost  → for TCoL formula
  commute_cost       → for TCoL formula
  living_cost_index  → for dashboard
  safety_score       → for dashboard
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, classification_report, accuracy_score)
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ==================== PATHS ====================
BASE_DIR         = r'C:\Users\harsh\Desktop\OptiSuit_Project\Optisuit_Project'
DATA_PATH        = os.path.join(BASE_DIR, 'data', 'processed',
                   'feature_engineered+processed_maindataset.xlsx')
RENT_MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'rent_model.pkl')
SUIT_MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'suitability_model.pkl')
OUTPUT_PATH      = os.path.join(BASE_DIR, 'data', 'processed',
                   'final_rent_main_dataset.xlsx')
ENC_PATH         = os.path.join(BASE_DIR, 'data', 'processed',
                   'label_encoders.pkl')

os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

# ==================== FEATURE COLUMNS ====================
# Only scaled columns for ML training
feature_cols = [
    'city', 'area', 'house_type', 'size_sqft', 'furnishing',
    'avg_meal_price', 'accident_count', 'crime_count',
    'police_station_count', 'traffic_level', 'congestion_index',
    'distance_km', 'risk_index', 'traffic_score'
]

print("=" * 55)
print("   OPTISUIT - ML MODELS")
print("=" * 55)

# ============================================================
# LOAD DATA
# ============================================================
print("\n📂 Loading Data...")
main = pd.read_excel(DATA_PATH, engine='openpyxl')
print(f"   ✅ Shape: {main.shape}")
print(f"   ✅ Columns: {list(main.columns)}")

X = main[feature_cols]

# ============================================================
# MODEL 1 - RENT PREDICTION
# ============================================================
print("\n" + "=" * 55)
print("   MODEL 1: RENT PREDICTION")
print("=" * 55)

print("\n📋 Preparing Features & Target...")
y_rent = main['actual_rent']
print(f"   ✅ Features : {len(feature_cols)} columns (scaled)")
print(f"   ✅ Target   : actual_rent (₹{y_rent.min()} - ₹{y_rent.max()})")

print("\n✂️  Train Test Split (80% / 20%)...")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_rent, test_size=0.2, random_state=42
)
print(f"   ✅ X_train: {X_train_r.shape} | X_test: {X_test_r.shape}")

# Train Random Forest
print("\n🤖 Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators = 100,
    random_state = 42,
    n_jobs       = -1
)
rf_model.fit(X_train_r, y_train_r)
rf_pred = rf_model.predict(X_test_r)
rf_mae  = mean_absolute_error(y_test_r, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test_r, rf_pred))
rf_r2   = r2_score(y_test_r, rf_pred)
print(f"   ✅ MAE: ₹{rf_mae:.2f} | RMSE: ₹{rf_rmse:.2f} | R2: {rf_r2:.4f}")

# Train XGBoost
print("\n🤖 Training XGBoost...")
xgb_r = XGBRegressor(
    n_estimators  = 100,
    learning_rate = 0.1,
    max_depth     = 6,
    random_state  = 42,
    verbosity     = 0
)
xgb_r.fit(X_train_r, y_train_r)
xgb_r_pred = xgb_r.predict(X_test_r)
xgb_r_mae  = mean_absolute_error(y_test_r, xgb_r_pred)
xgb_r_rmse = np.sqrt(mean_squared_error(y_test_r, xgb_r_pred))
xgb_r_r2   = r2_score(y_test_r, xgb_r_pred)
print(f"   ✅ MAE: ₹{xgb_r_mae:.2f} | RMSE: ₹{xgb_r_rmse:.2f} | R2: {xgb_r_r2:.4f}")

# Select best model
if xgb_r_r2 >= rf_r2:
    best_rent_model = xgb_r
    best_rent_name  = 'XGBoost'
    best_rent_r2    = xgb_r_r2
    best_rent_mae   = xgb_r_mae
else:
    best_rent_model = rf_model
    best_rent_name  = 'Random Forest'
    best_rent_r2    = rf_r2
    best_rent_mae   = rf_mae

print(f"\n🏆 Best Model : {best_rent_name}")
print(f"   R2         : {best_rent_r2:.4f}")
print(f"   MAE        : ₹{best_rent_mae:.2f}")

# Predict on full dataset
print("\n🔮 Predicting Rent on Full Dataset...")
main['predicted_rent'] = best_rent_model.predict(X).round(0).astype(int)
print(f"   ✅ predicted_rent filled!")
print(f"      min : ₹{main['predicted_rent'].min():,}")
print(f"      max : ₹{main['predicted_rent'].max():,}")
print(f"      mean: ₹{main['predicted_rent'].mean():,.0f}")

# Calculate predicted_TCoL using REAL values
print("\n⚙️  Calculating predicted_TCoL...")
print(f"   Formula: predicted_rent + monthly_food_cost + commute_cost")
main['predicted_TCoL'] = (
    main['predicted_rent']    +   # ML predicted  ✅
    main['monthly_food_cost'] +   # real ₹ value  ✅
    main['commute_cost']          # real ₹ value  ✅
).round(2)
print(f"   ✅ predicted_TCoL filled!")
print(f"      min : ₹{main['predicted_TCoL'].min():,.0f}")
print(f"      max : ₹{main['predicted_TCoL'].max():,.0f}")
print(f"      mean: ₹{main['predicted_TCoL'].mean():,.0f}")

# Save rent model
pickle.dump(best_rent_model, open(RENT_MODEL_PATH, 'wb'))
print(f"\n   ✅ rent_model.pkl saved!")

# ============================================================
# MODEL 2 - SUITABILITY CLASSIFICATION
# ============================================================
print("\n" + "=" * 55)
print("   MODEL 2: SUITABILITY CLASSIFICATION")
print("=" * 55)

print("\n🏷️  Creating Suitability Labels...")
def create_suitability(row):
    rent   = row['actual_rent']
    safety = row['actual_safety_score']
    if rent <= 10000 and safety >= 75:
        return 0   # Highly Suitable
    elif rent <= 20000 and safety >= 55:
        return 1   # Moderately Suitable
    else:
        return 2   # Not Suitable

label_map                 = {0: 'Highly Suitable',
                              1: 'Moderately Suitable',
                              2: 'Not Suitable'}
main['suitability_label'] = main.apply(create_suitability, axis=1)

for k, v in label_map.items():
    count = (main['suitability_label'] == k).sum()
    print(f"   ✅ {v:<25} : {count} rows")

y_suit = main['suitability_label']

print("\n✂️  Train Test Split (80% / 20%)...")
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y_suit, test_size=0.2, random_state=42, stratify=y_suit
)
print(f"   ✅ X_train: {X_train_s.shape} | X_test: {X_test_s.shape}")

print("\n⚖️  Applying SMOTE...")
print(f"   Before: {y_train_s.value_counts().to_dict()}")
smote                    = SMOTE(random_state=42)
X_train_sm, y_train_sm   = smote.fit_resample(X_train_s, y_train_s)
print(f"   After : {pd.Series(y_train_sm).value_counts().to_dict()}")
print(f"   ✅ Class imbalance handled!")

print("\n🤖 Training XGBoost Classifier...")
xgb_c = XGBClassifier(
    n_estimators  = 100,
    learning_rate = 0.1,
    max_depth     = 6,
    random_state  = 42,
    verbosity     = 0
)
xgb_c.fit(X_train_sm, y_train_sm)

pred_s = xgb_c.predict(X_test_s)
acc    = accuracy_score(y_test_s, pred_s)
print(f"\n🏆 Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"\n   Classification Report:")
print(classification_report(y_test_s, pred_s,
      target_names=['Highly Suitable', 'Moderately Suitable', 'Not Suitable']))

print("\n🔮 Predicting Suitability on Full Dataset...")
main['suitability_class'] = xgb_c.predict(X)
main['suitability_class'] = main['suitability_class'].map(label_map)
print(f"   ✅ suitability_class filled!")
for v in label_map.values():
    count = (main['suitability_class'] == v).sum()
    print(f"      {v:<25} : {count} rows")

pickle.dump(xgb_c, open(SUIT_MODEL_PATH, 'wb'))
print(f"\n   ✅ suitability_model.pkl saved!")

# ============================================================
# SAVE UPDATED DATASET
# ============================================================
print("\n💾 Saving Updated Dataset...")
main.drop(columns=['suitability_label'], inplace=True)
main.to_excel(OUTPUT_PATH, index=False, engine='openpyxl')
print(f"   ✅ optisuit_preprocessed_main_dataset.xlsx updated!")
print(f"   ✅ Shape: {main.shape}")

# ============================================================
# TERMINAL PREVIEW
# ============================================================
print("\n" + "=" * 55)
print("   PREDICTIONS PREVIEW (First 10 rows)")
print("=" * 55)

le = pickle.load(open(ENC_PATH, 'rb'))

print(f"\n{'#':<4} {'City':<12} {'Area':<18} {'Type':<8} {'Actual':>10} {'Predicted':>10} {'TCoL':>10} {'Suitability'}")
print("─" * 90)

for i, row in main.head(10).iterrows():
    city  = le['city'].inverse_transform([int(row['city'])])[0]
    area  = le['area'].inverse_transform([int(row['area'])])[0]
    htype = le['house_type'].inverse_transform([int(row['house_type'])])[0]
    print(f"{i+1:<4} {city:<12} {area:<18} {htype:<8} "
          f"₹{int(row['actual_rent']):>8,} "
          f"₹{int(row['predicted_rent']):>8,} "
          f"₹{row['predicted_TCoL']:>8,.0f} "
          f"{row['suitability_class']}")

print("\n📈 Overall Statistics:")
print(f"   Actual Rent    → avg: ₹{main['actual_rent'].mean():,.0f} | min: ₹{main['actual_rent'].min():,} | max: ₹{main['actual_rent'].max():,}")
print(f"   Predicted Rent → avg: ₹{main['predicted_rent'].mean():,.0f} | min: ₹{main['predicted_rent'].min():,} | max: ₹{main['predicted_rent'].max():,}")
print(f"   Predicted TCoL → avg: ₹{main['predicted_TCoL'].mean():,.0f} | min: ₹{main['predicted_TCoL'].min():,.0f} | max: ₹{main['predicted_TCoL'].max():,.0f}")
print(f"\n   Suitability Distribution:")
for v, c in main['suitability_class'].value_counts().items():
    print(f"      {v:<25} : {c} rows")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("   ✅ ALL ML MODELS COMPLETE!")
print("=" * 55)
print(f"""
Model 1 - Rent Prediction:
  Best Model : {best_rent_name}
  R2 Score   : {best_rent_r2:.4f}
  MAE        : ₹{best_rent_mae:.2f}
  ✅ predicted_rent  → filled! (real ₹ values)
  ✅ predicted_TCoL  → filled! (real ₹ values)

Model 2 - Suitability Classification:
  Algorithm  : XGBoost + SMOTE
  Accuracy   : {acc*100:.2f}%
  ✅ suitability_class → filled!

Models saved in models/:
  ✅ rent_model.pkl
  ✅ suitability_model.pkl

Next Step → Run src/ml_model3_clustering.py 🚀
""")
