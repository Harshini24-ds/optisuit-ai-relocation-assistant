# ============================================================
# OPTISUIT - PREPROCESSING + FEATURE ENGINEERING PIPELINE
# Input  : main_dataset.xlsx + food_dataset.xlsx (raw)
# Output : optisuit_model_ready.xlsx
# ============================================================

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = r'C:\Users\harsh\Desktop\OptiSuit_Project\Optisuit_Project'
RAW_DIR    = os.path.join(BASE_DIR, 'data', 'raw')
PROC_DIR   = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MAIN_INPUT  = os.path.join(RAW_DIR,  'main_dataset.xlsx')
FOOD_INPUT  = os.path.join(RAW_DIR,  'food_dataset.xlsx')
OUTPUT_PATH = os.path.join(PROC_DIR, 'optisuitmodel.xlsx')
ENC_PATH    = os.path.join(MODELS_DIR, 'label_encoders.pkl')

print("=" * 55)
print("  OPTISUIT — PREPROCESSING + FEATURE ENGINEERING")
print("=" * 55)

# ============================================================
# 1. LOAD
# ============================================================
main = pd.read_excel(MAIN_INPUT, engine='openpyxl')
food = pd.read_excel(FOOD_INPUT, engine='openpyxl')

print(f"\n[LOAD] Main : {main.shape}  |  Food : {food.shape}")

# ============================================================
# 2. DROP PLACEHOLDER COLUMNS (all NaN — filled by models later)
# ============================================================
placeholder_cols = [
    'predicted_rent', 'monthly_food_cost', 'commute_cost',
    'safety_score', 'predicted_TCoL', 'suitability_class',
    'area_cluster', 'outlier_flag'
]
main.drop(columns=[c for c in placeholder_cols if c in main.columns], inplace=True)
print(f"[LOAD] Dropped {len(placeholder_cols)} placeholder columns")

# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================
before = len(main)
main.drop_duplicates(inplace=True)
print(f"[DEDUP] Removed {before - len(main)} duplicate rows  →  {len(main)} rows remain")

# ============================================================
# 4. FIX TYPES
# ============================================================
num_cols = ['size_sqft', 'avg_meal_price', 'accident_count', 'crime_count',
            'police_station_count', 'congestion_index', 'distance_km',
            'actual_rent', 'actual_safety_score']
for col in num_cols:
    main[col] = pd.to_numeric(main[col], errors='coerce')

# Fill any numeric NaNs with median
main.fillna(main.median(numeric_only=True), inplace=True)
print(f"[TYPES] Numeric types fixed, NaNs filled with median")

# ============================================================
# 5. OUTLIER REMOVAL (IQR on key columns)
# ============================================================
def remove_iqr_outliers(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

before = len(main)
for col in ['size_sqft', 'actual_rent', 'distance_km']:
    main = remove_iqr_outliers(main, col)
main.reset_index(drop=True, inplace=True)
print(f"[OUTLIERS] Removed {before - len(main)} outlier rows  →  {len(main)} rows remain")

# ============================================================
# 6. LABEL ENCODING  (saved for dashboard use later)
# ============================================================
encoders = {}
encode_cols = ['house_type', 'furnishing', 'traffic_level']

for col in encode_cols:
    le = LabelEncoder()
    main[col] = le.fit_transform(main[col])
    encoders[col] = le
    print(f"[ENCODE] {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

pickle.dump(encoders, open(ENC_PATH, 'wb'))
print(f"[ENCODE] Encoders saved → {ENC_PATH}")

# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================
print("\n[FEATURES] Building engineered features...")

# --- Safety Score (already in dataset as actual_safety_score) ---
main['safety_score'] = main['actual_safety_score']

# --- Risk Index (crime weighted more than accidents) ---
main['risk_index'] = (0.6 * main['crime_count'] + 0.4 * main['accident_count']).round(2)

# --- Traffic Score ---
traffic_num = main['traffic_level']   # already 0/1/2 after encoding
main['traffic_score'] = (traffic_num * main['congestion_index']).round(2)

# --- Overall Area Score (higher = better area) ---
main['overall_score'] = (
    main['safety_score'] - main['traffic_score'] - main['risk_index']
).round(2)

# --- Accessibility Score ---
main['accessibility_score'] = (
    main['police_station_count'] / (main['distance_km'] + 1)
).round(4)

# --- Monthly Food Cost (from food dataset, per area) ---
# Use only training-safe aggregation: area-level mean meal price
area_food = food.groupby(['city', 'area'])['avg_meal_price'].mean().reset_index()
area_food.columns = ['city', 'area', 'area_avg_meal_price']

main = main.merge(area_food, on=['city', 'area'], how='left')
main['area_avg_meal_price'].fillna(main['area_avg_meal_price'].mean(), inplace=True)

# 60 meals/month = ~2 meals/day
main['monthly_food_cost'] = (main['area_avg_meal_price'] * 60).round(2)
main.drop(columns=['area_avg_meal_price'], inplace=True)

# --- Commute Cost ---
# Assumption: ₹4/km, 26 working days, 2 trips/day
main['commute_cost'] = (main['distance_km'] * 4 * 26 * 2).round(2)

print(f"  safety_score       : from actual_safety_score")
print(f"  risk_index         : 0.6*crime + 0.4*accident")
print(f"  traffic_score      : traffic_level * congestion_index")
print(f"  overall_score      : safety - traffic - risk")
print(f"  accessibility_score: police_stations / (distance + 1)")
print(f"  monthly_food_cost  : area_avg_meal_price * 60 meals")
print(f"  commute_cost       : distance * 4 * 26 * 2")


# ============================================================
# 9. FINAL CHECK
# ============================================================
print(f"\n[FINAL] Shape       : {main.shape}")
print(f"[FINAL] Columns     : {main.columns.tolist()}")
print(f"[FINAL] Missing vals:\n{main.isnull().sum()[main.isnull().sum() > 0]}")
if main.isnull().sum().sum() == 0:
    print("  -> No missing values")

# ============================================================
# 10. SAVE
# ============================================================
os.makedirs(PROC_DIR, exist_ok=True)
main.to_excel(OUTPUT_PATH, index=False, engine='openpyxl')
print(f"\n[SAVED] Model-ready dataset → {OUTPUT_PATH}")
print("\n" + "=" * 55)
print("  PIPELINE COMPLETE")
print("=" * 55)
print("\nNext step: run ml_model1_rent_prediction.py")
print("  DATA_PATH = 'data/processed/optisuit_model_ready.xlsx'")