# ============================================
# OPTISUIT - DATA PREPROCESSING (FINAL FIXED)
# ============================================

import os
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = r'C:\Users\harsh\Desktop\OptiSuit_Project\Optisuit_Project'

MAIN_INPUT = os.path.join(BASE_DIR, 'data', 'raw', 'main_dataset.xlsx')
FOOD_INPUT = os.path.join(BASE_DIR, 'data', 'raw', 'food_dataset.xlsx')

MAIN_OUTPUT = os.path.join(BASE_DIR, 'data', 'processed',
                          'optisuit_preprocessed_maindataset.xlsx')

FOOD_OUTPUT = os.path.join(BASE_DIR, 'data', 'processed',
                          'optisuit_preprocessed_fooddataset.xlsx')

ENC_PATH = os.path.join(BASE_DIR, 'data', 'processed',
                        'label_encoders.pkl')

print("🚀 PREPROCESSING (FINAL)")

# ================= LOAD =================
df = pd.read_excel(MAIN_INPUT, engine='openpyxl')
food = pd.read_excel(FOOD_INPUT, engine='openpyxl')

# ================= MAIN =================

# Missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode().iloc[0], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fix numeric
numeric_cols = [
    'size_sqft','avg_meal_price','accident_count',
    'crime_count','police_station_count',
    'congestion_index','distance_km',
    'actual_rent','actual_safety_score'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Outliers
def remove_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[col] >= Q1 - 1.5*IQR) & (data[col] <= Q3 + 1.5*IQR)]

for col in ['size_sqft','actual_rent','distance_km']:
    df = remove_outliers(df, col)

# ✅ CLEAN STRINGS (IMPORTANT)
df['city'] = df['city'].astype(str).str.strip().str.lower()
df['area'] = df['area'].astype(str).str.strip().str.lower()

# Encode ONLY non-location
from sklearn.preprocessing import LabelEncoder
encoders = {}

cat_cols = ['house_type','furnishing','traffic_level']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

pickle.dump(encoders, open(ENC_PATH, 'wb'))

# ================= FOOD =================

for col in food.select_dtypes(include=np.number).columns:
    food[col].fillna(food[col].median(), inplace=True)

for col in food.select_dtypes(include='object').columns:
    food[col].fillna(food[col].mode().iloc[0], inplace=True)

food.drop_duplicates(inplace=True)

food['avg_meal_price'] = pd.to_numeric(food['avg_meal_price'], errors='coerce')
food = food[(food['avg_meal_price'] > 20) & (food['avg_meal_price'] < 2000)]

# Clean keys for merge
food['city'] = food['city'].astype(str).str.strip().str.lower()
food['area'] = food['area'].astype(str).str.strip().str.lower()

# ================= SAVE =================
df.to_excel(MAIN_OUTPUT, index=False)
food.to_excel(FOOD_OUTPUT, index=False)

print("✅ PREPROCESSING DONE")