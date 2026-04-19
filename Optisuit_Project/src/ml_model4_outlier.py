import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = r"C:\Users\harsh\Desktop\OptiSuit_Project\OptiSuit_Project"
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "optisuitmodel.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "models", "dbscan_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "dbscan_scaler.pkl")
OUTPUT_PATH = DATA_PATH

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Use only numeric behavioral / area-quality features
feature_cols = [
    "house_type",
    "size_sqft",
    "furnishing",
    "avg_meal_price",
    "accident_count",
    "crime_count",
    "police_station_count",
    "traffic_level",
    "congestion_index",
    "distance_km",
    "risk_index",
    "traffic_score",
    "overall_score",
    "accessibility_score",
]

print("=" * 60)
print("  OPTISUIT - MODEL 4: OUTLIER DETECTION")
print("=" * 60)

# ================= LOAD DATA =================
main = pd.read_excel(DATA_PATH, engine="openpyxl")
print(f"\nDataset shape: {main.shape}")

feature_cols = [col for col in feature_cols if col in main.columns]
X = main[feature_cols].copy()

# ================= SCALE FEATURES =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= FIND EPS =================
print("\n--- Finding Optimal EPS ---")
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)

# 5th nearest-neighbor distance
distances = np.sort(distances[:, 4])
eps = round(float(np.percentile(distances, 90)), 2)
print(f"Chosen EPS: {eps}")

# ================= TRAIN DBSCAN =================
print("\n--- Training DBSCAN ---")
dbscan = DBSCAN(
    eps=eps,
    min_samples=5,
    metric="euclidean",
)

labels = dbscan.fit_predict(X_scaled)

main["outlier_flag"] = (labels == -1).astype(int)

n_outliers = int((main["outlier_flag"] == 1).sum())
n_normal = int((main["outlier_flag"] == 0).sum())
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f"Clusters found: {n_clusters}")
print(f"Normal rows   : {n_normal}")
print(f"Outlier rows  : {n_outliers}")

# ================= OUTLIER ANALYSIS =================
display_cols = [
    "city",
    "area",
    "actual_rent",
    "monthly_food_cost",
    "commute_cost",
    "safety_score",
    "overall_score",
]
display_cols = [col for col in display_cols if col in main.columns]

outliers = main[main["outlier_flag"] == 1][display_cols]

if len(outliers) > 0:
    print("\nSample outlier rows:")
    print(outliers.head(20).to_string(index=False))
else:
    print("\nNo outliers found.")

# ================= SAVE =================
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(dbscan, model_file)

with open(SCALER_PATH, "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

main.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")

print(f"\nSaved model: {MODEL_PATH}")
print(f"Saved scaler: {SCALER_PATH}")
print(f"Updated dataset: {OUTPUT_PATH}")

print("\n" + "=" * 60)
print("  MODEL 4 COMPLETE")
print("=" * 60)
