import os
import pickle
import warnings

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = r"C:\Users\harsh\Desktop\OptiSuit_Project\OptiSuit_Project"
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "optisuitmodel.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "kmeans_scaler.pkl")
OUTPUT_PATH = DATA_PATH

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Only numeric features for clustering
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
]

print("=" * 60)
print("  OPTISUIT - MODEL 3: AREA CLUSTERING")
print("=" * 60)

main = pd.read_excel(DATA_PATH, engine="openpyxl")
print(f"\nDataset shape: {main.shape}")

feature_cols = [col for col in feature_cols if col in main.columns]
X = main[feature_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Training KMeans with 3 clusters ---")
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
    max_iter=300,
)

main["area_cluster"] = kmeans.fit_predict(X_scaled)

summary_cols = [
    "actual_rent",
    "monthly_food_cost",
    "commute_cost",
    "safety_score",
    "overall_score",
]
summary_cols = [col for col in summary_cols if col in main.columns]

cluster_summary = main.groupby("area_cluster")[summary_cols].mean().round(2)

print("\nCluster summary:")
print(cluster_summary.to_string())

# Name clusters by average total cost
cluster_cost = (
    main.groupby("area_cluster")[["actual_rent", "monthly_food_cost", "commute_cost"]]
    .mean()
    .sum(axis=1)
    .sort_values()
)

sorted_clusters = cluster_cost.index.tolist()

cluster_name_map = {
    sorted_clusters[0]: "Budget-friendly",
    sorted_clusters[1]: "Balanced",
    sorted_clusters[2]: "Premium",
}

main["cluster_name"] = main["area_cluster"].map(cluster_name_map)

print("\nCluster mapping:")
for cluster_id, cluster_name in cluster_name_map.items():
    print(f"Cluster {cluster_id} -> {cluster_name}")

print("\nSample output:")
display_cols = [col for col in ["city", "area", "area_cluster", "cluster_name"] if col in main.columns]
print(main[display_cols].head(50).to_string(index=False))

with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(kmeans, model_file)

with open(SCALER_PATH, "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

main.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")

print(f"\nSaved model: {MODEL_PATH}")
print(f"Saved scaler: {SCALER_PATH}")
print(f"Updated dataset: {OUTPUT_PATH}")

print("\n" + "=" * 60)
print("  MODEL 3 COMPLETE")
print("=" * 60)
