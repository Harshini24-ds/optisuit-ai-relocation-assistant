import os
import pickle
import warnings

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = r"C:\Users\harsh\Desktop\OptiSuit_Project\OptiSuit_Project"
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "optisuitmodel.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "models", "suitability_model.pkl")

print("=" * 60)
print("  OPTISUIT - SUITABILITY CLASSIFICATION MODEL (v3)")
print("=" * 60)

# ================= LOAD =================
df = pd.read_excel(DATA_PATH, engine="openpyxl")
print(f"\nDataset shape: {df.shape}")

# ================= STEP 1: COMPUTE TCOL FOR LABELLING ONLY =================
# TCoL is used only to create labels. It is never passed as a feature.
df["TCoL"] = df["actual_rent"] + df["monthly_food_cost"] + df["commute_cost"]

print(f"\nTCoL range  : Rs.{df['TCoL'].min():,.0f} -> Rs.{df['TCoL'].max():,.0f}")
print(
    f"Safety range: {df['safety_score'].min():.1f} -> "
    f"{df['safety_score'].max():.1f}"
)

# ================= STEP 2: CREATE SUITABILITY SCORE + LABELS =================
# Instead of assigning only the extreme corners to high/low suitability and
# pushing almost everything else into the middle class, build a continuous
# score from affordability and safety, then split that score into 3 bands.
tcol_min = df["TCoL"].min()
tcol_max = df["TCoL"].max()
safe_min = df["safety_score"].min()
safe_max = df["safety_score"].max()

df["affordability_score"] = 100 * (
    1 - (df["TCoL"] - tcol_min) / (tcol_max - tcol_min)
)
df["safety_norm_score"] = 100 * (
    (df["safety_score"] - safe_min) / (safe_max - safe_min)
)
df["suitability_score"] = (
    0.55 * df["affordability_score"] + 0.45 * df["safety_norm_score"]
)

score_33 = df["suitability_score"].quantile(0.33)
score_66 = df["suitability_score"].quantile(0.66)

print("\nThresholds (used for label creation only):")
print(f"  Suitability Score - 33rd pct: {score_33:.2f}")
print(f"  Suitability Score - 66th pct: {score_66:.2f}")


def assign_suitability(score):
    if score >= score_66:
        return 0
    if score <= score_33:
        return 2
    return 1


df["suitability_class"] = df["suitability_score"].apply(assign_suitability)
label_map = {0: "Highly Suitable", 1: "Moderately Suitable", 2: "Not Suitable"}

print("\nClass distribution (before SMOTE):")
for key, label in label_map.items():
    count = (df["suitability_class"] == key).sum()
    pct = count / len(df) * 100
    print(f"  {label:<22}: {count:>4} rows ({pct:.1f}%)")

# ================= STEP 3: FEATURES (LEAK-FREE) =================
features = [
    "size_sqft",
    "house_type",
    "furnishing",
    "actual_rent",
    "avg_meal_price",
    "accident_count",
    "crime_count",
    "police_station_count",
    "risk_index",
    "accessibility_score",
    "traffic_level",
    "congestion_index",
    "traffic_score",
    "distance_km",
    "overall_score",
]
features = [feature for feature in features if feature in df.columns]

print(f"\nFeatures used (leak-free): {len(features)}")
print(f"  {features}")
print("\nRemoved vs v1: ['TCoL', 'monthly_food_cost', 'commute_cost']")

X = df[features].copy()
y = df["suitability_class"]

# ================= STEP 4: STRATIFIED SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print(f"\nTrain: {len(X_train)} rows  |  Test: {len(X_test)} rows")
print(f"Test class distribution: {dict(y_test.value_counts().sort_index())}")

# ================= STEP 5: LIGHT SMOTE (TRAIN ONLY) =================
print("\n--- LIGHT SMOTE: Balancing Training Classes ---")
print(f"Before SMOTE: {dict(y_train.value_counts().sort_index())}")

class_counts = y_train.value_counts().sort_index()
smote_targets = {
    0: max(int(class_counts.get(0, 0)), 200),
    1: int(class_counts.get(1, 0)),
    2: max(int(class_counts.get(2, 0)), 200),
}

if any(smote_targets[key] > int(class_counts.get(key, 0)) for key in smote_targets):
    smote = SMOTE(
        sampling_strategy=smote_targets,
        random_state=42,
    )
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"After  SMOTE: {dict(pd.Series(y_train_sm).value_counts().sort_index())}")
else:
    X_train_sm, y_train_sm = X_train.copy(), y_train.copy()
    print("After  SMOTE: skipped (classes already balanced enough)")

# ================= STEP 6: BASELINE CV =================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_model = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss",
    n_estimators=80,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=10,
    reg_lambda=20,
    min_child_weight=12,
    gamma=1.0,
)

print("\n--- CROSS VALIDATION (Baseline Regularized Model) ---")
cv_scores = cross_val_score(
    baseline_model,
    X_train_sm,
    y_train_sm,
    cv=skf,
    scoring="accuracy",
)
print(f"XGBoost Baseline | CV Scores: {[round(score, 4) for score in cv_scores]}")
print(f"                 | Mean: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")

# ================= STEP 7: FINAL MODEL =================
tuned_model = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss",
    n_estimators=100,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.75,
    colsample_bytree=0.75,
    reg_alpha=8,
    reg_lambda=20,
    min_child_weight=15,
    gamma=1.5,
)

print("\n--- FINAL MODEL CONFIGURATION ---")
print("  n_estimators    : 100")
print("  max_depth       : 3")
print("  learning_rate   : 0.03")
print("  subsample       : 0.75")
print("  colsample_bytree: 0.75")
print("  reg_alpha       : 8")
print("  reg_lambda      : 20")
print("  min_child_weight: 15")
print("  gamma           : 1.5")

print("\n--- CROSS VALIDATION (Final Model) ---")
cv_tuned = cross_val_score(
    tuned_model,
    X_train_sm,
    y_train_sm,
    cv=skf,
    scoring="accuracy",
)
print(f"XGBoost Final    | CV Scores: {[round(score, 4) for score in cv_tuned]}")
print(f"                 | Mean: {cv_tuned.mean():.4f} | Std: {cv_tuned.std():.4f}")

# ================= STEP 8: FINAL TRAINING + TEST EVALUATION =================
print("\n--- TEST SET EVALUATION ---")
tuned_model.fit(X_train_sm, y_train_sm)

train_acc = accuracy_score(y_train_sm, tuned_model.predict(X_train_sm))
test_acc = accuracy_score(y_test, tuned_model.predict(X_test))
gap = train_acc - test_acc

print(f"\n{'Metric':<20} {'Value':>10}")
print("-" * 32)
print(f"{'Train Accuracy':<20} {train_acc:>10.4f}")
print(f"{'Test Accuracy':<20} {test_acc:>10.4f}")
print(f"{'Gap':<20} {gap:>10.4f}")

# ================= STEP 9: DETAILED REPORT =================
y_pred = tuned_model.predict(X_test)
print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Highly Suitable", "Moderately Suitable", "Not Suitable"],
    )
)

print("CONFUSION MATRIX")
print("Rows = Actual | Cols = Predicted")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[
        "Act: Highly Suitable",
        "Act: Moderately Suitable",
        "Act: Not Suitable",
    ],
    columns=["Pred: Highly", "Pred: Moderate", "Pred: Not Suitable"],
)
print(cm_df.to_string())

# ================= STEP 10: DIAGNOSIS =================
print("\nDIAGNOSIS")
if gap < 0.10:
    print(f"  Generalization: Good - gap is {gap:.4f} (target met)")
else:
    print(f"  Overfitting   : Yes - gap is {gap:.4f} (target < 0.10)")

if test_acc >= 0.85:
    print("  Accuracy      : Strong (above 85%)")
elif test_acc >= 0.75:
    print("  Accuracy      : Good (above 75% target met)")
elif test_acc >= 0.65:
    print("  Accuracy      : Acceptable")
else:
    print("  Accuracy      : Needs improvement")

# ================= STEP 11: FEATURE IMPORTANCE =================
feat_imp = pd.Series(tuned_model.feature_importances_, index=features)
print("\nFEATURE IMPORTANCES")
print(feat_imp.sort_values(ascending=False).to_string())

# ================= STEP 12: SAMPLE PREDICTIONS =================
sample_df = pd.DataFrame(
    {
        "Actual Class": [label_map[value] for value in y_test.values[:100]],
        "Predicted Class": [label_map[value] for value in y_pred[:100]],
        "Actual Rent": df.loc[X_test.index[:100], "actual_rent"].values,
        "Safety Score": df.loc[X_test.index[:100], "safety_score"].values,
        "TCoL (Rs.)": df.loc[X_test.index[:100], "TCoL"].round(0).values,
    }
)
print("\nSAMPLE PREDICTIONS (first 50 test rows)")
print(sample_df.to_string(index=False))

# ================= STEP 13: SAVE MODEL BUNDLE =================
model_bundle = {
    "model": tuned_model,
    "features": features,
    "label_map": label_map,
    "thresholds": {
        "score_33": score_33,
        "score_66": score_66,
    },
}
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model_bundle, model_file)

print(f"\nModel bundle saved -> {MODEL_PATH}")
print("  (includes model + features + label_map + thresholds)")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
