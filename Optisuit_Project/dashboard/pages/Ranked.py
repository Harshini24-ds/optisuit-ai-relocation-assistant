import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Ranked",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(87,117,144,0.20), transparent 32%),
            radial-gradient(circle at top right, rgba(122,138,165,0.16), transparent 28%),
            linear-gradient(135deg, #07111f 0%, #101b2e 48%, #1c2738 100%);
        color: #f3f6fb;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .page-shell {
        background: rgba(15, 23, 35, 0.56);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 16px 34px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 35, 0.60);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.18);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: rgba(15, 23, 35, 0.62);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: linear-gradient(90deg, #2a4a72 0%, #4f6f93 100%);
        color: white;
        font-weight: 700;
        min-height: 3rem;
    }

    hr {
        border-color: rgba(255,255,255,0.09);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    st.page_link("app.py", label="🏠 Home")
with nav2:
    st.page_link("pages/Ranked.py", label="📊 Ranked")
with nav3:
    st.page_link("pages/Food.py", label="🍽 Food")
with nav4:
    st.page_link("pages/Comparison.py", label="⚖️ Comparison")

st.divider()

BASE_DIR = Path(r"C:\Users\harsh\Desktop\OptiSuit_Project\OptiSuit_Project")
DATA_PATH = BASE_DIR / "data" / "processed" / "optisuitmodel.xlsx"
RENT_MODEL_PATH = BASE_DIR / "models" / "rent_model.pkl"

HOUSE_TYPE_MAP = {"1BHK": 0, "2BHK": 1, "PG": 2}
PRIORITY_MAP = {"Low": 1, "Medium": 2, "High": 3}

SUITABILITY_COLORS = {
    "Highly Suitable": "#4CAF50",
    "Moderately Suitable": "#D4A017",
    "Not Suitable": "#C94C4C",
}


@st.cache_data
def load_main_data():
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    df["city"] = df["city"].astype(str).str.title()
    df["area"] = df["area"].astype(str).str.title()
    df["work_location"] = df["work_location"].astype(str)
    return df


@st.cache_resource
def load_rent_bundle():
    with open(RENT_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def rupees(value):
    return f"₹{float(value):,.0f}"


def outlier_text(flag):
    return "⚠️ Unusual" if int(flag) == 1 else "Normal"


def predict_rent(area_row, rent_bundle, house_type_value):
    area_mean = rent_bundle["area_mean"]
    city_mean = rent_bundle["city_mean"]
    global_mean = rent_bundle["global_mean"]
    feature_order = rent_bundle["features"]

    feature_dict = {
        "size_sqft": area_row["size_sqft"],
        "house_type": house_type_value,
        "furnishing": area_row["furnishing"],
        "safety_score": area_row["safety_score"],
        "risk_index": area_row["risk_index"],
        "traffic_score": area_row["traffic_score"],
        "overall_score": area_row["overall_score"],
        "accessibility_score": area_row["accessibility_score"],
        "police_station_count": area_row["police_station_count"],
        "congestion_index": area_row["congestion_index"],
        "distance_km": area_row["distance_km"],
        "traffic_level": area_row["traffic_level"],
    }

    area_rent_enc = area_mean.get(str(area_row["area"]).title(), global_mean)
    city_rent_enc = city_mean.get(str(area_row["city"]).title(), global_mean)

    feature_dict["area_rent_enc"] = area_rent_enc
    feature_dict["city_rent_enc"] = city_rent_enc
    feature_dict["size_x_housetype"] = feature_dict["size_sqft"] * feature_dict["house_type"]
    feature_dict["size_x_area"] = feature_dict["size_sqft"] * feature_dict["area_rent_enc"]
    feature_dict["housetype_x_area"] = feature_dict["house_type"] * feature_dict["area_rent_enc"]
    feature_dict["safety_x_area"] = feature_dict["safety_score"] * feature_dict["area_rent_enc"]
    feature_dict["size_sqft_sq"] = feature_dict["size_sqft"] ** 2

    X = pd.DataFrame([[feature_dict[col] for col in feature_order]], columns=feature_order)
    pred_log = rent_bundle["model"].predict(X)[0]
    return float(np.expm1(pred_log))


def predict_suitability_label(row, monthly_budget, priorities):
    budget_w = PRIORITY_MAP[priorities["budget_priority"]]
    safety_w = PRIORITY_MAP[priorities["safety_priority"]]
    commute_w = PRIORITY_MAP[priorities["commute_priority"]]
    food_w = PRIORITY_MAP[priorities["food_priority"]]

    budget_ratio = row["tcol"] / max(monthly_budget, 1)

    if budget_ratio <= 0.80:
        affordability = 100
    elif budget_ratio <= 1.00:
        affordability = 85
    elif budget_ratio <= 1.20:
        affordability = 65
    elif budget_ratio <= 1.40:
        affordability = 45
    else:
        affordability = 25

    safety = max(0, min(float(row["safety_score"]), 100))
    commute_score = max(20, 100 - (float(row["commute_cost"]) / 60))
    food_score = max(20, 100 - (float(row["monthly_food_cost"]) / 120))

    total_weight = budget_w + safety_w + commute_w + food_w
    final_score = (
        affordability * budget_w
        + safety * safety_w
        + commute_score * commute_w
        + food_score * food_w
    ) / total_weight

    if final_score >= 75:
        label = "Highly Suitable"
    elif final_score >= 50:
        label = "Moderately Suitable"
    else:
        label = "Not Suitable"

    return label, round(final_score, 2)


def build_city_area_scores(city_df, rent_bundle, monthly_budget, house_type_label, eating_frequency, priorities):
    rows = []
    house_type_value = HOUSE_TYPE_MAP[house_type_label]

    grouped = city_df.groupby(["city", "area"], as_index=False).agg({
        "size_sqft": "median",
        "furnishing": "median",
        "avg_meal_price": "mean",
        "accident_count": "mean",
        "crime_count": "mean",
        "police_station_count": "mean",
        "traffic_level": "median",
        "congestion_index": "mean",
        "distance_km": "mean",
        "work_location": "first",
        "safety_score": "mean",
        "risk_index": "mean",
        "traffic_score": "mean",
        "overall_score": "mean",
        "accessibility_score": "mean",
        "area_cluster": "first",
        "cluster_name": "first",
        "outlier_flag": "max",
    })

    for _, area_row in grouped.iterrows():
        predicted_rent = predict_rent(area_row, rent_bundle, house_type_value)
        monthly_food_cost = float(area_row["avg_meal_price"]) * eating_frequency
        commute_cost = float(area_row["distance_km"]) * 4 * 26 * 2
        tcol = predicted_rent + monthly_food_cost + commute_cost

        label, suitability_score = predict_suitability_label(
            {
                "tcol": tcol,
                "safety_score": area_row["safety_score"],
                "monthly_food_cost": monthly_food_cost,
                "commute_cost": commute_cost,
            },
            monthly_budget=monthly_budget,
            priorities=priorities,
        )

        rows.append({
            "city": area_row["city"],
            "area": area_row["area"],
            "predicted_rent": round(predicted_rent, 0),
            "monthly_food_cost": round(monthly_food_cost, 0),
            "commute_cost": round(commute_cost, 0),
            "tcol": round(tcol, 0),
            "safety_score": round(float(area_row["safety_score"]), 2),
            "cluster_name": area_row.get("cluster_name", "Balanced"),
            "outlier_flag": int(area_row.get("outlier_flag", 0)),
            "suitability_label": label,
            "suitability_score": suitability_score,
            "work_location": area_row["work_location"],
        })

    result = pd.DataFrame(rows)
    result = result.sort_values(
        by=["suitability_score", "tcol", "safety_score"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


main_df = load_main_data()
rent_bundle = load_rent_bundle()

st.title("Ranked Areas")
st.caption("Explore ranked residential options")

city = st.selectbox("City", sorted(main_df["city"].dropna().unique().tolist()))
city_df = main_df[main_df["city"] == city].copy()
work_locations = sorted(city_df["work_location"].dropna().unique().tolist())

f1, f2, f3, f4 = st.columns(4)
with f1:
    monthly_budget = st.slider("Monthly Budget", 5000, 50000, 20000, step=500)
with f2:
    house_type_label = st.selectbox("House Type", ["PG", "1BHK", "2BHK"])
with f3:
    eating_frequency = st.slider("Meals / Month", 10, 90, 60)
with f4:
    selected_work_location = st.selectbox("Work Location", work_locations)

p1, p2, p3, p4 = st.columns(4)
with p1:
    budget_priority = st.selectbox("Budget Priority", ["Low", "Medium", "High"], index=2)
with p2:
    safety_priority = st.selectbox("Safety Priority", ["Low", "Medium", "High"], index=2)
with p3:
    commute_priority = st.selectbox("Commute Priority", ["Low", "Medium", "High"], index=1)
with p4:
    food_priority = st.selectbox("Food Priority", ["Low", "Medium", "High"], index=1)

run_ranking = st.button("Find Best Areas", use_container_width=True)

if run_ranking:
    ranking_df = city_df.copy()
    if selected_work_location:
        ranking_df = ranking_df[
            ranking_df["work_location"].astype(str) == str(selected_work_location)
        ].copy()

    priorities = {
        "budget_priority": budget_priority,
        "safety_priority": safety_priority,
        "commute_priority": commute_priority,
        "food_priority": food_priority,
    }

    st.session_state["ranked_df"] = build_city_area_scores(
        city_df=ranking_df,
        rent_bundle=rent_bundle,
        monthly_budget=monthly_budget,
        house_type_label=house_type_label,
        eating_frequency=eating_frequency,
        priorities=priorities,
    )

ranked_df = st.session_state.get("ranked_df")

if ranked_df is None or ranked_df.empty:
    st.info("Choose filters and click 'Find Best Areas' to see recommendations.")
else:
    selected_area = st.selectbox("Select an area to view details", ranked_df["area"].tolist())

    show_df = ranked_df[
        ["rank", "area", "predicted_rent", "monthly_food_cost", "commute_cost", "tcol", "safety_score", "cluster_name", "suitability_label", "outlier_flag"]
    ].rename(columns={
        "rank": "Rank",
        "area": "Area",
        "predicted_rent": "Predicted Rent",
        "monthly_food_cost": "Food Cost",
        "commute_cost": "Commute Cost",
        "tcol": "Total Cost",
        "safety_score": "Safety Score",
        "cluster_name": "Cluster",
        "suitability_label": "Suitability",
        "outlier_flag": "Outlier",
    })
    show_df["Outlier"] = show_df["Outlier"].apply(outlier_text)

    st.dataframe(show_df, use_container_width=True, hide_index=True)

    selected_row = ranked_df[ranked_df["area"] == selected_area].iloc[0]

    st.markdown("## Particular Section")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Area", selected_row["area"])
    c2.metric("Predicted Rent", rupees(selected_row["predicted_rent"]))
    c3.metric("TCoL", rupees(selected_row["tcol"]))
    c4.metric("Safety Score", f"{selected_row['safety_score']:.2f}")

    st.write(f"Suitability: **{selected_row['suitability_label']}**")
    st.write(f"Cluster: **{selected_row['cluster_name']}**")
    st.write(f"Outlier: **{outlier_text(selected_row['outlier_flag'])}**")

    fig = px.bar(
        ranked_df,
        x="tcol",
        y="area",
        orientation="h",
        color="suitability_label",
        color_discrete_map=SUITABILITY_COLORS,
        title="Total Cost of Living by Area",
    )
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)
