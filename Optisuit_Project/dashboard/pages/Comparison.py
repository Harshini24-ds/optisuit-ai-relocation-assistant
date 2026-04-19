from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Comparison",
    page_icon="⚖️",
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
FOOD_PATH = BASE_DIR / "data" / "raw" / "food_dataset.xlsx"


@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    df["city"] = df["city"].astype(str).str.title()
    df["area"] = df["area"].astype(str).str.title()
    return df


@st.cache_data
def load_food():
    df = pd.read_excel(FOOD_PATH, engine="openpyxl")
    df["city"] = df["city"].astype(str).str.title()
    df["area"] = df["area"].astype(str).str.title()
    return df


def rupees(x):
    return f"₹{float(x):,.0f}"


def dashboard_suitability_label(row, monthly_budget):
    budget_ratio = row["TCoL"] / max(monthly_budget, 1)
    if budget_ratio <= 0.80:
        return "Highly Suitable"
    elif budget_ratio <= 1.20:
        return "Moderately Suitable"
    return "Not Suitable"


main_df = load_data()
food_df = load_food()

st.title("Comparison")
st.caption("Compare selected areas side by side")

city = st.selectbox("City", sorted(main_df["city"].unique()))
city_df = main_df[main_df["city"] == city].copy()

if "predicted_rent" not in city_df.columns:
    city_df["predicted_rent"] = city_df["actual_rent"]

city_df["TCoL"] = city_df["predicted_rent"] + city_df["monthly_food_cost"] + city_df["commute_cost"]

areas = sorted(city_df["area"].unique())

f1, f2, f3 = st.columns(3)
with f1:
    monthly_budget = st.slider("Monthly Budget", 5000, 50000, 20000, step=500)
with f2:
    area_1 = st.selectbox("Current Area", areas, index=0)
with f3:
    area_2 = st.selectbox("Target Area", areas, index=1 if len(areas) > 1 else 0)

row_1 = city_df[city_df["area"] == area_1].iloc[0]
row_2 = city_df[city_df["area"] == area_2].iloc[0]

tabs = st.tabs(["Area Comparison", "Food Comparison"])

with tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(area_1)
        st.metric("Predicted Rent", rupees(row_1["predicted_rent"]))
        st.metric("Food Cost", rupees(row_1["monthly_food_cost"]))
        st.metric("Commute Cost", rupees(row_1["commute_cost"]))
        st.metric("TCoL", rupees(row_1["TCoL"]))
        st.metric("Safety Score", f"{row_1['safety_score']:.2f}")
        st.write(f"Suitability: **{dashboard_suitability_label(row_1, monthly_budget)}**")
        st.write(f"Cluster: **{row_1.get('cluster_name', 'N/A')}**")

    with col2:
        st.subheader(area_2)
        st.metric("Predicted Rent", rupees(row_2["predicted_rent"]))
        st.metric("Food Cost", rupees(row_2["monthly_food_cost"]))
        st.metric("Commute Cost", rupees(row_2["commute_cost"]))
        st.metric("TCoL", rupees(row_2["TCoL"]))
        st.metric("Safety Score", f"{row_2['safety_score']:.2f}")
        st.write(f"Suitability: **{dashboard_suitability_label(row_2, monthly_budget)}**")
        st.write(f"Cluster: **{row_2.get('cluster_name', 'N/A')}**")

    comparison_df = pd.DataFrame(
        [
            {
                "Area": area_1,
                "Predicted Rent": row_1["predicted_rent"],
                "Food Cost": row_1["monthly_food_cost"],
                "Commute Cost": row_1["commute_cost"],
                "TCoL": row_1["TCoL"],
                "Safety Score": row_1["safety_score"],
            },
            {
                "Area": area_2,
                "Predicted Rent": row_2["predicted_rent"],
                "Food Cost": row_2["monthly_food_cost"],
                "Commute Cost": row_2["commute_cost"],
                "TCoL": row_2["TCoL"],
                "Safety Score": row_2["safety_score"],
            },
        ]
    )

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_bar(x=comparison_df["Area"], y=comparison_df["Predicted Rent"], name="Rent")
    fig.add_bar(x=comparison_df["Area"], y=comparison_df["Food Cost"], name="Food")
    fig.add_bar(x=comparison_df["Area"], y=comparison_df["Commute Cost"], name="Commute")
    fig.add_trace(go.Scatter(x=comparison_df["Area"], y=comparison_df["TCoL"], mode="lines+markers", name="TCoL"))
    fig.update_layout(template="plotly_dark", height=500, barmode="group", title="Cost Comparison")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    food_type_filter = st.selectbox("Food Type", ["Both", "Restaurant", "Cloud Kitchen"])

    area1_food = food_df[(food_df["city"] == city) & (food_df["area"] == area_1)].copy()
    area2_food = food_df[(food_df["city"] == city) & (food_df["area"] == area_2)].copy()

    if food_type_filter != "Both":
        area1_food = area1_food[area1_food["food_type"].str.lower() == food_type_filter.lower()]
        area2_food = area2_food[area2_food["food_type"].str.lower() == food_type_filter.lower()]

    food_summary = pd.DataFrame(
        [
            {
                "Area": area_1,
                "Total Shops": len(area1_food),
                "Avg Meal Price": round(area1_food["avg_meal_price"].mean(), 2) if not area1_food.empty else 0,
                "Avg Rating": round(area1_food["rating"].mean(), 2) if not area1_food.empty else 0,
            },
            {
                "Area": area_2,
                "Total Shops": len(area2_food),
                "Avg Meal Price": round(area2_food["avg_meal_price"].mean(), 2) if not area2_food.empty else 0,
                "Avg Rating": round(area2_food["rating"].mean(), 2) if not area2_food.empty else 0,
            },
        ]
    )

    st.dataframe(food_summary, use_container_width=True, hide_index=True)

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(f"### {area_1} Shops")
        if area1_food.empty:
            st.info("No food data available.")
        else:
            top1 = area1_food.sort_values(by=["rating", "avg_meal_price"], ascending=[False, True]).head(10)
            st.dataframe(
                top1[["restaurant_name", "food_type", "cuisine", "avg_meal_price", "rating"]].rename(
                    columns={
                        "restaurant_name": "Shop Name",
                        "food_type": "Food Type",
                        "cuisine": "Cuisine",
                        "avg_meal_price": "Avg Meal Price",
                        "rating": "Rating",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with right_col:
        st.markdown(f"### {area_2} Shops")
        if area2_food.empty:
            st.info("No food data available.")
        else:
            top2 = area2_food.sort_values(by=["rating", "avg_meal_price"], ascending=[False, True]).head(10)
            st.dataframe(
                top2[["restaurant_name", "food_type", "cuisine", "avg_meal_price", "rating"]].rename(
                    columns={
                        "restaurant_name": "Shop Name",
                        "food_type": "Food Type",
                        "cuisine": "Cuisine",
                        "avg_meal_price": "Avg Meal Price",
                        "rating": "Rating",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
