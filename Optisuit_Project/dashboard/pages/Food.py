from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Food",
    page_icon="🍽️",
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
FOOD_PATH = BASE_DIR / "data" / "raw" / "food_dataset.xlsx"


@st.cache_data
def load_food():
    df = pd.read_excel(FOOD_PATH, engine="openpyxl")
    df["city"] = df["city"].astype(str).str.title()
    df["area"] = df["area"].astype(str).str.title()
    return df


food_df = load_food()

st.title("Food Suggestions")
st.caption("Dedicated page for restaurants and cloud kitchens")

city = st.selectbox("City", sorted(food_df["city"].unique()))
city_df = food_df[food_df["city"] == city].copy()

area = st.selectbox("Area", sorted(city_df["area"].unique()))
area_df = city_df[city_df["area"] == area].copy()

f1, f2, f3, f4 = st.columns(4)
with f1:
    food_type = st.selectbox("Food Type", ["Both", "Restaurant", "Cloud Kitchen"])
with f2:
    cuisine_options = ["All"] + sorted(area_df["cuisine"].dropna().unique().tolist())
    cuisine = st.selectbox("Cuisine", cuisine_options)
with f3:
    max_price = st.slider("Max Meal Price", 50, int(area_df["avg_meal_price"].max()), min(300, int(area_df["avg_meal_price"].max())))
with f4:
    min_rating = st.slider("Minimum Rating", 0.0, 5.0, 4.0, step=0.1)

filtered = area_df.copy()

if food_type != "Both":
    filtered = filtered[filtered["food_type"].str.lower() == food_type.lower()]

if cuisine != "All":
    filtered = filtered[filtered["cuisine"] == cuisine]

filtered = filtered[filtered["avg_meal_price"] <= max_price]
filtered = filtered[filtered["rating"] >= min_rating]

if filtered.empty:
    st.warning("No shops match the selected filters.")
else:
    filtered = filtered.sort_values(by=["rating", "avg_meal_price"], ascending=[False, True])

    c1, c2, c3 = st.columns(3)
    c1.metric("Matching Shops", len(filtered))
    c2.metric("Avg Meal Price", f"₹{filtered['avg_meal_price'].mean():.0f}")
    c3.metric("Avg Rating", f"{filtered['rating'].mean():.2f}")

    st.dataframe(
        filtered[["restaurant_name", "food_type", "cuisine", "avg_meal_price", "rating"]].rename(
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

    fig = px.scatter(
        filtered,
        x="avg_meal_price",
        y="rating",
        color="food_type",
        hover_name="restaurant_name",
        size="rating",
        title=f"Food Options in {area}",
    )
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)
