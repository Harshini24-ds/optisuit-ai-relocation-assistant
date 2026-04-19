import streamlit as st

st.set_page_config(
    page_title="OptiSuit",
    page_icon="🏠",
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

    .hero-card {
        background: rgba(16, 24, 39, 0.52);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 18px 40px rgba(0,0,0,0.22);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }

    .hero-sub {
        color: #c9d4df;
        font-size: 1rem;
        line-height: 1.65;
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

    .section-card {
        background: rgba(15, 23, 35, 0.52);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1.1rem 1.2rem;
        margin-top: 1rem;
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

st.title("OptiSuit")
st.caption("AI-Driven Relocation Assistant")

c1, c2 = st.columns([1.4, 1])

with c1:
    st.subheader("Find the right area before you relocate")
    st.write(
        """
        OptiSuit helps users evaluate areas using rent, food cost,
        commute cost, safety score, clustering, and suitability.
        Use the top navigation buttons to move across pages.
        """
    )

with c2:
    st.metric("Cities Covered", "2")
    st.metric("Main Modes", "3")
    st.metric("Focus", "Cost + Safety + Commute")

st.markdown("## About")
st.write(
    """
    This dashboard is built for relocation decision support in Chennai and Bengaluru.
    It helps users rank areas, compare areas, and explore food options by area.
    """
)

st.markdown("## Services")
s1, s2, s3 = st.columns(3)
with s1:
    st.write("### Ranked Areas")
    st.write("See area-wise recommendations.")
with s2:
    st.write("### Food Suggestions")
    st.write("Explore restaurants and cloud kitchens.")
with s3:
    st.write("### Comparison")
    st.write("Compare areas side by side.")
