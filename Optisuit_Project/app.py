import pickle
from functools import lru_cache
from pathlib import Path
import os
from fastapi import params
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for
from src.nlp.query_parser import parse_search_query
from src.translation.language_detector import detect_language
from src.translation.translator import (
    translate_dynamic_value,
    translate_from_working_language,
    translate_to_working_language,
)
from src.translation.ui_translations import get_ui_bundle

app = Flask(__name__)

BASE_DIR = Path(r"C:\Users\harsh\Desktop\OptiSuit_Project\Optisuit_Project")
DATA_PATH = BASE_DIR / "data" / "processed" / "optisuitmodel.xlsx"
FOOD_PATH = BASE_DIR / "data" / "raw" / "food_dataset.xlsx"
RENT_MODEL_PATH = BASE_DIR / "models" / "rent_model.pkl"

HOUSE_TYPE_MAP = {"1BHK": 0, "2BHK": 1, "PG": 2}
PRIORITY_MAP = {"Low": 1, "Medium": 2, "High": 3}
VALUE_TRANSLATIONS = {
    "ta": {
        "Low": "குறைவு",
        "Medium": "மிதமான",
        "High": "அதிகம்",
        "PG": "பிஜி",
        "1BHK": "1BHK",
        "2BHK": "2BHK",
        "Both": "இரண்டும்",
        "Restaurant": "உணவகம்",
        "Cloud Kitchen": "கிளவுட் கிச்சன்",
        "All": "அனைத்தும்",
        "Bengaluru": "பெங்களூரு",
        "Chennai": "சென்னை",
        "Adyar": "அடையார்",
        "Ambattur": "அம்பத்தூர்",
        "Anna Nagar": "அண்ணா நகர்",
        "Bellandur": "பெல்லந்தூர்",
        "Btm Layout": "பிடிஎம் லேஅவுட்",
        "Electronic City": "எலெக்ட்ரானிக் சிட்டி",
        "Guindy": "கிண்டி",
        "Hebbal": "ஹெப்பால்",
        "Hsr Layout": "எச்எஸ்ஆர் லேஅவுட்",
        "Indiranagar": "இந்திராநகர்",
        "Jayanagar": "ஜெயநகர்",
        "Jp Nagar": "ஜேபி நகர்",
        "Koramangala": "கோரமங்களா",
        "Marathahalli": "மரத்தஹள்ளி",
        "Mylapore": "மயிலாப்பூர்",
        "Nungambakkam": "நுங்கம்பாக்கம்",
        "Omr": "ஓஎம்ஆர்",
        "Perambur": "பெரம்பூர்",
        "Perungudi": "பெருங்குடி",
        "Porur": "போரூர்",
        "Rajajinagar": "ராஜாஜிநகர்",
        "Rt Nagar": "ஆர்டி நகர்",
        "Sholinganallur": "சோழிங்கநல்லூர்",
        "T Nagar": "டி நகர்",
        "Tambaram": "தாம்பரம்",
        "Vadapalani": "வடபழனி",
        "Velachery": "வேளச்சேரி",
        "Whitefield": "வைட்ஃபீல்ட்",
        "Yelahanka": "யெலஹங்கா",
        "Yeshwanthpur": "யஷ்வந்த்பூர்",
        "Manyata Tech Park": "மன்யாட்டா டெக் பார்க்",
        "Omr It Corridor": "ஓஎம்ஆர் ஐடி காரிடார்",
        "Taramani": "தரமணி",
        "Tech Park Jayanagar": "டெக் பார்க் ஜெயநகர்",
        "Tech Park Jp Nagar": "டெக் பார்க் ஜேபி நகர்",
        "Tech Park Mylapore": "டெக் பார்க் மயிலாப்பூர்",
        "Tech Park Nungambakkam": "டெக் பார்க் நுங்கம்பாக்கம்",
        "Tech Park Perambur": "டெக் பார்க் பெரம்பூர்",
        "Tech Park Yeshwanthpur": "டெக் பார்க் யஷ்வந்த்பூர்",
        "Whitefield Tech Park": "வைட்ஃபீல்ட் டெக் பார்க்",
        "Balanced": "சமநிலை",
        "Budget-Friendly": "பட்ஜெட்டுக்கு ஏற்றது",
        "Premium": "பிரீமியம்",
        "American": "அமெரிக்கன்",
        "Andhra": "ஆந்திரா",
        "Arabian": "அரேபியன்",
        "Asian": "ஆசியன்",
        "Bakery": "பேக்கரி",
        "Beverages": "பானங்கள்",
        "Biryani": "பிரியாணி",
        "Burgers": "பர்கர்கள்",
        "Chaat": "சாட்",
        "Chettinad": "செட்டிநாடு",
        "Chinese": "சைனீஸ்",
        "Combo": "காம்போ",
        "Continental": "காண்டினென்டல்",
        "Desserts": "இனிப்புகள்",
        "European": "ஐரோப்பிய",
        "Fast Food": "ஃபாஸ்ட் ஃபுட்",
        "French": "பிரெஞ்சு",
        "Healthy Food": "ஆரோக்கிய உணவு",
        "Home Food": "வீட்டு உணவு",
        "Ice Cream": "ஐஸ் கிரீம்",
        "Indian": "இந்திய",
        "Italian": "இத்தாலியன்",
        "Japanese": "ஜப்பானிய",
        "Juices": "ஜூஸ்கள்",
        "Kebabs": "கபாப்",
        "Kerala": "கேரளா",
        "Mexican": "மெக்ஸிகன்",
        "North Indian": "வட இந்திய",
        "Oriental": "ஓரியண்டல்",
        "Paan": "பான்",
        "Pan-Asian": "பான்-ஆசியன்",
        "Pastas": "பாஸ்டா",
        "Pizzas": "பீட்சா",
        "Punjabi": "பஞ்சாபி",
        "Rajasthani": "ராஜஸ்தானி",
        "Seafood": "கடல் உணவு",
        "Snacks": "ஸ்நாக்ஸ்",
        "South Indian": "தென் இந்திய",
        "Street Food": "தெரு உணவு",
        "Sweets": "இனிப்புகள்",
        "Tandoor": "தந்தூர்",
        "Thalis": "தாளி",
        "Tibetan": "திபெத்திய",
        "Waffle": "வாஃபிள்",
    },
    "kn": {
        "Low": "ಕಡಿಮೆ",
        "Medium": "ಮಧ್ಯಮ",
        "High": "ಹೆಚ್ಚು",
        "PG": "ಪಿಜಿ",
        "Both": "ಎರಡೂ",
        "Restaurant": "ರೆಸ್ಟೋರೆಂಟ್",
        "Cloud Kitchen": "ಕ್ಲೌಡ್ ಕಿಚನ್",
        "All": "ಎಲ್ಲಾ",
        "Bengaluru": "ಬೆಂಗಳೂರು",
        "Chennai": "ಚೆನ್ನೈ",
        "Balanced": "ಸಮತೋಲನ",
        "Budget-Friendly": "ಬಜೆಟ್ ಸ್ನೇಹಿ",
        "Premium": "ಪ್ರೀಮಿಯಂ",
    },
    "hi": {
        "Low": "कम",
        "Medium": "मध्यम",
        "High": "उच्च",
        "PG": "पीजी",
        "Both": "दोनों",
        "Restaurant": "रेस्तरां",
        "Cloud Kitchen": "क्लाउड किचन",
        "All": "सभी",
        "Bengaluru": "बेंगलुरु",
        "Chennai": "चेन्नई",
        "Balanced": "संतुलित",
        "Budget-Friendly": "बजट अनुकूल",
        "Premium": "प्रीमियम",
    },
}


def load_main_data():
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    df["city"] = df["city"].astype(str).str.title()
    df["area"] = df["area"].astype(str).str.title()
    df["work_location"] = df["work_location"].astype(str).str.title()
    return df


def load_food_data():
    df = pd.read_excel(FOOD_PATH, engine="openpyxl")
    df["city"] = df["city"].astype(str).str.title()
    df["area"] = df["area"].astype(str).str.title()
    return df


def load_rent_bundle():
    with open(RENT_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def summarize_area_data(df):
    return df.groupby(["city", "area", "work_location"], as_index=False).agg({
        "size_sqft": "median",
        "furnishing": "median",
        "avg_meal_price": "mean",
        "accident_count": "mean",
        "crime_count": "mean",
        "police_station_count": "mean",
        "traffic_level": "median",
        "congestion_index": "mean",
        "distance_km": "mean",
        "safety_score": "mean",
        "risk_index": "mean",
        "traffic_score": "mean",
        "overall_score": "mean",
        "accessibility_score": "mean",
        "area_cluster": "first",
        "cluster_name": "first",
        "outlier_flag": "max",
        "actual_rent": "mean",
        "predicted_rent": "mean",
    })


MAIN_DF = load_main_data()
FOOD_DF = load_food_data()
RENT_BUNDLE = load_rent_bundle()
AREA_SUMMARY_DF = summarize_area_data(MAIN_DF)
AREA_COORDS = {
    # Chennai
    "Adyar": [13.0067, 80.2574],
    "Anna Nagar": [13.0850, 80.2101],
    "Guindy": [13.0100, 80.2200],
    "Mylapore": [13.0330, 80.2690],
    "Nungambakkam": [13.0604, 80.2420],
    "Omr": [12.9400, 80.2450],
    "Perambur": [13.1180, 80.2330],
    "Perungudi": [12.9647, 80.2486],
    "Porur": [13.0350, 80.1580],
    "T Nagar": [13.0418, 80.2341],
    "Tambaram": [12.9249, 80.1000],
    "Vadapalani": [13.0500, 80.2121],
    "Velachery": [12.9791, 80.2209],
    "Sholinganallur": [12.8996, 80.2279],

    # Bengaluru
    "Bellandur": [12.9250, 77.6760],
    "Btm Layout": [12.9166, 77.6101],
    "Electronic City": [12.8456, 77.6603],
    "Hebbal": [13.0358, 77.5970],
    "Hsr Layout": [12.9116, 77.6474],
    "Indiranagar": [12.9719, 77.6412],
    "Jayanagar": [12.9250, 77.5938],
    "Jp Nagar": [12.9077, 77.5850],
    "Koramangala": [12.9352, 77.6245],
    "Marathahalli": [12.9591, 77.6974],
    "Rajajinagar": [12.9910, 77.5550],
    "Rt Nagar": [13.0244, 77.5946],
    "Whitefield": [12.9698, 77.7500],
    "Yelahanka": [13.1007, 77.5963],
    "Yeshwanthpur": [13.0285, 77.5400],
}

CACHE_FILE = "travel_cache.xlsx"

def load_cache():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame(columns=["origin", "destination", "car_time"])

def save_cache(df):
    df.to_csv(CACHE_FILE, index=False)

def get_coords(place):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1}
        headers = {"User-Agent": "OptiSuit"}

        res = requests.get(url, params=params, headers=headers, timeout=5)
        data = res.json()

        if data:
            return float(data[0]["lon"]), float(data[0]["lat"])
    except:
        pass
    return None

def get_osrm_time(origin, destination):
    try:
        start = get_coords(origin)
        end = get_coords(destination)

        if not start or not end:
            return None

        url = f"http://router.project-osrm.org/route/v1/driving/{start[0]},{start[1]};{end[0]},{end[1]}?overview=false"
        res = requests.get(url, timeout=5)
        data = res.json()

        return round(data["routes"][0]["duration"] / 60)
    except:
        return None

def get_hybrid_travel_times(area, work_location, distance_km):
    cache = load_cache()

    match = cache[
        (cache["origin"] == area) &
        (cache["destination"] == work_location)
    ]

    if not match.empty:
        car_time = int(match.iloc[0]["car_time"])
    else:
        car_time = get_osrm_time(area, work_location)

        if car_time is None:
            car_time = round(distance_km * 3)

        new_row = pd.DataFrame([{
            "origin": area,
            "destination": work_location,
            "car_time": car_time
        }])

        cache = pd.concat([cache, new_row], ignore_index=True)
        save_cache(cache)

    bike_time = round(car_time * 0.8)
    bus_time = round(car_time * 1.6)
    train_time = round(car_time * 0.7)
    metro_time = round(car_time * 0.75)

    return car_time, bike_time, bus_time, train_time, metro_time


def rupees(value):
    return f"Rs.{float(value):,.0f}"


def pick_valid_choice(selected_value, options, fallback_index=0):
    if not options:
        return ""
    if selected_value in options:
        return selected_value
    safe_index = min(max(fallback_index, 0), len(options) - 1)
    return options[safe_index]


def style_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f6fb", size=11),
        margin=dict(l=30, r=20, t=45, b=30),
        height=320,

        colorway=[
            "#60A5FA",  # blue
            "#A78BFA",  # violet
            "#34D399",  # green
            "#22D3EE",  # cyan
            "#C084FC",  # purple
            "#93C5FD",  # light blue
            "#6EE7B7",  # mint
            "#818CF8"   # indigo
        ]
    )
    return fig



def local_translate_value(text, language_code):
    if not text:
        return ""
    return VALUE_TRANSLATIONS.get(language_code, {}).get(str(text).strip(), "")


def safe_translate(text, language_code):
    if not text or language_code == "en":
        return str(text or "")
    local_value = local_translate_value(text, language_code)
    if local_value:
        return local_value
    try:
        return translate_dynamic_value(str(text), language_code)
    except Exception:
        return str(text)


def build_display_options(values, language_code):
    return [
        {
            "original": value,
            "display": safe_translate(value, language_code),
        }
        for value in values
    ]


def translate_status(label, ui_text):
    mapping = {
        "Highly Suitable": ui_text.get("highly_suitable", "Highly Suitable"),
        "Moderately Suitable": ui_text.get("moderately_suitable", "Moderately Suitable"),
        "Not Suitable": ui_text.get("not_suitable", "Not Suitable"),
        "Unusual": ui_text.get("unusual", "Unusual"),
        "Normal": ui_text.get("normal", "Normal"),
    }
    return mapping.get(label, label)


def get_language_context(params):
    search_query = str(params.get("search_query", "")).strip()
    selected_language = str(params.get("lang", "")).strip()

    if selected_language:
        current_language = selected_language
    else:
        current_language = detect_language(search_query) if search_query else "en"

    try:
        working_query = translate_to_working_language(search_query, current_language)
    except Exception:
        working_query = search_query

    parsed_query = parse_search_query(working_query) if working_query else {}
    ui_text = get_ui_bundle(current_language)

    return {
        "search_query": search_query,
        "current_language": current_language,
        "working_query": working_query,
        "parsed_query": parsed_query,
        "ui_text": ui_text,
    }


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
    budget_w  = PRIORITY_MAP[priorities["budget_priority"]]
    safety_w  = PRIORITY_MAP[priorities["safety_priority"]]
    commute_w = PRIORITY_MAP[priorities["commute_priority"]]
    food_w    = PRIORITY_MAP[priorities["food_priority"]]

    # Budget Score
    budget_ratio = row["tcol"] / max(monthly_budget, 1)

    if budget_ratio <= 0.8:
        affordability = 100
    elif budget_ratio <= 1.0:
        affordability = 85
    elif budget_ratio <= 1.2:
        affordability = 65
    elif budget_ratio <= 1.4:
        affordability = 45
    else:
        affordability = 20

    # Safety
    safety = max(0, min(float(row["safety_score"]), 100))

    # Better Penalty Scores
    commute_score = max(0, 100 - (float(row["commute_cost"]) / 40))
    food_score    = max(0, 100 - (float(row["monthly_food_cost"]) / 80))

    total_weight = budget_w + safety_w + commute_w + food_w

    final_score = (
        affordability * budget_w +
        safety * safety_w +
        commute_score * commute_w +
        food_score * food_w
    ) / total_weight

    if final_score >= 75:
        label = "Highly Suitable"
    elif final_score >= 50:
        label = "Moderately Suitable"
    else:
        label = "Not Suitable"

    return label, round(final_score, 2)

@lru_cache(maxsize=256)
def build_city_area_scores_cached(
    city,
    work_location,
    monthly_budget,
    house_type_label,
    eating_frequency,
    budget_priority,
    safety_priority,
    commute_priority,
    food_priority,
):
    priorities = {
        "budget_priority": budget_priority,
        "safety_priority": safety_priority,
        "commute_priority": commute_priority,
        "food_priority": food_priority,
    }
    city_df = AREA_SUMMARY_DF[AREA_SUMMARY_DF["city"] == city].copy()
    if work_location:
        city_df = city_df[city_df["work_location"].astype(str) == str(work_location)].copy()
    return build_city_area_scores(city_df, monthly_budget, house_type_label, eating_frequency, priorities)


def build_city_area_scores(city_df, monthly_budget, house_type_label, eating_frequency, priorities):
    rows = []
    house_type_value = HOUSE_TYPE_MAP[house_type_label]

    for _, area_row in city_df.iterrows():
        predicted_rent = predict_rent(area_row, RENT_BUNDLE, house_type_value)

        monthly_food_cost = float(area_row["avg_meal_price"]) * eating_frequency
        commute_cost = float(area_row["distance_km"]) * 4 * 26 * 2
        tcol = predicted_rent + monthly_food_cost + commute_cost

        if priorities["budget_priority"] == "High" and tcol > monthly_budget * 1.10:
            continue

        distance = float(area_row["distance_km"])

        car_time, bike_time, bus_time, train_time, metro_time = get_hybrid_travel_times(
            area_row["area"],
            area_row["work_location"],
            distance
        )

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

            "predicted_rent": rupees(round(predicted_rent, 0)),
            "predicted_rent_value": round(predicted_rent, 0),

            "monthly_food_cost": rupees(round(monthly_food_cost, 0)),
            "monthly_food_cost_value": round(monthly_food_cost, 0),

            "commute_cost": rupees(round(commute_cost, 0)),
            "commute_cost_value": round(commute_cost, 0),

            "tcol": rupees(round(tcol, 0)),
            "tcol_value": round(tcol, 0),

            "car_time": f"{car_time} mins",
            "bike_time": f"{bike_time} mins",
            "bus_time": f"{bus_time} mins",
            "train_time": f"{train_time} mins",
            "metro_time": f"{metro_time} mins",

            "safety_score": round(float(area_row["safety_score"]), 2),
            "cluster_name": area_row.get("cluster_name", "Balanced"),
            "outlier_flag": int(area_row.get("outlier_flag", 0)),
            "outlier_text": "Unusual" if int(area_row.get("outlier_flag", 0)) == 1 else "Normal",

            "suitability_label": label,
            "suitability_score": suitability_score,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return pd.DataFrame(columns=[
            "area",
            "cluster_name",
            "suitability_label",
            "outlier_text",
            "safety_score",
            "tcol_value",
            "predicted_rent_value",
            "suitability_score"
        ])


    result = result.sort_values(
        by=["suitability_score", "tcol_value", "safety_score"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    result["rank"] = range(1, len(result) + 1)
    return result


@app.route("/")
def home():
    params = request.args
    language_context = get_language_context(params)

    ui_text = language_context["ui_text"]
    current_language = language_context["current_language"]
    search_query = language_context["search_query"]
    parsed = language_context["parsed_query"]

    # =====================================
    # SMART SEARCH REDIRECT
    # =====================================
    if search_query and parsed:
        intent = parsed.get("intent", "ranked_search")

        if intent == "comparison":
            return redirect(
                url_for(
                    "comparison",
                    search_query=search_query,
                    lang=current_language
                )
            )

        elif intent == "food_search":
            return redirect(
                url_for(
                    "food",
                    search_query=search_query,
                    lang=current_language
                )
            )

        else:
            return redirect(
                url_for(
                    "ranked",
                    search_query=search_query,
                    lang=current_language
                )
            )

    # =====================================
    # EXISTING HOME PAGE CHARTS
    # =====================================
    city_summary = AREA_SUMMARY_DF.groupby("city", as_index=False).agg({
        "predicted_rent": "mean",
        "actual_rent": "mean",
        "avg_meal_price": "mean",
        "distance_km": "mean",
        "safety_score": "mean",
        "area": pd.Series.nunique,
    }).rename(columns={"area": "area_count"})

    city_summary["rent_basis"] = city_summary["predicted_rent"].fillna(city_summary["actual_rent"])
    rent_max = max(city_summary["rent_basis"].max(), 1)
    food_max = max(city_summary["avg_meal_price"].max(), 1)
    distance_max = max(city_summary["distance_km"].max(), 1)

    profile_rows = []
    radar_axes = ["Affordability", "Food Value", "Commute Ease", "Safety"]

    for _, row in city_summary.iterrows():
        profile_rows.append({
            "city": safe_translate(row["city"], current_language),
            "Affordability": round(100 - ((row["rent_basis"] / rent_max) * 100), 1),
            "Food Value": round(100 - ((row["avg_meal_price"] / food_max) * 100), 1),
            "Commute Ease": round(100 - ((row["distance_km"] / distance_max) * 100), 1),
            "Safety": round(float(row["safety_score"]), 1),
        })

    radar_fig = go.Figure()

    for row in profile_rows:
        radar_fig.add_trace(go.Scatterpolar(
            r=[row[axis] for axis in radar_axes] + [row[radar_axes[0]]],
            theta=radar_axes + [radar_axes[0]],
            fill="toself",
            name=row["city"],
        ))

    radar_fig.update_layout(
        title=ui_text["city_living_profile"],
        polar=dict(radialaxis=dict(visible=True, range=[0, 100]))
    )
    style_plotly(radar_fig)

    funnel_fig = go.Figure()
    stages = ["All Areas", "Budget Fit", "Safe Options", "Shortlisted"]
    translated_stages = [safe_translate(stage, current_language) for stage in stages]

    for _, row in city_summary.iterrows():
        area_count = int(row["area_count"])
        budget_fit = round(area_count * 0.75)
        safe_options = round(budget_fit * 0.68)
        shortlisted = round(safe_options * 0.58)

        funnel_fig.add_trace(go.Funnel(
            name=safe_translate(row["city"], current_language),
            y=translated_stages,
            x=[area_count, budget_fit, safe_options, shortlisted],
        ))

    funnel_fig.update_layout(title=ui_text["relocation_decision_funnel"])
    style_plotly(funnel_fig)

    return render_template(
        "index.html",
        active_page="home",
        city_profile_chart=radar_fig.to_json(),
        city_funnel_chart=funnel_fig.to_json(),
        ui_text=ui_text,
        current_language=current_language,
        search_query=search_query,
        parsed_query=parsed,
    )

def get_destination_options(city, user_type):
    options = {
        "Chennai": {
            "student": ["Guindy", "Tambaram", "Adyar", "Anna Nagar"],
            "it_professional": ["Omr", "Guindy", "Tidel Park", "Perungudi"],
            "family": ["Anna Nagar", "Velachery", "Adyar", "Tambaram"],
            "senior": ["Mylapore", "Adyar", "Anna Nagar"],
            "bachelor": ["Perungudi", "Tambaram", "Velachery"],
            "luxury": ["Nungambakkam", "Adyar", "OMR"],
        },
        "Bengaluru": {
            "student": ["Jayanagar", "Yelahanka", "Whitefield"],
            "it_professional": ["Whitefield", "Electronic City", "Koramangala"],
            "family": ["HSR Layout", "Hebbal", "Jayanagar"],
            "senior": ["Rajajinagar", "Jayanagar"],
            "bachelor": ["BTM Layout", "Marathahalli"],
            "luxury": ["Indiranagar", "Koramangala", "Whitefield"],
        }
    }

    city_map = options.get(city, {})
    values = city_map.get(user_type, [])

    if not values:
        values = sorted(
            AREA_SUMMARY_DF[
                AREA_SUMMARY_DF["city"] == city
            ]["work_location"].dropna().unique().tolist()
        )

    return [{"original": x, "display": x} for x in values]

def map_unknown_place(custom_place, city):
    text = custom_place.lower().strip()

    rules = {
        "Chennai": {
            "sathyabama": "Omr",
            "airport": "Guindy",
            "central": "Perambur",
            "apollo": "Adyar",
            "tidel": "Tidel Park",
            "guindy college": "Guindy",
        },
        "Bengaluru": {
            "airport": "Yelahanka",
            "infosys": "Electronic City",
            "majestic": "Rajajinagar",
            "mg road": "Indiranagar",
            "manyata": "Hebbal",
        }
    }

    city_rules = rules.get(city, {})

    for key, area in city_rules.items():
        if key in text:
            return area

    return ""

import math

def nearest_area_from_click(lat, lon):
    nearest = None
    min_dist = float("inf")

    for area, coords in AREA_COORDS.items():
        a_lat, a_lon = coords
        dist = math.sqrt((lat - a_lat)**2 + (lon - a_lon)**2)

        if dist < min_dist:
            min_dist = dist
            nearest = area

    return nearest

@app.route("/ranked", methods=["GET", "POST"])
def ranked():
    params = request.args if request.method == "GET" else request.form
    user_type = params.get("user_type", "student")
    custom_place = params.get("custom_place", "").strip()
    map_lat = params.get("map_lat")
    map_lon = params.get("map_lon")
    language_context = get_language_context(params)

    ui_text = language_context["ui_text"]
    current_language = language_context["current_language"]
    parsed = language_context["parsed_query"]
    if parsed:
        user_type = parsed.get("user_type", user_type)

    cities = sorted(MAIN_DF["city"].dropna().unique().tolist())
    house_types = ["PG", "1BHK", "2BHK"]
    priorities = ["Low", "Medium", "High"]

    selected_city = pick_valid_choice(params.get("city"), cities)
    monthly_budget = int(params.get("monthly_budget", 20000))
    house_type = pick_valid_choice(params.get("house_type"), house_types)
    eating_frequency = int(params.get("eating_frequency", 60))

    budget_priority = pick_valid_choice(params.get("budget_priority"), priorities, 1)
    safety_priority = pick_valid_choice(params.get("safety_priority"), priorities, 1)
    commute_priority = pick_valid_choice(params.get("commute_priority"), priorities, 1)
    food_priority = pick_valid_choice(params.get("food_priority"), priorities, 1)

    if parsed:
        selected_city = pick_valid_choice(parsed.get("city", selected_city), cities)
        house_type = pick_valid_choice(parsed.get("house_type", house_type), house_types)

        if parsed.get("monthly_budget"):
            monthly_budget = parsed["monthly_budget"]

        budget_priority = pick_valid_choice(parsed.get("budget_priority", budget_priority), priorities, 1)
        safety_priority = pick_valid_choice(parsed.get("safety_priority", safety_priority), priorities, 1)
        commute_priority = pick_valid_choice(parsed.get("commute_priority", commute_priority), priorities, 1)
        food_priority = pick_valid_choice(parsed.get("food_priority", food_priority), priorities, 1)

    work_location_options = get_destination_options(selected_city, user_type)
    work_locations = [item["original"] for item in work_location_options]

    selected_work_location = pick_valid_choice(
        parsed.get("work_location", params.get("work_location")),
        work_locations
    )

    show_map_suggestions = False


    if map_lat and map_lon:
        try:
            clicked_area = nearest_area_from_click(float(map_lat), float(map_lon))
            if clicked_area:
                selected_work_location = clicked_area
                show_map_suggestions = True
        except:
            pass


    if custom_place:
        mapped_area = map_unknown_place(custom_place, selected_city)
        if mapped_area:
            selected_work_location = mapped_area
            show_map_suggestions = True

    ranked_rows = []
    map_rows = []
    best_area = None
    tcost_chart = None
    scatter_chart = None
    suitability_chart = None
    cluster_chart = None

    if selected_city:
        ranked_df = build_city_area_scores_cached(
            city=selected_city,
            work_location=selected_work_location,
            monthly_budget=monthly_budget,
            house_type_label=house_type,
            eating_frequency=eating_frequency,
            budget_priority=budget_priority,
            safety_priority=safety_priority,
            commute_priority=commute_priority,
            food_priority=food_priority,
        )

        if not ranked_df.empty:
            ranked_df["area_display"] = ranked_df["area"].apply(
                lambda v: safe_translate(v, current_language)
            )
            ranked_df["cluster_display"] = ranked_df["cluster_name"].apply(
                lambda v: safe_translate(v, current_language)
            )
            ranked_df["suitability_display"] = ranked_df["suitability_label"].apply(
                lambda v: translate_status(v, ui_text)
            )
            ranked_df["outlier_display"] = ranked_df["outlier_text"].apply(
                lambda v: translate_status(v, ui_text)
            )

            ranked_rows = ranked_df.to_dict(orient="records")

            for row in ranked_rows:
                coords = AREA_COORDS.get(row["area"])
                row["lat"] = coords[0] if coords else None
                row["lon"] = coords[1] if coords else None

            map_rows = [r for r in ranked_rows if r["lat"] is not None][:5]

            if ranked_rows:
                best_area = ranked_rows[0]

                fig1 = px.bar(
                    ranked_df,
                    x="tcol_value",
                    y="area_display",
                    orientation="h",
                    color="suitability_label",
                    color_discrete_map={
                        "Highly Suitable": "#34D399",
                        "Moderately Suitable": "#60A5FA",
                        "Not Suitable": "#A78BFA",
                    },
                    title=ui_text["tcost_comparison"],
                )
                style_plotly(fig1)
                tcost_chart = fig1.to_json()

                fig2 = px.scatter(
                    ranked_df,
                    x="safety_score",
                    y="tcol_value",
                    color="cluster_display",
                    hover_name="area_display",
                    size="predicted_rent_value",
                    title=ui_text["safety_vs_tcol"],
                )
                style_plotly(fig2)
                scatter_chart = fig2.to_json()

                fig3 = px.bar(
                    ranked_df.sort_values("suitability_score", ascending=True),
                    x="suitability_score",
                    y="area_display",
                    orientation="h",
                    color="suitability_label",
                    color_discrete_map={
                        "Highly Suitable": "#34D399",
                        "Moderately Suitable": "#60A5FA",
                        "Not Suitable": "#A78BFA",
                    },
                    title=ui_text["suitability_score_ladder"],
                )
                style_plotly(fig3)
                suitability_chart = fig3.to_json()

                cluster_df = ranked_df.groupby(
                    "cluster_display", as_index=False
                ).agg(area_count=("area", "count"))

                fig4 = px.pie(
                    cluster_df,
                    values="area_count",
                    names="cluster_display",
                    hole=0.55,
                    title=ui_text["cluster_mix"],
                )
                style_plotly(fig4)
                cluster_chart = fig4.to_json()

    return render_template(
        "ranked.html",
        active_page="ranked",
        cities=cities,
        city_options=build_display_options(cities, current_language),
        house_types=house_types,
        house_type_options=build_display_options(house_types, current_language),
        priorities=priorities,
        priority_options=build_display_options(priorities, current_language),
        selected_city=selected_city,
        monthly_budget=monthly_budget,
        house_type=house_type,
        work_locations=work_locations,
        selected_work_location=selected_work_location,
        eating_frequency=eating_frequency,
        budget_priority=budget_priority,
        safety_priority=safety_priority,
        commute_priority=commute_priority,
        food_priority=food_priority,
        ranked_rows=ranked_rows,
        best_area=best_area,
        tcost_chart=tcost_chart,
        scatter_chart=scatter_chart,
        suitability_chart=suitability_chart,
        cluster_chart=cluster_chart,
        ui_text=ui_text,
        current_language=current_language,
        search_query=language_context["search_query"],
        parsed_query=parsed,
        map_rows=map_rows,
        user_type=user_type,
        work_location_options=work_location_options,
        custom_place=custom_place,
        show_map_suggestions=show_map_suggestions,
    )


@app.route("/comparison", methods=["GET", "POST"])
def comparison():
    params = request.args if request.method == "GET" else request.form
    language_context = get_language_context(params)

    ui_text = language_context["ui_text"]
    current_language = language_context["current_language"]
    parsed = language_context["parsed_query"]

    cities = sorted(MAIN_DF["city"].dropna().unique().tolist())
    selected_city = pick_valid_choice(params.get("city"), cities)
    monthly_budget = int(params.get("monthly_budget", 20000))

    if parsed and parsed.get("city"):
        selected_city = pick_valid_choice(parsed["city"], cities)

    if parsed and parsed.get("monthly_budget"):
        monthly_budget = parsed["monthly_budget"]

    city_df = AREA_SUMMARY_DF[AREA_SUMMARY_DF["city"] == selected_city].copy()
    areas = sorted(city_df["area"].dropna().unique().tolist())

    current_area = pick_valid_choice(
        parsed.get("current_area", params.get("current_area")),
        areas,
        0
    )

    # ✅ MULTI TARGET AREAS
    target_defaults = [area for area in areas if area != current_area]

    target_areas = params.getlist("target_areas")

    if parsed and parsed.get("target_area"):
        target_areas = [parsed.get("target_area")]

    if not target_areas:
        target_areas = target_defaults[:2]

    target_areas = [a for a in target_areas if a in areas and a != current_area]

    comparison_rows = []
    comparison_chart = None
    radar_chart = None
    food_rows = []
    food_summary_chart = None
    rent_message = ""

    if selected_city and current_area and target_areas:
        rows = []

        # ✅ Compare current + many targets
        for area_name in [current_area] + target_areas:
            area_df = city_df[city_df["area"] == area_name].copy()

            if area_df.empty:
                continue

            row = area_df.iloc[0]

            predicted_rent = (
                row["predicted_rent"]
                if not pd.isna(row["predicted_rent"])
                else row["actual_rent"]
            )

            monthly_food_cost = float(row["avg_meal_price"]) * 60
            commute_cost = float(row["distance_km"]) * 4 * 26 * 2
            tcol = float(predicted_rent) + monthly_food_cost + commute_cost

            suitability = predict_suitability_label(
                {
                    "tcol": tcol,
                    "safety_score": row["safety_score"],
                    "monthly_food_cost": monthly_food_cost,
                    "commute_cost": commute_cost,
                },
                monthly_budget,
                {
                    "budget_priority": "High",
                    "safety_priority": "High",
                    "commute_priority": "Medium",
                    "food_priority": "Medium",
                },
            )[0]

            rows.append({
                "area": area_name,
                "area_display": safe_translate(area_name, current_language),
                "predicted_rent": rupees(predicted_rent),
                "predicted_rent_value": float(predicted_rent),
                "monthly_food_cost": rupees(monthly_food_cost),
                "monthly_food_cost_value": float(monthly_food_cost),
                "commute_cost": rupees(commute_cost),
                "commute_cost_value": float(commute_cost),
                "tcol": rupees(tcol),
                "tcol_value": float(tcol),
                "safety_score": round(float(row["safety_score"]), 2),
                "suitability_label": suitability,
                "suitability_display": translate_status(suitability, ui_text),
            })

        comparison_rows = rows

        if rows:
            chart_df = pd.DataFrame(rows)

            fig = go.Figure()
            fig.add_bar(
                x=chart_df["area_display"],
                y=chart_df["predicted_rent_value"],
                name=ui_text["predicted_rent"]
            )
            fig.add_bar(
                x=chart_df["area_display"],
                y=chart_df["monthly_food_cost_value"],
                name=ui_text["food_cost"]
            )
            fig.add_bar(
                x=chart_df["area_display"],
                y=chart_df["commute_cost_value"],
                name=ui_text["commute_cost"]
            )
            fig.add_trace(go.Scatter(
                x=chart_df["area_display"],
                y=chart_df["tcol_value"],
                mode="lines+markers",
                name=ui_text["total_cost"],
            ))

            fig.update_layout(
                barmode="group",
                title=ui_text["cost_comparison_chart"]
            )

            style_plotly(fig)
            comparison_chart = fig.to_json()

            rent_max = max(chart_df["predicted_rent_value"].max(), 1)
            food_max = max(chart_df["monthly_food_cost_value"].max(), 1)
            commute_max = max(chart_df["commute_cost_value"].max(), 1)
            tcol_max = max(chart_df["tcol_value"].max(), 1)

            radar_fig = go.Figure()

            axes = [
                "Affordability",
                "Food Value",
                "Commute Ease",
                "Safety",
                "TCoL Efficiency"
            ]

            axes_display = [
                safe_translate(axis, current_language)
                for axis in axes
            ]

            for _, row in chart_df.iterrows():
                affordability = round(
                    100 - ((row["predicted_rent_value"] / rent_max) * 100), 1
                )
                food_value = round(
                    100 - ((row["monthly_food_cost_value"] / food_max) * 100), 1
                )
                commute_ease = round(
                    100 - ((row["commute_cost_value"] / commute_max) * 100), 1
                )
                tcol_efficiency = round(
                    100 - ((row["tcol_value"] / tcol_max) * 100), 1
                )

                radar_fig.add_trace(go.Scatterpolar(
                    r=[
                        affordability,
                        food_value,
                        commute_ease,
                        round(row["safety_score"], 1),
                        tcol_efficiency,
                        affordability
                    ],
                    theta=axes_display + [axes_display[0]],
                    fill="toself",
                    name=row["area_display"],
                ))

            radar_fig.update_layout(
                title=ui_text["decision_radar"],
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                )
            )

            style_plotly(radar_fig)
            radar_chart = radar_fig.to_json()

        # ✅ Food Compare
        for area_name in [current_area] + target_areas:
            area_food = FOOD_DF[
                (FOOD_DF["city"] == selected_city) &
                (FOOD_DF["area"] == area_name)
            ].copy()

            food_rows.append({
                "area": area_name,
                "area_display": safe_translate(area_name, current_language),
                "total_shops": len(area_food),
                "avg_meal_price": round(
                    area_food["avg_meal_price"].mean(), 2
                ) if not area_food.empty else 0,
                "avg_rating": round(
                    area_food["rating"].mean(), 2
                ) if not area_food.empty else 0,
            })

        if food_rows:
            food_df = pd.DataFrame(food_rows)

            food_fig = go.Figure()
            food_fig.add_bar(
                x=food_df["area_display"],
                y=food_df["total_shops"],
                name=ui_text["total_shops"]
            )

            food_fig.add_trace(go.Scatter(
                x=food_df["area_display"],
                y=food_df["avg_rating"],
                mode="lines+markers",
                name=ui_text["avg_rating"],
                yaxis="y2",
            ))

            food_fig.update_layout(
                title=ui_text["food_availability_snapshot"],
                yaxis=dict(title=ui_text["total_shops"]),
                yaxis2=dict(
                    title=ui_text["avg_rating"],
                    overlaying="y",
                    side="right",
                    range=[0, 5]
                ),
            )

            style_plotly(food_fig)
            food_summary_chart = food_fig.to_json()

        # ✅ Insight
        if len(target_areas) == 1 and len(comparison_rows) >= 2:
            current_rent = comparison_rows[0]["predicted_rent_value"]
            target_rent = comparison_rows[1]["predicted_rent_value"]

            diff = target_rent - current_rent
            target_name = target_areas[0]

            if diff > 0:
                rent_message = f"{target_name} requires ₹{abs(round(diff))} higher monthly rent than {current_area}."
            elif diff < 0:
                rent_message = f"{target_name} saves ₹{abs(round(diff))} compared to {current_area}."
            else:
                rent_message = "Both areas have similar rent."
        else:
            rent_message = f"Compared {len(comparison_rows)} selected areas."

    return render_template(
        "comparison.html",
        active_page="comparison",
        cities=cities,
        city_options=build_display_options(cities, current_language),
        selected_city=selected_city,
        monthly_budget=monthly_budget,
        areas=areas,
        area_options=build_display_options(areas, current_language),
        current_area=current_area,
        target_areas=target_areas,
        comparison_rows=comparison_rows,
        comparison_chart=comparison_chart,
        radar_chart=radar_chart,
        food_rows=food_rows,
        food_summary_chart=food_summary_chart,
        ui_text=ui_text,
        current_language=current_language,
        search_query=language_context["search_query"],
        parsed_query=parsed,
        rent_message=rent_message,
    )

@app.route("/food", methods=["GET", "POST"])
def food():
    params = request.args if request.method == "GET" else request.form
    language_context = get_language_context(params)
    ui_text = language_context["ui_text"]
    current_language = language_context["current_language"]

    cities = sorted(FOOD_DF["city"].dropna().unique().tolist())
    selected_city = pick_valid_choice(params.get("city"), cities)

    parsed = language_context["parsed_query"]
    if parsed and parsed.get("city"):
        selected_city = pick_valid_choice(parsed["city"], cities)

    city_food = FOOD_DF[FOOD_DF["city"] == selected_city].copy()
    areas = sorted(city_food["area"].dropna().unique().tolist())
    
    selected_area = pick_valid_choice(
    parsed.get("area", params.get("area")),
    areas
)
    selected_food_type = parsed.get("food_type", params.get("food_type", "Both"))
    selected_cuisine = parsed.get("cuisine", params.get("cuisine", "All"))

    max_price = int(float(parsed.get("monthly_budget", params.get("max_price", 300))))
    min_rating = float(params.get("min_rating", 4.0))

    food_types = ["Both", "Restaurant", "Cloud Kitchen"]
    translated_food_types = [
        {"original": v, "display": safe_translate(v, current_language)}
        for v in food_types
    ]

    area_food = city_food[city_food["area"] == selected_area].copy()
    cuisines = ["All"] + sorted(area_food["cuisine"].dropna().astype(str).unique().tolist())

    if selected_food_type not in food_types:
        selected_food_type = "Both"
    if selected_cuisine not in cuisines:
        selected_cuisine = "All"

    food_table = []
    food_stats = None
    food_chart = None
    cuisine_chart = None
    value_chart = None

    if selected_city and selected_area:
        filtered = area_food.copy()

        if selected_food_type != "Both":
            filtered = filtered[filtered["food_type"].astype(str).str.lower() == selected_food_type.lower()]

        if selected_cuisine != "All":
            filtered = filtered[filtered["cuisine"] == selected_cuisine]

        filtered = filtered[filtered["avg_meal_price"] <= max_price]
        filtered = filtered[filtered["rating"] >= min_rating]

        if not filtered.empty:
            display_df = filtered.copy()
            display_df["restaurant_name_display"] = display_df["restaurant_name"].apply(lambda v: safe_translate(v, current_language))
            display_df["food_type_display"] = display_df["food_type"].apply(lambda v: safe_translate(v, current_language))
            display_df["cuisine_display"] = display_df["cuisine"].apply(lambda v: safe_translate(v, current_language))
            food_table = display_df.sort_values(
                by=["rating", "avg_meal_price"],
                ascending=[False, True]
            ).to_dict(orient="records")

            food_stats = {
                "count": len(display_df),
                "avg_price": rupees(round(display_df["avg_meal_price"].mean(), 0)),
                "avg_rating": round(display_df["rating"].mean(), 2),
            }

            fig = px.scatter(
                display_df,
                x="avg_meal_price",
                y="rating",
                color="food_type_display",
                hover_name="restaurant_name_display",
                size="rating",
                title=ui_text["food_price_vs_rating"],
            )
            style_plotly(fig)
            food_chart = fig.to_json()

            cuisine_df = display_df.groupby("cuisine_display", as_index=False).agg(
                shop_count=("restaurant_name_display", "count")
            ).sort_values("shop_count", ascending=False).head(8)
            cuisine_fig = px.bar(
                cuisine_df,
                x="cuisine_display",
                y="shop_count",
                title=ui_text["cuisine_spread"],
            )
            style_plotly(cuisine_fig)
            cuisine_chart = cuisine_fig.to_json()

            value_df = display_df.copy()
            value_df["value_score"] = (value_df["rating"] * 20) / value_df["avg_meal_price"].clip(lower=1)
            value_df = value_df.sort_values("value_score", ascending=False).head(8)
            value_fig = px.bar(
                value_df.sort_values("value_score", ascending=True),
                x="value_score",
                y="restaurant_name_display",
                orientation="h",
                color="food_type_display",
                title=ui_text["best_value_picks"],
                hover_data=["rating", "avg_meal_price", "cuisine_display"],
            )
            style_plotly(value_fig)
            value_chart = value_fig.to_json()

    return render_template(
        "food.html",
        active_page="food",
        cities=cities,
        city_options=build_display_options(cities, current_language),
        selected_city=selected_city,
        areas=areas,
        area_options=build_display_options(areas, current_language),
        selected_area=selected_area,
        food_types=food_types,
        food_type_options=translated_food_types,
        selected_food_type=selected_food_type,
        cuisines=cuisines,
        cuisine_options=build_display_options(cuisines, current_language),
        selected_cuisine=selected_cuisine,
        max_price=max_price,
        min_rating=min_rating,
        food_table=food_table,
        food_stats=food_stats,
        food_chart=food_chart,
        cuisine_chart=cuisine_chart,
        value_chart=value_chart,
        ui_text=ui_text,
        current_language=current_language,
        search_query=language_context["search_query"],
        parsed_query=parsed,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
