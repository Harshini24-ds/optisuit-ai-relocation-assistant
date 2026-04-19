import re

CITY_HINTS = {
    "chennai": "Chennai",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
}

HOUSE_TYPE_HINTS = {
    "pg": "PG",
    "1bhk": "1BHK",
    "2bhk": "2BHK",
}

WORK_LOCATION_HINTS = {
    "adyar": "Adyar",
    "anna nagar": "Anna Nagar",
    "velachery": "Velachery",
    "tambaram": "Tambaram",
    "porur": "Porur",
    "guindy": "Guindy",
    "omr": "Omr It Corridor",
    "taramani": "Taramani",
    "whitefield": "Whitefield Tech Park",
    "marathahalli": "Marathahalli",
    "koramangala": "Koramangala",
}

AREA_HINTS = {
    "adyar": "Adyar",
    "anna nagar": "Anna Nagar",
    "velachery": "Velachery",
    "tambaram": "Tambaram",
    "porur": "Porur",
    "mylapore": "Mylapore",
    "nungambakkam": "Nungambakkam",
    "perungudi": "Perungudi",
    "sholinganallur": "Sholinganallur",
}

FOOD_TYPES = {
    "restaurant": "Restaurant",
    "cloud kitchen": "Cloud Kitchen",
}

CUISINES = [
    "Biryani", "South Indian", "North Indian", "Chinese",
    "Italian", "Fast Food", "Bakery", "Desserts", "Street Food"
]


def extract_budget(text):
    match = re.search(r'(\d{4,6})', text)
    return int(match.group(1)) if match else None


def extract_city(text):
    for key, val in CITY_HINTS.items():
        if key in text:
            return val
    return ""


def extract_house_type(text):
    for key, val in HOUSE_TYPE_HINTS.items():
        if key in text:
            return val
    return ""


def extract_work_location(text):
    for key, val in WORK_LOCATION_HINTS.items():
        if key in text:
            return val
    return ""


AREA_HINTS = {
    "adyar": "Adyar",
    "tambaram": "Tambaram",
    "guindy": "Guindy",
    "velachery": "Velachery",
    "nungambakkam": "Nungambakkam",
    "perungudi": "Perungudi",
    "porur": "Porur",
    "ambattur": "Ambattur",
    "anna nagar": "Anna Nagar",
    "t nagar": "T Nagar",
    "tnagar": "T Nagar",

    # OMR aliases
    "omr": "Omr",
    "old mahabalipuram road": "Omr",
}

AREA_TO_CITY = {
    # Chennai
    "Adyar": "Chennai",
    "Tambaram": "Chennai",
    "Omr": "Chennai",
    "Guindy": "Chennai",
    "Velachery": "Chennai",
    "Nungambakkam": "Chennai",
    "Perungudi": "Chennai",
    "Porur": "Chennai",
    "Ambattur": "Chennai",
    "Anna Nagar": "Chennai",
    "T Nagar": "Chennai",

    # Bengaluru
    "Whitefield": "Bengaluru",
    "Koramangala": "Bengaluru",
    "Indiranagar": "Bengaluru",
    "Electronic City": "Bengaluru",
    "Marathahalli": "Bengaluru",
    "HSR Layout": "Bengaluru",
    "BTM Layout": "Bengaluru",
    "Jayanagar": "Bengaluru",
    "Hebbal": "Bengaluru",
    "Yelahanka": "Bengaluru",
    "Malleshwaram": "Bengaluru",
}


def extract_area(text):
    text = text.lower().strip()

    for key, val in AREA_HINTS.items():
        if key in text:
            return val

    return ""


def detect_priorities(text):
    result = {}

    if any(word in text for word in ["cheap", "affordable", "budget"]):
        result["budget_priority"] = "High"

    if any(word in text for word in ["safe", "security"]):
        result["safety_priority"] = "High"

    if any(word in text for word in ["near", "office", "work"]):
        result["commute_priority"] = "High"

    if "food" in text:
        result["food_priority"] = "High"

    return result


def detect_intent(text):
    if "compare" in text or "vs" in text:
        return "comparison"
    if any(word in text for word in ["food", "restaurant", "biryani"]):
        return "food_search"
    return "ranked_search"


def extract_compare_areas(text):
    text = text.lower().strip()

    match = re.search(r'compare\s+(.*?)\s+vs\s+(.*)', text)
    if not match:
        return "", ""

    left_part = match.group(1).strip()
    right_part = match.group(2).strip()

    # Remove extra trailing words from right side
    right_part = re.split(r'\s+in\s+|\s+under\s+|\s+with\s+', right_part)[0].strip()

    current_area = extract_area(left_part)
    target_area = extract_area(right_part)

    return current_area, target_area


def extract_food_type(text):
    for key, val in FOOD_TYPES.items():
        if key in text:
            return val
    return "Both"


def extract_cuisine(text):
    for item in CUISINES:
        if item.lower() in text:
            return item
    return "All"
USER_TYPE_HINTS = {
    "student": "student",
    "it professional": "it_professional",
    "software engineer": "it_professional",
    "developer": "it_professional",
    "family": "family",
    "senior": "senior",
    "senior citizen": "senior",
    "bachelor": "bachelor",
    "luxury": "luxury",
}

def extract_user_type(text):
    for key, val in USER_TYPE_HINTS.items():
        if key in text:
            return val
    return ""

def parse_search_query(query):
    text = query.lower().strip()

    result = {
        "intent": detect_intent(text),
        "city": extract_city(text),
        "house_type": extract_house_type(text),
        "monthly_budget": extract_budget(text),
        "work_location": extract_work_location(text),
        "area": extract_area(text),
        "food_type": extract_food_type(text),
        "cuisine": extract_cuisine(text),
        "user_type": extract_user_type(text),
    }

    current_area, target_area = extract_compare_areas(text)
    result["current_area"] = current_area
    result["target_area"] = target_area

    # Auto detect city if user didn't type city
    if not result["city"]:
        if result["area"] in AREA_TO_CITY:
            result["city"] = AREA_TO_CITY[result["area"]]
        elif current_area in AREA_TO_CITY:
            result["city"] = AREA_TO_CITY[current_area]
        elif target_area in AREA_TO_CITY:
            result["city"] = AREA_TO_CITY[target_area]

    result.update(detect_priorities(text))

    return result