"""Maps multilingual query keywords to dashboard-friendly preferences."""

BUDGET_KEYWORDS = [
    # English
    "cheap", "affordable", "low rent", "budget", "less cost", "low budget",
    "economical", "save money", "budget-friendly",

    # Tamil
    "குறைந்த வாடகை", "குறைந்த செலவு", "மலிவு", "பட்ஜெட்",

    # Kannada
    "ಕಡಿಮೆ ಬಾಡಿಗೆ", "ಕಡಿಮೆ ವೆಚ್ಚ", "ಅಗ್ಗದ", "ಬಜೆಟ್",

    # Hindi
    "कम किराया", "कम खर्च", "सस्ता", "बजट",
]

SAFETY_KEYWORDS = [
    # English
    "safe", "safest", "secure", "low crime", "family area", "peaceful",

    # Tamil
    "பாதுகாப்பான", "பாதுகாப்பு", "அமைதியான",

    # Kannada
    "ಸುರಕ್ಷಿತ", "ಭದ್ರ", "ಶಾಂತ ಪ್ರದೇಶ",

    # Hindi
    "सुरक्षित", "सुरक्षा", "शांत",
]

COMMUTE_KEYWORDS = [
    # English
    "near office", "close to office", "near work", "short commute",
    "less travel", "near workplace", "office nearby",

    # Tamil
    "அலுவலகத்திற்கு அருகில்", "வேலை இடத்திற்கு அருகில்", "குறைந்த பயணம்",

    # Kannada
    "ಕಚೇರಿಯ ಹತ್ತಿರ", "ಕೆಲಸದ ಸ್ಥಳದ ಹತ್ತಿರ", "ಕಡಿಮೆ ಪ್ರಯಾಣ",

    # Hindi
    "ऑफिस के पास", "कार्यस्थल के पास", "कम यात्रा",
]

FOOD_KEYWORDS = [
    # English
    "food", "restaurants", "good food", "eating", "cafes", "cloud kitchen",
    "food options", "restaurants nearby",

    # Tamil
    "உணவு", "உணவகம்", "சாப்பாடு", "ரெஸ்டாரண்ட்", "கிளவுட் கிச்சன்",

    # Kannada
    "ಆಹಾರ", "ಉಪಹಾರ ಗೃಹ", "ಊಟ", "ರೆಸ್ಟೋರೆಂಟ್", "ಕ್ಲೌಡ್ ಕಿಚನ್",

    # Hindi
    "भोजन", "रेस्तरां", "खाना", "फूड", "क्लाउड किचन",
]

PREMIUM_KEYWORDS = [
    # English
    "premium", "luxury", "high class", "posh", "expensive",

    # Tamil
    "பிரீமியம்", "லக்சுரி", "விலையுயர்ந்த",

    # Kannada
    "ಪ್ರೀಮಿಯಂ", "ಐಷಾರಾಮಿ", "ದುಬಾರಿ",

    # Hindi
    "प्रीमियम", "लक्ज़री", "महंगा",
]


def map_keywords_to_preferences(text: str) -> dict:
    """
    Convert a multilingual query into priority preferences using keyword matching.
    """
    text = str(text or "").lower()

    preferences = {
        "budget_priority": "Medium",
        "safety_priority": "Medium",
        "commute_priority": "Medium",
        "food_priority": "Medium",
    }

    if any(keyword in text for keyword in BUDGET_KEYWORDS):
        preferences["budget_priority"] = "High"

    if any(keyword in text for keyword in SAFETY_KEYWORDS):
        preferences["safety_priority"] = "High"

    if any(keyword in text for keyword in COMMUTE_KEYWORDS):
        preferences["commute_priority"] = "High"

    if any(keyword in text for keyword in FOOD_KEYWORDS):
        preferences["food_priority"] = "High"

    if any(keyword in text for keyword in PREMIUM_KEYWORDS):
        preferences["budget_priority"] = "Low"

    return preferences
