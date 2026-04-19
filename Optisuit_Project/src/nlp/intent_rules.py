"""Rule-based intent extraction for relocation queries."""

from src.nlp.keyword_mapper import map_keywords_to_preferences


def infer_intent(query_text: str) -> dict:
    """
    Infer the user's intent from a free-text query.

    Supported intents:
    - ranked_search
    - comparison
    - food_search
    """
    text = str(query_text or "").lower()

    result = {
        "intent": "ranked_search",
        "budget_priority": "Medium",
        "safety_priority": "Medium",
        "commute_priority": "Medium",
        "food_priority": "Medium",
        "comparison_mode": False,
        "food_mode": False,
    }

    if any(word in text for word in ["compare", "comparison", "vs", "versus", "better than"]):
        result["intent"] = "comparison"
        result["comparison_mode"] = True

    elif any(word in text for word in ["food", "restaurant", "eat", "cuisine", "cloud kitchen"]):
        result["intent"] = "food_search"
        result["food_mode"] = True

    result.update(map_keywords_to_preferences(text))
    return result
