"""Reusable text and currency formatting helpers."""


def as_currency(value) -> str:
    """
    Format a number as Indian rupees for display.
    """
    try:
        return f"Rs.{float(value):,.0f}"
    except (TypeError, ValueError):
        return "Rs.0"
