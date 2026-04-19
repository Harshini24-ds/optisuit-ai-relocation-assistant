"""General utility helpers."""


def safe_number(value, default=0):
    """
    Convert a value to float safely.
    If conversion fails, return the default.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
