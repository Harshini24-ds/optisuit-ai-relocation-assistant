"""Detect a user's language before translation or NLP processing."""

import re


def detect_language(text: str) -> str:
    """
    Simple script-based language detection.

    Returns:
    - 'kn' for Kannada
    - 'ta' for Tamil
    - 'hi' for Hindi
    - 'en' for English/default
    """
    text = str(text or "").strip()

    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kn"

    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"

    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    return "en"
