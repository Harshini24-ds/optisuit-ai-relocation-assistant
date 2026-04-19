"""Text cleanup utilities for multilingual user search queries."""

import re


def normalize_query(text: str) -> str:
    """
    Clean a user query before language detection or intent parsing.

    Steps:
    - convert None to empty string
    - strip extra spaces
    - remove repeated punctuation
    - lowercase for easier keyword matching
    """
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[!?,.;:]{2,}", " ", text)
    return text.lower()
