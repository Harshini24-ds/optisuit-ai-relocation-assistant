"""Helpers for translating model responses before rendering them."""

from src.translation.translator import translate_from_working_language


def translate_response_payload(payload: dict, language_code: str) -> dict:
    """
    Translate string values in a response payload to the user's language.
    Non-string values are returned as-is.
    """
    translated = {}

    for key, value in payload.items():
        if isinstance(value, str):
            translated[key] = translate_from_working_language(value, language_code)
        else:
            translated[key] = value

    return translated

