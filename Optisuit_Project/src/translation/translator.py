"""Translation helpers using deep-translator."""

from deep_translator import GoogleTranslator

WORKING_LANGUAGE = "en"
_translation_cache = {}

def _translate_text(text: str, source_language: str, target_language: str) -> str:
    text = str(text or "").strip()

    if not text:
        return ""

    if source_language == target_language:
        return text

    cache_key = (text, source_language, target_language)
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    try:
        translated = GoogleTranslator(
            source=source_language,
            target=target_language
        ).translate(text)

        _translation_cache[cache_key] = translated
        return translated

    except Exception:
        return text


def translate_to_working_language(text: str, source_language: str) -> str:
    return _translate_text(text, source_language, WORKING_LANGUAGE)


def translate_from_working_language(text: str, target_language: str) -> str:
    return _translate_text(text, WORKING_LANGUAGE, target_language)


def translate_dynamic_value(value: str, target_language: str) -> str:
    if not value:
        return ""

    if target_language == "en":
        return str(value)

    return translate_from_working_language(str(value), target_language)