COMPRESSION_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

LANGUAGES = {
    # "fi": "Finnish",
    "fr": "French",
    "hu": "Hungarian",
    # "ja": "Japanese",
    # "ko": "Korean",
    "lt": "Lithuanian",
    "pl": "Polish",
    "ro": "Romanian",
    "no": "Norwegian",
}


OPUS_MT_MODELS = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    # "ja": "Helsinki-NLP/opus-tatoeba-en-ja",
    # "ko": "Helsinki-NLP/opus-mt-tc-big-en-ko",
    "pl": "Helsinki-NLP/opus-mt-en-zlw",
    "no": "Helsinki-NLP/opus-mt-en-gmq",
    # "da": "Helsinki-NLP/opus-mt-en-gmq",
    "hu": "Helsinki-NLP/opus-mt-tc-big-en-hu",
    "lt": "Helsinki-NLP/opus-mt-tc-big-en-lt",
}


LANG_CODES = {
    "fi": "fin",
    "fr": "fra",
    "ja": "jpn",
    "ko": "kor",
    "lt": "lit",
    "pl": "pol",
    "ro": "ron",
    "no": "nob",  # norwegian bokmål
}


prefixes = {
    "no": ">>nob<<",
    "da": ">>dan<<",
}


def get_prefix(lang):
    if lang in LANG_CODES:
        return f">>{LANG_CODES[lang]}<<"
    return ""
