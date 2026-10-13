from __future__ import annotations

import re
import unicodedata

# Characters that are invisible or change how text renders versus how it is processed:
# C0/C1 controls (except \n \t), zero-width and bidi controls, soft hyphen, Arabic letter
# mark, Mongolian vowel separator, line/paragraph separators, variation selectors and the
# Unicode *tag* block (used to smuggle invisible ASCII to a model).
_STRIP = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
    r"\u00ad\u061c\u180e\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u2064\u2066-\u2069"
    r"\ufe00-\ufe0f\ufeff\U000e0000-\U000e007f\U000e0100-\U000e01ef]"
)


def strip_invisible(text: str) -> str:
    return _STRIP.sub("", text)


def clean_user_text(text: str) -> str:
    return unicodedata.normalize("NFC", strip_invisible(text)).strip()
