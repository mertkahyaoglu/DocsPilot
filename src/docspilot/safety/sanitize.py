from __future__ import annotations

import re
import unicodedata

# C0/C1 controls except \n and \t; zero-width characters; bidi overrides/isolates (the
# "trojan source" family: text that renders differently from what is processed).
_STRIP = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏‪-‮⁠-⁤⁦-⁩﻿]")


def clean_user_text(text: str) -> str:
    return unicodedata.normalize("NFC", _STRIP.sub("", text)).strip()
