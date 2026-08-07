"""Gloss to Sentence NLG engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

# Semantic role tags
_ROLE_MAP: dict[str, str] = {
    "hello": "GREETING",
    "goodbye": "GREETING",
    "welcome": "GREETING",
    "thanks": "POLITE",
    "please": "POLITE",
    "sorry": "POLITE",
    "thank_you": "POLITE",
    "yes": "AFFIRM",
    "no": "AFFIRM",
    "maybe": "AFFIRM",
    "happy": "EMOTION",
    "sad": "EMOTION",
    "love": "EMOTION",
    "what": "QUESTION",
    "where": "QUESTION",
    "how": "QUESTION",
    "why": "QUESTION",
    "mother": "PERSON",
    "father": "PERSON",
    "friend": "PERSON",
    "family": "PERSON",
    "go": "ACTION",
    "come": "ACTION",
    "see": "ACTION",
    "know": "ACTION",
    "eat_food": "ACTION",
    "drink": "ACTION",
    "want": "ACTION",
    "need": "ACTION",
    "help": "ACTION",
    "stop": "ACTION",
    "wait": "ACTION",
    "finish": "ACTION",
    "work": "ACTION",
    "understand": "ACTION",
    "good": "DESC",
    "big": "DESC",
    "small": "DESC",
    "home": "LOCATION",
    "school": "LOCATION",
    "outside": "LOCATION",
    "inside": "LOCATION",
    "today": "TIME",
    "tomorrow": "TIME",
    "yesterday": "TIME",
    "later": "TIME",
    "soon": "TIME",
    "always": "TIME",
    "never": "TIME",
    "sometimes": "TIME",
    "morning": "TIME",
    "night": "TIME",
    "afternoon": "TIME",
    "evening": "TIME",
    "more": "QUANTITY",
    "again": "QUANTITY",
    "name": "NAME",
    "water": "NOUN",
    "fingerspell_a": "FINGERSPELL",
    "fingerspell_b": "FINGERSPELL",
    "fingerspell_c": "FINGERSPELL",
    "fingerspell_d": "FINGERSPELL",
    "fingerspell_e": "FINGERSPELL",
    "fingerspell_f": "FINGERSPELL",
    "fingerspell_g": "FINGERSPELL",
    "fingerspell_h": "FINGERSPELL",
    "fingerspell_i": "FINGERSPELL",
    "fingerspell_j": "FINGERSPELL",
    "fingerspell_k": "FINGERSPELL",
    "fingerspell_l": "FINGERSPELL",
    "fingerspell_m": "FINGERSPELL",
    "fingerspell_n": "FINGERSPELL",
    "fingerspell_o": "FINGERSPELL",
    "fingerspell_p": "FINGERSPELL",
    "fingerspell_q": "FINGERSPELL",
    "fingerspell_r": "FINGERSPELL",
    "fingerspell_s": "FINGERSPELL",
    "fingerspell_t": "FINGERSPELL",
    "fingerspell_u": "FINGERSPELL",
    "fingerspell_v": "FINGERSPELL",
    "fingerspell_w": "FINGERSPELL",
    "fingerspell_x": "FINGERSPELL",
    "fingerspell_y": "FINGERSPELL",
    "fingerspell_z": "FINGERSPELL",
}

# Single-gloss sentences (EN)
_SINGLE_SENTENCES: dict[str, str] = {
    "hello": "Hello!",
    "thanks": "Thank you.",
    "yes": "Yes.",
    "no": "No.",
    "help": "I need help.",
    "please": "Please.",
    "love": "I love this.",
    "name": "What is your name?",
    "water": "I want water.",
    "good": "That is good.",
    "goodbye": "Goodbye!",
    "sorry": "I am sorry.",
    "stop": "Please stop.",
    "want": "I want that.",
    "need": "I need this.",
    "happy": "I am happy.",
    "sad": "I feel sad.",
    "mother": "Mother.",
    "father": "Father.",
    "friend": "This is my friend.",
    "eat_food": "I want to eat.",
    "drink": "I want a drink.",
    "home": "I am going home.",
    "school": "I am at school.",
    "go": "Let's go.",
    "come": "Please come here.",
    "see": "I see it.",
    "know": "I know.",
    "big": "It is big.",
    "small": "It is small.",
    "welcome": "Welcome!",
    "maybe": "Maybe.",
    "wait": "Please wait.",
    "today": "Today.",
    "understand": "I understand.",
    "again": "Again.",
    "more": "I want more.",
    "finish": "I am finished.",
    "what": "What?",
    "where": "Where?",
    "how": "How?",
    "why": "Why?",
    "family": "My family.",
    "work": "I am working.",
    "later": "See you later.",
    "tomorrow": "Tomorrow.",
    "yesterday": "Yesterday.",
    "outside": "Outside.",
    "inside": "Inside.",
    "night": "Good night.",
    "morning": "Good morning.",
    "afternoon": "Good afternoon.",
    "evening": "Good evening.",
    "soon": "See you soon.",
    "always": "Always.",
    "never": "Never.",
    "sometimes": "Sometimes.",
    "thank_you": "Thank you.",
    "fingerspell_a": "A.",
    "fingerspell_b": "B.",
    "fingerspell_c": "C.",
    "fingerspell_d": "D.",
    "fingerspell_e": "E.",
    "fingerspell_f": "F.",
    "fingerspell_g": "G.",
    "fingerspell_h": "H.",
    "fingerspell_i": "I.",
    "fingerspell_j": "J.",
    "fingerspell_k": "K.",
    "fingerspell_l": "L.",
    "fingerspell_m": "M.",
    "fingerspell_n": "N.",
    "fingerspell_o": "O.",
    "fingerspell_p": "P.",
    "fingerspell_q": "Q.",
    "fingerspell_r": "R.",
    "fingerspell_s": "S.",
    "fingerspell_t": "T.",
    "fingerspell_u": "U.",
    "fingerspell_v": "V.",
    "fingerspell_w": "W.",
    "fingerspell_x": "X.",
    "fingerspell_y": "Y.",
    "fingerspell_z": "Z.",
}

# Single-gloss sentences (VI)
_VI_SINGLE: dict[str, str] = {
    "hello": "Xin chao!",
    "thanks": "Cam on.",
    "yes": "Vang.",
    "no": "Khong.",
    "help": "Toi can giup do.",
    "please": "Lam on.",
    "love": "Toi thich dieu nay.",
    "name": "Ten ban la gi?",
    "water": "Toi muon nuoc.",
    "good": "Tot.",
    "goodbye": "Tam biet!",
    "sorry": "Toi xin loi.",
    "stop": "Xin dung lai.",
    "want": "Toi muon cai do.",
    "need": "Toi can cai nay.",
    "happy": "Toi vui.",
    "sad": "Toi buon.",
    "mother": "Me.",
    "father": "Cha.",
    "friend": "Day la ban toi.",
    "eat_food": "Toi muon an.",
    "drink": "Toi muon uong.",
    "home": "Toi ve nha.",
    "school": "Toi o truong.",
    "go": "Di nao.",
    "come": "Moi vao.",
    "see": "Toi thay.",
    "know": "Toi biet.",
    "big": "No to.",
    "small": "No nho.",
    "welcome": "Chao mung!",
    "maybe": "Co le.",
    "wait": "Xin cho.",
    "today": "Hom nay.",
    "understand": "Toi hieu.",
    "again": "Lan nua.",
    "more": "Toi muon them.",
    "finish": "Toi xong roi.",
    "what": "Cai gi?",
    "where": "O dau?",
    "how": "The nao?",
    "why": "Tai sao?",
    "family": "Gia dinh toi.",
    "work": "Toi dang lam viec.",
    "later": "Hen gap lai.",
    "tomorrow": "Ngay mai.",
    "yesterday": "Hom qua.",
    "outside": "Ben ngoai.",
    "inside": "Ben trong.",
    "night": "Chuc ngu ngon.",
    "morning": "Chao buoi sang.",
    "afternoon": "Chao buoi chieu.",
    "evening": "Chao buoi toi.",
    "soon": "Sap gap lai.",
    "always": "Luon luon.",
    "never": "Khong bao gio.",
    "sometimes": "Thinh thoang.",
    "thank_you": "Cam on ban.",
    "fingerspell_a": "A.",
    "fingerspell_b": "B.",
    "fingerspell_c": "C.",
    "fingerspell_d": "D.",
    "fingerspell_e": "E.",
    "fingerspell_f": "F.",
    "fingerspell_g": "G.",
    "fingerspell_h": "H.",
    "fingerspell_i": "I.",
    "fingerspell_j": "J.",
    "fingerspell_k": "K.",
    "fingerspell_l": "L.",
    "fingerspell_m": "M.",
    "fingerspell_n": "N.",
    "fingerspell_o": "O.",
    "fingerspell_p": "P.",
    "fingerspell_q": "Q.",
    "fingerspell_r": "R.",
    "fingerspell_s": "S.",
    "fingerspell_t": "T.",
    "fingerspell_u": "U.",
    "fingerspell_v": "V.",
    "fingerspell_w": "W.",
    "fingerspell_x": "X.",
    "fingerspell_y": "Y.",
    "fingerspell_z": "Z.",
}

# Multi-gloss templates (EN)
_MULTI_TEMPLATES_EN: list[tuple[tuple[str, ...], str]] = [
    (("GREETING", "FINGERSPELL"), "%s, my name is %s."),
    (("GREETING", "NAME"), "%s, my name is ___?"),
    (("QUESTION", "PERSON"), "%s is %s?"),
    (("QUESTION", "LOCATION"), "%s is the %s?"),
    (("QUESTION", "TIME"), "%s time is %s?"),
    (("QUESTION", "NOUN"), "%s is the %s?"),
    (("POLITE", "ACTION"), "%s, I want to %s."),
    (("ACTION", "QUANTITY"), "I want to %s %s."),
    (("PERSON", "ACTION"), "My %s %s."),
    (("DESC", "NOUN"), "That is %s %s."),
    (("EMOTION", "PERSON"), "I feel %s about my %s."),
    (("TIME", "ACTION"), "%s I %s."),
    (("LOCATION", "ACTION"), "I %s to %s."),
    (("AFFIRM", "QUESTION"), "%s, %s?"),
]

# Multi-gloss templates (VI)
_MULTI_TEMPLATES_VI: list[tuple[tuple[str, ...], str]] = [
    (("GREETING", "FINGERSPELL"), "%s, toi ten la %s."),
    (("GREETING", "NAME"), "%s, ten toi la ___?"),
    (("QUESTION", "PERSON"), "%s la %s?"),
    (("QUESTION", "LOCATION"), "%s la %s?"),
    (("QUESTION", "NOUN"), "%s la %s?"),
    (("POLITE", "ACTION"), "%s, toi muon %s."),
    (("ACTION", "QUANTITY"), "Toi muon %s %s."),
    (("PERSON", "ACTION"), "%s cua toi %s."),
    (("DESC", "NOUN"), "Do la %s %s."),
    (("EMOTION", "PERSON"), "Toi cam thay %s ve %s cua toi."),
    (("TIME", "ACTION"), "%s toi %s."),
    (("LOCATION", "ACTION"), "Toi %s den %s."),
    (("AFFIRM", "QUESTION"), "%s, %s?"),
]


class GlossNLG:
    """Template-based NLG engine that maps gloss token sequences to sentences.

    Pipeline: tokenize -> tag semantic roles -> match templates -> reorder -> fill.

    Supports English ("EN") and Vietnamese ("VI") locale stubs.
    """

    _templates_file: ClassVar[Path | None] = None

    def __init__(self, locale: str = "EN") -> None:
        self.locale = locale.upper()
        self._single: dict[str, str] = _VI_SINGLE if self.locale == "VI" else _SINGLE_SENTENCES
        self._multi: list[tuple[tuple[str, ...], str]] = (
            _MULTI_TEMPLATES_VI if self.locale == "VI" else _MULTI_TEMPLATES_EN
        )

    @classmethod
    def from_templates_file(cls, path: Path, locale: str = "EN") -> "GlossNLG":
        """Load templates from a JSON file and return a configured engine."""
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        instance = cls(locale=locale)
        templates = data.get("templates", [])
        loaded: list[tuple[tuple[str, ...], str]] = []
        for t in templates:
            pattern = tuple(t["pattern"])
            sentence = t.get(f"sentence_{locale.lower()}", t.get("sentence_en", ""))
            loaded.append((pattern, sentence))
        if loaded:
            instance._multi = loaded
        instances = data.get("instances", {})
        loc = instances.get(f"locale_{locale.lower()}", instances.get("locale_en", {}))
        if loc:
            instance._single.update(loc)
        return instance

    def generate(self, glosses: list[str]) -> str:
        """Convert a sequence of gloss tokens to a natural-language sentence."""
        if not glosses or all(not g.strip() for g in glosses):
            return ""
        clean = [g.strip().lower() for g in glosses if g.strip()]
        if not clean:
            return ""
        if len(clean) == 1:
            return self._single_gloss(clean[0])
        return self._multi_gloss(clean)

    def _single_gloss(self, gloss: str) -> str:
        key = gloss.lower().strip()
        if key in self._single:
            return self._single[key]
        return key.replace("_", " ").strip().capitalize() + "."

    def _multi_gloss(self, glosses: list[str]) -> str:
        roles = self._tag(glosses)
        sentence = self._match_template(roles, glosses)
        if sentence is not None:
            return sentence
        parts = [self._single_gloss(g).rstrip(".") for g in glosses]
        return " ".join(parts) + "."

    def _tag(self, glosses: list[str]) -> list[str]:
        return [_ROLE_MAP.get(g, "UNKNOWN") for g in glosses]

    def _match_template(
        self, roles: list[str], glosses: list[str]
    ) -> str | None:
        pattern = tuple(roles)
        for tmpl_pattern, tmpl_str in self._multi:
            if pattern == tmpl_pattern:
                resolved = [self._single_gloss(g).rstrip(".!?") for g in glosses]
                return tmpl_str % tuple(resolved)
        return None
