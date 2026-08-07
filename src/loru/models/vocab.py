from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Runnable demo gloss vocabulary (isolated signs with bundled synthetic samples).
DEFAULT_GLOSS = [
    "hello",
    "thanks",
    "yes",
    "no",
    "help",
    "please",
    "love",
    "name",
    "water",
    "good",
    "goodbye",
    "sorry",
    "stop",
    "want",
    "need",
    "happy",
    "sad",
    "mother",
    "father",
    "friend",
    "eat_food",
    "drink",
    "home",
    "school",
    "go",
    "come",
    "see",
    "know",
    "big",
    "small",
    "welcome",
    "maybe",
    "wait",
    "today",
    "understand",
    "again",
    "more",
    "finish",
    "what",
    "where",
    "how",
    "why",
    "family",
    "work",
    "later",
    "tomorrow",
    "yesterday",
    "outside",    "inside",
    "night",
    "morning",
    "afternoon",
    "evening",
    "soon",
    "always",
    "never",
    "sometimes",
    "thank_you",
    "fingerspell_z",
    "fingerspell_y",
    "fingerspell_x",
    "fingerspell_w",
    "fingerspell_v",
    "fingerspell_u",
    "fingerspell_t",
    "fingerspell_s",
    "fingerspell_r",
    "fingerspell_q",
    "fingerspell_p",
    "fingerspell_o",
    "fingerspell_n",
    "fingerspell_m",
    "fingerspell_l",
    "fingerspell_k",
    "fingerspell_j",
    "fingerspell_i",
    "fingerspell_h",
    "fingerspell_g",
    "fingerspell_f",
    "fingerspell_e",
    "fingerspell_d",
    "fingerspell_c",
    "fingerspell_b",
    "fingerspell_a",
]

VOCAB_PATH = Path("data/vocab.json")


def _try_load_yaml(path: Path) -> list[str] | None:
    """Try to load a YAML vocab file. Returns None if YAML not available."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip().lower() for x in data]
    except Exception:
        pass
    return None


def load_vocab(path: Optional[Path | str] = None) -> list[str]:
    """Load gloss vocabulary from a JSON or YAML file.

    If the file exists and contains a list of strings, return it.
    Falls back to DEFAULT_GLOSS if the file is missing or invalid.
    """
    target = Path(path) if path else VOCAB_PATH
    if not target.exists():
        return DEFAULT_GLOSS

    # Try JSON first
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip().lower() for x in data]
    except (json.JSONDecodeError, OSError):
        pass

    # Try YAML
    yaml_result = _try_load_yaml(target)
    if yaml_result is not None:
        return yaml_result

    return DEFAULT_GLOSS


def save_vocab(glosses: list[str], path: Optional[Path | str] = None) -> Path:
    """Save gloss vocabulary to a JSON file. Returns the path written."""
    target = Path(path) if path else VOCAB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [g.strip().lower() for g in glosses if g.strip()]
    with open(target, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    return target


def _active_vocab() -> list[str]:
    """Return the currently active vocabulary (loaded from file or default)."""
    return load_vocab()


def gloss_to_id(gloss: str) -> int:
    key = gloss.strip().lower()
    vocab = _active_vocab()
    if key not in vocab:
        raise KeyError(f"unknown gloss {gloss!r}; known count={len(vocab)}")
    return vocab.index(key)


def id_to_gloss(idx: int) -> str:
    vocab = _active_vocab()
    if idx < 0 or idx >= len(vocab):
        return "unknown"
    return vocab[idx]
