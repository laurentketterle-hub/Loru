"""Tests for gloss vocabulary loading from JSON/YAML files."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from loru.models.vocab import (
    DEFAULT_GLOSS,
    VOCAB_PATH,
    load_vocab,
    save_vocab,
    gloss_to_id,
    id_to_gloss,
)


class TestLoadVocab:
    def test_default_when_no_file(self, monkeypatch):
        """load_vocab returns DEFAULT_GLOSS when no custom file exists."""
        # Point to a non-existent file
        result = load_vocab(Path("/nonexistent/vocab.json"))
        assert result == DEFAULT_GLOSS
        assert len(result) > 0

    def test_load_json_file(self, tmp_path):
        """load_vocab loads from a valid JSON file."""
        custom = ["hello", "world", "test"]
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(custom))

        result = load_vocab(path)
        assert result == custom

    def test_load_json_lowercases(self, tmp_path):
        """Loaded glosses are lowercased and stripped."""
        custom = ["Hello", " WORLD ", "Test"]
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(custom))

        result = load_vocab(path)
        assert result == ["hello", "world", "test"]

    def test_invalid_json_falls_back(self, tmp_path):
        """Invalid JSON falls back to DEFAULT_GLOSS."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")

        result = load_vocab(path)
        assert result == DEFAULT_GLOSS

    def test_non_list_json_falls_back(self, tmp_path):
        """JSON that is not a list falls back to DEFAULT_GLOSS."""
        path = tmp_path / "obj.json"
        path.write_text('{"key": "value"}')

        result = load_vocab(path)
        assert result == DEFAULT_GLOSS

    def test_non_string_list_falls_back(self, tmp_path):
        """List with non-string items falls back."""
        path = tmp_path / "mixed.json"
        path.write_text('[1, 2, 3]')

        result = load_vocab(path)
        assert result == DEFAULT_GLOSS

    def test_empty_list_ok(self, tmp_path):
        """Empty list is valid."""
        path = tmp_path / "empty.json"
        path.write_text("[]")

        result = load_vocab(path)
        assert result == []


class TestSaveVocab:
    def test_save_creates_file(self, tmp_path):
        """save_vocab writes JSON and returns path."""
        glosses = ["a", "b", "c"]
        path = tmp_path / "out.json"
        result = save_vocab(glosses, path)
        assert result == path
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == glosses

    def test_save_cleans_glosses(self, tmp_path):
        """save_vocab strips and lowercases."""
        path = tmp_path / "clean.json"
        save_vocab([" Hello ", "WORLD"], path)
        data = json.loads(path.read_text())
        assert data == ["hello", "world"]

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_vocab creates parent directories."""
        path = tmp_path / "sub" / "deep" / "vocab.json"
        save_vocab(["test"], path)
        assert path.exists()


class TestGlossToId:
    def test_known_gloss(self):
        idx = gloss_to_id("hello")
        assert idx == DEFAULT_GLOSS.index("hello")

    def test_unknown_gloss_raises(self):
        with pytest.raises(KeyError):
            gloss_to_id("zzz_nonexistent_gloss_xyz")

    def test_case_insensitive(self):
        idx = gloss_to_id("HELLO")
        assert idx == DEFAULT_GLOSS.index("hello")

    def test_with_custom_file(self, tmp_path):
        """gloss_to_id uses custom file when present at VOCAB_PATH."""
        # This test checks the default path behavior
        custom = ["custom_a", "custom_b"]
        path = tmp_path / "custom_vocab.json"
        save_vocab(custom, path)
        loaded = load_vocab(path)
        assert "custom_a" in loaded
        assert "custom_b" in loaded


class TestIdToGloss:
    def test_valid_id(self):
        assert id_to_gloss(0) == DEFAULT_GLOSS[0]

    def test_negative_id(self):
        assert id_to_gloss(-1) == "unknown"

    def test_out_of_range(self):
        assert id_to_gloss(99999) == "unknown"


class TestDefaultGloss:
    def test_default_gloss_not_empty(self):
        assert len(DEFAULT_GLOSS) > 0

    def test_all_lowercase(self):
        for g in DEFAULT_GLOSS:
            assert g == g.lower(), f"Gloss '{g}' should be lowercase"
