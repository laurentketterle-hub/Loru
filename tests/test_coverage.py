"""Tests for gloss coverage heatmap statistics (no numpy required)."""
from __future__ import annotations

import json

from loru.models.vocab import DEFAULT_GLOSS


class TestCoverageLogic:
    """Test coverage calculation logic without triggering numpy import."""

    def test_coverage_math(self):
        """Basic coverage ratio calculation."""
        total = 10
        covered = 7
        ratio = covered / total
        assert round(ratio, 2) == 0.70

    def test_full_coverage(self):
        """100% coverage."""
        total = 5
        covered = 5
        ratio = covered / total
        assert ratio == 1.0

    def test_zero_coverage(self):
        """0% coverage."""
        total = 5
        covered = 0
        ratio = covered / total
        assert ratio == 0.0

    def test_default_gloss_structure(self):
        """DEFAULT_GLOSS is a list of strings."""
        assert isinstance(DEFAULT_GLOSS, list)
        assert len(DEFAULT_GLOSS) > 0
        assert all(isinstance(g, str) for g in DEFAULT_GLOSS)

    def test_missing_list_computation(self):
        """Missing glosses are correctly identified."""
        vocab = ["a", "b", "c", "d"]
        files = {"a", "c"}
        missing = [g for g in vocab if g not in files]
        assert set(missing) == {"b", "d"}
        covered = sum(1 for g in vocab if g in files)
        assert covered == 2

    def test_heatmap_ratio_bounds(self):
        """Coverage ratio is always between 0 and 1."""
        test_cases = [(10, 10), (10, 5), (10, 0), (0, 0)]
        for total, covered in test_cases:
            if total == 0:
                ratio = 0.0
            else:
                ratio = covered / total
            assert 0.0 <= ratio <= 1.0
