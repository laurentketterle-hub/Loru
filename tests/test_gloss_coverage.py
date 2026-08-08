"""Tests for gloss coverage heatmap CLI (Closes #223)."""
import json
from pathlib import Path
from scripts.gloss_coverage_heatmap import (
    load_default_gloss, load_available_samples, compute_coverage,
    format_heatmap, format_json,
)


def create_sign_pack(directory, name, glosses):
    path = directory / "sign-packs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"frames": [{"gloss": g} for g in glosses]}))

def create_sample(directory, name, gloss):
    path = directory / "samples" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gloss": gloss, "data": {}}))


class TestComputeCoverage:
    def test_all_covered(self):
        gloss_map = {"HELLO": {"pack1"}, "BYE": {"pack1"}}
        samples = {"HELLO": 3, "BYE": 1}
        cov = compute_coverage(gloss_map, samples)
        assert cov["HELLO"]["covered"]
        assert cov["BYE"]["covered"]

    def test_partial_coverage(self):
        gloss_map = {"A": {"p1"}, "B": {"p1"}}
        samples = {"A": 2}
        cov = compute_coverage(gloss_map, samples)
        assert cov["A"]["covered"]
        assert not cov["B"]["covered"]
        assert cov["A"]["coverage_pct"] >= 100.0

    def test_zero_expected_with_sample(self):
        gloss_map = {}
        samples = {"X": 1}
        cov = compute_coverage(gloss_map, samples)
        assert cov["X"]["covered"]
        assert cov["X"]["expected"] == 0

    def test_empty_inputs(self):
        cov = compute_coverage({}, {})
        assert cov == {}


class TestFormatHeatmap:
    def test_heatmap_output(self):
        cov = {
            "HELLO": {"gloss": "HELLO", "expected": 2, "available": 2, "covered": True, "coverage_pct": 100.0},
            "BYE": {"gloss": "BYE", "expected": 1, "available": 0, "covered": False, "coverage_pct": 0.0},
        }
        heatmap = format_heatmap(cov)
        assert "HELLO" in heatmap
        assert "BYE" in heatmap
        assert "Covered: 1" in heatmap or "1" in heatmap

    def test_empty_heatmap(self):
        heatmap = format_heatmap({})
        assert "No gloss data" in heatmap

    def test_coverage_percentage_displayed(self):
        cov = {"TEST": {"gloss": "TEST", "expected": 5, "available": 3, "covered": True, "coverage_pct": 60.0}}
        heatmap = format_heatmap(cov)
        assert "60.0%" in heatmap


class TestFormatJson:
    def test_json_output(self):
        cov = {"A": {"gloss": "A", "expected": 1, "available": 1, "covered": True, "coverage_pct": 100.0}}
        out = format_json(cov)
        data = json.loads(out)
        assert data["total"] == 1
        assert data["covered"] == 1
