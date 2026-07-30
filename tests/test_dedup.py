from __future__ import annotations

"""Unit tests for loru.tools.dedup — near-duplicate gloss frame detector."""

import json
from pathlib import Path

import pytest

from loru.tools.dedup import (
    _hash_frame,
    compare_hashes,
    hash_sequence,
    load_gloss_file,
    scan_directory,
    scan_recursive,
)


# ── synthetic helpers ──────────────────────────────────────────────

def _gloss_json(tmp_path: Path, name: str, gloss: str, language: str,
                frames: list) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({
        "gloss": gloss,
        "language": language,
        "fps": 15,
        "frames": frames,
    }), encoding="utf-8")
    return path


def _frame(values: list[float]) -> list[list[float]]:
    return [values]


# ── hash_sequence ──────────────────────────────────────────────────

def test_hash_sequence_empty() -> None:
    assert hash_sequence([]) == []


def test_hash_sequence_deterministic() -> None:
    frames = [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]
    h1 = hash_sequence(frames)
    h2 = hash_sequence(frames)
    assert h1 == h2


def test_hash_sequence_different_frames() -> None:
    h1 = hash_sequence([[[1.0, 0.0, 0.0]]])
    h2 = hash_sequence([[[2.0, 0.0, 0.0]]])
    assert h1 != h2


# ── compare_hashes ─────────────────────────────────────────────────

def test_compare_identical() -> None:
    hashes = ["a", "b", "c"]
    result = compare_hashes(hashes, hashes)
    assert result["identical_count"] == 3
    assert result["overlap_ratio"] == 1.0
    assert result["offsets_a"] == [0, 1, 2]
    assert result["offsets_b"] == [0, 1, 2]


def test_compare_no_overlap() -> None:
    result = compare_hashes(["a", "b"], ["c", "d"])
    assert result["identical_count"] == 0
    assert result["overlap_ratio"] == 0.0


def test_compare_partial_overlap() -> None:
    result = compare_hashes(["a", "b", "c"], ["b", "c", "d"])
    assert result["identical_count"] == 2
    assert result["overlap_ratio"] == 2 / 3


def test_compare_empty() -> None:
    result = compare_hashes([], [])
    assert result["identical_count"] == 0
    assert result["overlap_ratio"] == 0.0


# ── scan_directory ─────────────────────────────────────────────────

def test_scan_identical_clones(tmp_path: Path) -> None:
    """Two files with identical frames → reported as duplicate."""
    frames = [[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]]
    _gloss_json(tmp_path, "a.json", "hello", "asl", frames)
    _gloss_json(tmp_path, "b.json", "hello", "demo-asl", frames)

    reports = scan_directory(tmp_path, threshold=0.5)
    assert len(reports) == 1
    r = reports[0]
    assert r["overlap_ratio"] == 1.0
    assert r["identical_count"] == 2


def test_scan_no_duplicates(tmp_path: Path) -> None:
    """Completely different frames → no duplicates reported."""
    _gloss_json(tmp_path, "a.json", "hello", "asl",
                [[[0.1, 0.0, 0.0]], [[0.2, 0.0, 0.0]]])
    _gloss_json(tmp_path, "b.json", "bye", "asl",
                [[[0.9, 0.0, 0.0]], [[0.8, 0.0, 0.0]]])

    reports = scan_directory(tmp_path, threshold=0.5)
    assert len(reports) == 0


def test_scan_below_threshold(tmp_path: Path) -> None:
    """Partial overlap below threshold → not reported."""
    frames_a = [[[float(i), 0.0, 0.0]] for i in range(10)]
    frames_b = [[[float(i), 0.0, 0.0]] for i in range(2)] + \
               [[[float(i + 100), 0.0, 0.0]] for i in range(8)]
    _gloss_json(tmp_path, "a.json", "a", "x", frames_a)
    _gloss_json(tmp_path, "b.json", "b", "x", frames_b)

    reports = scan_directory(tmp_path, threshold=0.8)
    assert len(reports) == 0


def test_scan_synthetic_near_duplicate(tmp_path: Path) -> None:
    """Two gloss files sharing 80% of frames → detected at 0.7 threshold."""
    frames_shared = [[[float(i), 0.0, 0.0]] for i in range(8)]
    frames_a = frames_shared + [[[100.0, 0.0, 0.0]], [[101.0, 0.0, 0.0]]]
    frames_b = frames_shared + [[[200.0, 0.0, 0.0]], [[201.0, 0.0, 0.0]]]

    _gloss_json(tmp_path, "hello.json", "hello", "asl", frames_a)
    _gloss_json(tmp_path, "hello_demo.json", "hello", "demo-asl", frames_b)

    reports = scan_directory(tmp_path, threshold=0.7)
    assert len(reports) == 1
    r = reports[0]
    assert r["overlap_ratio"] == 0.8
    assert r["identical_count"] == 8


def test_scan_skips_invalid_json(tmp_path: Path) -> None:
    """Non-JSON files are silently skipped."""
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    _gloss_json(tmp_path, "ok.json", "ok", "x", [[[0.0, 0.0, 0.0]]])
    reports = scan_directory(tmp_path)
    assert len(reports) == 0  # only one valid file, no pair to compare


def test_scan_single_file_no_pairs(tmp_path: Path) -> None:
    """Single file → no duplicates possible."""
    _gloss_json(tmp_path, "only.json", "only", "x", [[[0.0, 0.0, 0.0]]])
    reports = scan_directory(tmp_path)
    assert len(reports) == 0


# ── scan_recursive ─────────────────────────────────────────────────

def test_scan_recursive_cross_directory(tmp_path: Path) -> None:
    """Detects near-duplicates across subdirectories."""
    frames = [[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]]
    sub_a = tmp_path / "asl"
    sub_b = tmp_path / "demo"
    sub_a.mkdir()
    sub_b.mkdir()
    _gloss_json(sub_a, "hello.json", "hello", "asl", frames)
    _gloss_json(sub_b, "hello.json", "hello", "demo-asl", frames)

    reports = scan_recursive(tmp_path, threshold=0.5)
    assert len(reports) == 1
    r = reports[0]
    assert r["overlap_ratio"] == 1.0


def test_scan_recursive_empty(tmp_path: Path) -> None:
    reports = scan_recursive(tmp_path)
    assert reports == []


# ── load_gloss_file ────────────────────────────────────────────────

def test_load_gloss_file_returns_metadata(tmp_path: Path) -> None:
    path = _gloss_json(tmp_path, "test.json", "hello", "asl",
                       [[[0.0, 0.0, 0.0]]])
    data = load_gloss_file(path)
    assert data["gloss"] == "hello"
    assert data["language"] == "asl"
    assert len(data["frames"]) == 1


# ── real-world: hello.json variants ────────────────────────────────

def test_real_hello_variants_are_near_duplicates(tmp_path: Path) -> None:
    """Simulates the real data/samples/hello.json vs data/samples/asl/hello.json
    pattern where all hand keypoints (indices 1-19) are identical but the
    root/pelvis keypoint differs between frames."""
    import copy

    # Create 5 frames with 20 keypoints each (like real data)
    def make_frames(root_offsets):
        frames = []
        for i, offset in enumerate(root_offsets):
            frame = []
            # Keypoint 0: root (varies)
            frame.append([0.9 + offset, 0.4, 0.8])
            # Keypoints 1-19: identical hand keypoints
            for k in range(19):
                frame.append([0.0129 * (k + 1), 0.0258 * (k + 1), -0.0258 * (k + 1)])
            frames.append(frame)
        return frames

    frames_a = make_frames([0.0, 0.01, 0.02, 0.03, 0.04])
    frames_b = make_frames([0.0, 0.01, 0.02, 0.03, 0.04])  # identical root too

    _gloss_json(tmp_path, "hello.json", "hello", "demo-asl", frames_a)
    _gloss_json(tmp_path, "hello_asl.json", "hello", "asl", frames_b)

    reports = scan_directory(tmp_path, threshold=0.9)
    assert len(reports) == 1
    assert reports[0]["overlap_ratio"] == 1.0
