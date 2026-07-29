"""Tests for near-duplicate gloss frame detector."""

from __future__ import annotations

import json
from pathlib import Path

from loru.data.near_dup import (
    _frame_distance,
    _sequence_similarity,
    _best_offset_similarity,
    _flatten_landmarks,
    find_near_duplicates,
)


# --- Unit tests for internal functions ---


def test_frame_distance_identical() -> None:
    """Identical landmark vectors produce zero distance."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert _frame_distance(a, b) == 0.0


def test_frame_distance_different() -> None:
    """Different vectors produce positive distance."""
    a = [0.0, 0.0, 0.0]
    b = [1.0, 1.0, 1.0]
    d = _frame_distance(a, b)
    assert d > 0.0


def test_frame_distance_empty() -> None:
    """Empty vectors return infinite distance."""
    assert _frame_distance([], []) == float("inf")
    assert _frame_distance([1.0, 2.0], []) == float("inf")
    assert _frame_distance([], [1.0, 2.0]) == float("inf")


def test_frame_distance_mixed_lengths() -> None:
    """Vectors of different lengths compare their overlapping prefix."""
    a = [0.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0, 9.0, 9.0]
    assert _frame_distance(a, b) == 0.0


def test_frame_distance_normalized() -> None:
    """Distance is normalized by sqrt(dim), so values are bounded."""
    a = [0.0] * 30
    b = [1.0] * 30
    d = _frame_distance(a, b)
    # sqrt(30 * 1^2) / sqrt(30) = 1.0
    assert abs(d - 1.0) < 0.001


def test_flatten_landmarks() -> None:
    """flatten_landmarks converts [[x,y,z], ...] into [x,y,z, ...]."""
    landmarks = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = _flatten_landmarks(landmarks)
    assert result == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_sequence_similarity_identical() -> None:
    """Identical frame sequences have similarity 1.0."""
    frames = [[[0.5, 0.5, 0.0], [0.6, 0.4, 0.0]], [[0.55, 0.45, 0.0], [0.65, 0.35, 0.0]]]
    sim = _sequence_similarity(frames, frames)
    assert sim == 1.0


def test_sequence_similarity_different() -> None:
    """Very different sequences have low similarity."""
    a = [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]
    b = [[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]]
    sim = _sequence_similarity(a, b)
    # exp(-1.0) ≈ 0.368
    assert 0.36 < sim < 0.38


def test_sequence_similarity_range() -> None:
    """Similarity is always in [0, 1]."""
    a = [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]
    b = [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]
    sim = _sequence_similarity(a, b)
    assert 0.0 <= sim <= 1.0

    c = [[[9.0, 9.0, 9.0]], [[8.0, 8.0, 8.0]]]
    sim2 = _sequence_similarity(a, c)
    assert 0.0 <= sim2 <= 1.0


def test_best_offset_similarity_no_offset() -> None:
    """Identical sequences found with similarity=1.0 at offset=0."""
    frames = [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]]
    sim, offset = _best_offset_similarity(frames, frames, max_offset=3)
    assert sim == 1.0
    assert offset == 0


def test_best_offset_similarity_shifted() -> None:
    """Shifted duplicate subsequence detected at correct offset.

    seq_a = [A, B, C, D, E]
    seq_b = [X, Y, A, B, C, D]
    Overlap A,B,C,D when offset=-2 (seq_b shifted right by 2).
    """
    A = [[1.0, 0.0, 0.0]]
    B = [[0.0, 1.0, 0.0]]
    C = [[1.0, 1.0, 0.0]]
    D = [[0.5, 0.5, 0.0]]
    E = [[0.0, 0.0, 1.0]]
    X = [[9.0, 9.0, 9.0]]
    Y = [[8.0, 8.0, 8.0]]

    seq_a = [A, B, C, D, E]
    seq_b = [X, Y, A, B, C, D]
    sim, offset = _best_offset_similarity(seq_a, seq_b, max_offset=5)
    # The overlapping part [A,B,C,D] is identical, so sim should be 1.0
    assert sim == 1.0
    assert offset == -2


# --- Integration tests with temporary directories ---


def test_find_near_duplicates_empty_dir(tmp_path: Path) -> None:
    """Empty directory returns empty list."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert find_near_duplicates(samples_dir=empty) == []


def test_find_near_duplicates_no_dups(tmp_path: Path) -> None:
    """No duplicates when all glosses are clearly different."""
    files = [
        {
            "gloss": "hello",
            "language": "vsl",
            "frames": [[[0.5, 0.5, 0.0], [0.6, 0.4, 0.0]]],
        },
        {
            "gloss": "goodbye",
            "language": "vsl",
            "frames": [[[0.9, 0.9, 0.0], [0.8, 0.8, 0.0], [0.7, 0.7, 0.0]]],
        },
    ]
    for f in files:
        (tmp_path / f"{f['gloss']}.json").write_text(
            json.dumps(f), encoding="utf-8"
        )
    result = find_near_duplicates(threshold=0.95, samples_dir=tmp_path)
    assert result == []


def test_find_near_duplicates_found(tmp_path: Path) -> None:
    """Near-duplicate glosses are detected at high similarity."""
    identical_frames = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
    ]
    files = [
        {"gloss": "hello", "language": "vsl", "frames": identical_frames},
        {"gloss": "hi", "language": "vsl", "frames": identical_frames},
        {"gloss": "world", "language": "vsl", "frames": [[[9.0, 9.0, 0.0]]]},
    ]
    for f in files:
        (tmp_path / f"{f['gloss']}.json").write_text(
            json.dumps(f), encoding="utf-8"
        )
    result = find_near_duplicates(threshold=0.95, samples_dir=tmp_path)
    assert len(result) >= 1
    pair = result[0]
    assert pair["similarity"] >= 0.95
    assert {pair["gloss_a"], pair["gloss_b"]} == {"hello", "hi"}


def test_find_near_duplicates_synthetic_clones(tmp_path: Path) -> None:
    """Synthetic clones with slight noise are detected with appropriate threshold."""
    base = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
    ]
    noisy = [
        [[1.01, 0.0, 0.0], [0.0, 0.99, 0.0]],
        [[1.0, 1.02, 0.0], [0.49, 0.5, 0.0]],
        [[0.0, 0.01, 1.0], [0.99, 0.0, 1.01]],
    ]
    files = [
        {"gloss": "original", "language": "vsl", "frames": base},
        {"gloss": "clone_noisy", "language": "vsl", "frames": noisy},
        {"gloss": "different", "language": "vsl", "frames": [[[9.0, 9.0, 0.0]]]},
    ]
    for f in files:
        (tmp_path / f"{f['gloss']}.json").write_text(
            json.dumps(f), encoding="utf-8"
        )
    # Low-ish threshold catches the noisy clone
    result = find_near_duplicates(threshold=0.8, samples_dir=tmp_path)
    assert len(result) >= 1
    glosses = {result[0]["gloss_a"], result[0]["gloss_b"]}
    assert "original" in glosses
    assert "clone_noisy" in glosses


def test_find_near_duplicates_nested_dirs(tmp_path: Path) -> None:
    """Detector scans subdirectories recursively."""
    sub = tmp_path / "vsl"
    sub.mkdir()
    identical_frames = [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]
    (tmp_path / "a.json").write_text(
        json.dumps({"gloss": "alpha", "frames": identical_frames}), encoding="utf-8"
    )
    (sub / "b.json").write_text(
        json.dumps({"gloss": "beta", "frames": identical_frames}), encoding="utf-8"
    )
    result = find_near_duplicates(threshold=0.95, samples_dir=tmp_path)
    assert len(result) >= 1
    assert {result[0]["gloss_a"], result[0]["gloss_b"]} == {"alpha", "beta"}


def test_find_near_duplicates_ignores_bad_files(tmp_path: Path) -> None:
    """Non-JSON and empty-frame files are silently skipped."""
    (tmp_path / "bad.txt").write_text("not json", encoding="utf-8")
    (tmp_path / "empty.json").write_text(
        json.dumps({"gloss": "empty", "frames": []}), encoding="utf-8"
    )
    (tmp_path / "good.json").write_text(
        json.dumps({"gloss": "good", "frames": [[[0.5, 0.5, 0.0]]]}),
        encoding="utf-8",
    )
    result = find_near_duplicates(samples_dir=tmp_path)
    assert result == []  # Only one valid gloss, can't form pairs


def test_find_near_duplicates_respects_threshold(tmp_path: Path) -> None:
    """High threshold excludes moderately different pairs."""
    frames_a = [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]
    frames_b = [[[1.0, 0.0, 0.0]], [[0.0, 0.5, 0.0]]]  # Slightly different
    files = [
        {"gloss": "x", "frames": frames_a},
        {"gloss": "y", "frames": frames_b},
    ]
    for f in files:
        (tmp_path / f"{f['gloss']}.json").write_text(
            json.dumps(f), encoding="utf-8"
        )
    # Very high threshold should exclude this
    result = find_near_duplicates(threshold=0.999, samples_dir=tmp_path)
    assert result == []
    # Lower threshold should catch it
    result2 = find_near_duplicates(threshold=0.5, samples_dir=tmp_path)
    assert len(result2) == 1
