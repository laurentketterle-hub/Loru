from __future__ import annotations

"""Near-duplicate gloss frame detector.

Detects when two gloss files share near-identical frame sequences by
hashing each frame and comparing the hash sequences. Reports files,
overlap ratio, and frame offsets where duplication occurs.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


def _hash_frame(frame: list[Any]) -> str:
    """Hash a single frame (list of keypoints) to a stable hex digest."""
    raw = json.dumps(frame, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def hash_sequence(frames: list[list[Any]]) -> list[str]:
    """Return per-frame SHA-256 hashes for a frame sequence."""
    return [_hash_frame(f) for f in frames]


def load_gloss_file(path: Path) -> dict[str, Any]:
    """Load a gloss JSON file, returning gloss metadata and frames."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def compare_hashes(
    hashes_a: list[str],
    hashes_b: list[str],
) -> dict[str, Any]:
    """Compare two hash sequences and report overlap.

    Returns:
        dict with:
        - identical_count: number of frames with identical hash
        - overlap_ratio: identical_count / max(len(a), len(b))
        - offsets_a: frame indices in sequence A where match occurs
        - offsets_b: frame indices in sequence B where match occurs
    """
    len_a, len_b = len(hashes_a), len(hashes_b)
    max_len = max(len_a, len_b)

    if max_len == 0:
        return {
            "identical_count": 0,
            "overlap_ratio": 0.0,
            "offsets_a": [],
            "offsets_b": [],
        }

    # Build index of hash -> positions in B
    b_index: dict[str, list[int]] = {}
    for idx, h in enumerate(hashes_b):
        b_index.setdefault(h, []).append(idx)

    offsets_a: list[int] = []
    offsets_b: list[int] = []
    used_b: set[int] = set()

    for idx_a, h in enumerate(hashes_a):
        positions = b_index.get(h, [])
        for pos_b in positions:
            if pos_b not in used_b:
                offsets_a.append(idx_a)
                offsets_b.append(pos_b)
                used_b.add(pos_b)
                break

    identical = len(offsets_a)
    return {
        "identical_count": identical,
        "overlap_ratio": identical / max_len,
        "offsets_a": offsets_a,
        "offsets_b": offsets_b,
    }


def scan_directory(
    directory: Path,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Scan a directory of gloss JSON files for near-duplicates.

    Args:
        directory: Path containing *.json gloss files
        threshold: Minimum overlap ratio to report (0.0-1.0)

    Returns:
        List of duplicate reports, each with file_a, file_b, and comparison data.
    """
    json_files = sorted(directory.glob("*.json"))
    reports: list[dict[str, Any]] = []

    # Pre-load and hash all files
    file_data: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for fpath in json_files:
        try:
            data = load_gloss_file(fpath)
            frames = data.get("frames", [])
            hashes = hash_sequence(frames)
            file_data[str(fpath)] = (data, hashes)
        except (json.JSONDecodeError, OSError):
            continue

    paths = list(file_data.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            data_a, hashes_a = file_data[paths[i]]
            data_b, hashes_b = file_data[paths[j]]
            cmp = compare_hashes(hashes_a, hashes_b)
            if cmp["overlap_ratio"] >= threshold:
                reports.append(
                    {
                        "file_a": paths[i],
                        "gloss_a": data_a.get("gloss", "?"),
                        "language_a": data_a.get("language", "?"),
                        "file_b": paths[j],
                        "gloss_b": data_b.get("gloss", "?"),
                        "language_b": data_b.get("language", "?"),
                        **cmp,
                    }
                )

    # Sort by overlap ratio descending
    reports.sort(key=lambda r: r["overlap_ratio"], reverse=True)
    return reports


def scan_recursive(
    root: Path,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Recursively scan all subdirectories for near-duplicate gloss files.

    Each subdirectory is scanned independently (cross-directory comparisons
    are included).
    """
    all_files = sorted(root.rglob("*.json"))
    if not all_files:
        return []

    file_data: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for fpath in all_files:
        try:
            data = load_gloss_file(fpath)
            frames = data.get("frames", [])
            hashes = hash_sequence(frames)
            file_data[str(fpath)] = (data, hashes)
        except (json.JSONDecodeError, OSError):
            continue

    paths = list(file_data.keys())
    reports: list[dict[str, Any]] = []

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            data_a, hashes_a = file_data[paths[i]]
            data_b, hashes_b = file_data[paths[j]]
            cmp = compare_hashes(hashes_a, hashes_b)
            if cmp["overlap_ratio"] >= threshold:
                reports.append(
                    {
                        "file_a": paths[i],
                        "gloss_a": data_a.get("gloss", "?"),
                        "language_a": data_a.get("language", "?"),
                        "file_b": paths[j],
                        "gloss_b": data_b.get("gloss", "?"),
                        "language_b": data_b.get("language", "?"),
                        **cmp,
                    }
                )

    reports.sort(key=lambda r: r["overlap_ratio"], reverse=True)
    return reports
