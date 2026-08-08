"""Detect near-duplicate gloss frames via hash and offset comparison (Closes #246).

Detects pairs of gloss files that share near-identical frame sequences by computing
fingerprint hashes per frame and identifying files with high similarity ratios.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple


def frame_fingerprint(frame: dict) -> str:
    """Compute a stable hash fingerprint for a single gloss frame."""
    key_fields = sorted(frame.keys())
    canonical = json.dumps({k: frame[k] for k in key_fields if k in frame}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_gloss_frames(path: Path) -> List[dict]:
    """Load frames from a gloss JSON file. Expects a top-level 'frames' array."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and 'frames' in data:
        return data['frames']
    if isinstance(data, list):
        return data
    return []


def build_file_fingerprint_set(frames: List[dict]) -> set:
    """Build a set of frame fingerprints for a file."""
    return {frame_fingerprint(f) for f in frames}


def detect_near_duplicates(
    gloss_dir: Path,
    *,
    threshold: float = 0.70,
) -> List[Tuple[str, str, float, int, int]]:
    """Detect pairs of gloss files with near-duplicate frame sequences.

    Returns list of (file_a, file_b, similarity_ratio, shared_frames, max_len) tuples.
    """
    files = sorted(gloss_dir.glob("*.json"))
    if len(files) < 2:
        return []

    file_fingerprints: Dict[str, set] = {}
    file_frame_counts: Dict[str, int] = {}

    for fpath in files:
        frames = load_gloss_frames(fpath)
        if not frames:
            continue
        file_fingerprints[str(fpath)] = build_file_fingerprint_set(frames)
        file_frame_counts[str(fpath)] = len(frames)

    duplicates: List[Tuple[str, str, float, int, int]] = []

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            fa, fb = str(files[i]), str(files[j])
            fps_a = file_fingerprints.get(fa, set())
            fps_b = file_fingerprints.get(fb, set())
            if not fps_a or not fps_b:
                continue

            shared = fps_a & fps_b
            max_len = max(len(fps_a), len(fps_b))
            if max_len == 0:
                continue

            ratio = len(shared) / max_len
            if ratio >= threshold:
                duplicates.append((
                    files[i].name, files[j].name,
                    round(ratio, 3),
                    len(shared), max_len
                ))

    return duplicates


def format_duplicate_report(
    duplicates: List[Tuple[str, str, float, int, int]],
) -> str:
    """Format duplicate detection results as a readable report."""
    if not duplicates:
        return "No near-duplicate gloss files detected."

    lines = ["Near-duplicate gloss frame detection report:", "=" * 60]
    for file_a, file_b, ratio, shared, total in duplicates:
        lines.append(
            "  {} <-> {} : {:.1%} similarity ({} shared / {} max frames)".format(
                file_a, file_b, ratio, shared, total
            )
        )
    lines.append("=" * 60)
    lines.append("Total near-duplicate pairs: {}".format(len(duplicates)))
    return "\n".join(lines)


def detect_duplicates_cli(
    gloss_dir: str = "data/sign-packs",
    threshold: float = 0.70,
    *,
    verbose: bool = False,
) -> int:
    """CLI entry point for duplicate frame detection.

    Returns exit code: 0 = no issues, 1 = duplicates found.
    """
    path = Path(gloss_dir)
    if not path.is_dir():
        print("Error: directory not found: {}".format(gloss_dir))
        return 2

    duplicates = detect_near_duplicates(path, threshold=threshold)
    report = format_duplicate_report(duplicates)

    if verbose or duplicates:
        print(report)

    return 1 if duplicates else 0
