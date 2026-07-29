"""
Near-duplicate gloss frame detector for VSL sign packs.

Scans gloss JSON files in data/samples/ and reports pairs whose frame
sequences are suspiciously similar. Uses Euclidean distance on landmark
vectors with sliding-offset alignment to catch shifted copies.

Usage:
    python -m loru.data.near_dup [--threshold 0.05] [--max-offset 5] [--json]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "samples"


def _load_gloss_file(path: Path) -> dict[str, Any] | None:
    """Load a single gloss JSON and return a normalized dict, or None on failure."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return {
        "path": str(path),
        "gloss": raw.get("gloss", path.stem),
        "language": raw.get("language", "?"),
        "fps": raw.get("fps", 15),
        "frames": raw.get("frames", []),
    }


def _flatten_landmarks(landmark_list: list[list[float]]) -> list[float]:
    """Flatten a list of [x,y,z] vectors into a single list of floats."""
    flat: list[float] = []
    for lm in landmark_list:
        if isinstance(lm, list):
            flat.extend(lm)
    return flat


def _frame_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Euclidean distance between two landmark vectors (padded/truncated to min length).

    Normalized by sqrt(dim) so distances are comparable across different frame sizes.
    """
    n = min(len(vec_a), len(vec_b))
    if n == 0:
        return float("inf")
    pairs = list(zip(vec_a[:n], vec_b[:n]))
    raw = math.sqrt(sum((x - y) ** 2 for x, y in pairs))
    return raw / math.sqrt(n)


def _sequence_similarity(
    seq_a: list[list[float]], seq_b: list[list[float]]
) -> float:
    """Average frame similarity (1.0 = identical, 0.0 = completely different).

    Converts Euclidean distance to a similarity score in [0, 1].
    """
    flat_a = [_flatten_landmarks(f) for f in seq_a]
    flat_b = [_flatten_landmarks(f) for f in seq_b]

    n = min(len(flat_a), len(flat_b))
    if n == 0:
        return 0.0

    total_dist = 0.0
    for i in range(n):
        total_dist += _frame_distance(flat_a[i], flat_b[i])

    avg_dist = total_dist / n
    # Convert distance to similarity: exp(-dist) maps [0, inf) -> (0, 1]
    return math.exp(-avg_dist)


def _best_offset_similarity(
    seq_a: list[list[float]],
    seq_b: list[list[float]],
    max_offset: int = 5,
) -> tuple[float, int]:
    """Find the best similarity score by sliding seq_b over seq_a.

    Returns (best_similarity, offset) where offset > 0 means seq_b is shifted
    right relative to seq_a, and offset < 0 means shifted left.
    A similarity of 1.0 means identical.
    """
    best_sim = 0.0
    best_offset = 0
    for offset in range(-max_offset, max_offset + 1):
        if offset >= 0:
            a_slice = seq_a[offset:]
            b_slice = seq_b[: len(seq_a) - offset]
        else:
            a_slice = seq_a[: len(seq_b) + offset]
            b_slice = seq_b[-offset:]

        overlap = min(len(a_slice), len(b_slice))
        if overlap < 2:
            continue

        sim = _sequence_similarity(a_slice[:overlap], b_slice[:overlap])
        if sim > best_sim:
            best_sim = sim
            best_offset = offset

    return best_sim, best_offset


def find_near_duplicates(
    threshold: float = 0.85,
    max_offset: int = 5,
    samples_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Scan gloss files and return near-duplicate pairs above similarity threshold.

    Args:
        threshold: Minimum similarity score (0.0-1.0) to flag as duplicate.
                   Higher values = stricter matching. Default 0.85.
        max_offset: Max frame offset for sliding alignment.
        samples_dir: Directory to scan recursively for .json gloss files.

    Returns:
        List of duplicate reports sorted by similarity (highest first).
        Each report contains gloss_a, gloss_b, similarity, offset_frames, etc.
    """
    samples_dir = samples_dir or SAMPLES_DIR
    if not samples_dir.is_dir():
        return []

    # Recursively find all JSON files
    gloss_files = sorted(samples_dir.rglob("*.json"))
    glosses: list[dict[str, Any]] = []
    for p in gloss_files:
        if p.is_file():
            g = _load_gloss_file(p)
            if g and g.get("frames"):
                glosses.append(g)

    if len(glosses) < 2:
        return []

    duplicates: list[dict[str, Any]] = []
    checked: set[tuple[str, str]] = set()

    total = len(glosses) * (len(glosses) - 1) // 2
    compared = 0

    for i in range(len(glosses)):
        for j in range(i + 1, len(glosses)):
            a = glosses[i]
            b = glosses[j]
            compared += 1

            key = (a["gloss"], b["gloss"])
            if key in checked:
                continue
            checked.add(key)

            frames_a: list = a["frames"]
            frames_b: list = b["frames"]

            if not frames_a or not frames_b:
                continue

            # Skip if frame count difference is too large (obviously different)
            if abs(len(frames_a) - len(frames_b)) > max_offset + 3:
                continue

            sim, offset = _best_offset_similarity(
                frames_a, frames_b, max_offset=max_offset
            )

            if sim >= threshold:
                duplicates.append(
                    {
                        "gloss_a": a["gloss"],
                        "path_a": a["path"],
                        "gloss_b": b["gloss"],
                        "path_b": b["path"],
                        "similarity": round(sim, 6),
                        "offset_frames": offset,
                        "frames_a": len(frames_a),
                        "frames_b": len(frames_b),
                    }
                )

    return sorted(duplicates, key=lambda x: x["similarity"], reverse=True)


def main() -> None:
    """CLI entry point: scan samples and report near-duplicate gloss frames."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect near-duplicate gloss frame sequences in sign pack data."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum similarity score (0.0-1.0) to flag. Default: 0.85",
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=5,
        help="Maximum frame offset for alignment. Default: 5",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (machine-readable)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Custom samples directory (default: data/samples/)",
    )
    args = parser.parse_args()

    samples_dir = Path(args.dir) if args.dir else None
    duplicates = find_near_duplicates(
        threshold=args.threshold,
        max_offset=args.max_offset,
        samples_dir=samples_dir,
    )

    if args.json:
        json.dump(
            {
                "threshold": args.threshold,
                "max_offset": args.max_offset,
                "duplicate_pairs": len(duplicates),
                "duplicates": duplicates,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        if not duplicates:
            print(
                f"No near-duplicate gloss frames found (threshold={args.threshold})."
            )
            sys.exit(0)

        print(
            f"Found {len(duplicates)} near-duplicate pair(s) "
            f"(threshold={args.threshold}):\n"
        )
        for d in duplicates:
            print(
                f"  {d['gloss_a']} <-> {d['gloss_b']}: "
                f"similarity={d['similarity']:.4f} offset={d['offset_frames']}f "
                f"(len={d['frames_a']}/{d['frames_b']})"
            )
            print(f"    {d['path_a']}")
            print(f"    {d['path_b']}")

        sys.exit(1)  # Non-zero exit signals duplicates found (CI-friendly)


if __name__ == "__main__":
    main()
