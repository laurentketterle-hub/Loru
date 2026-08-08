"""Gloss coverage heatmap CLI — shows which DEFAULT_GLOSS entries lack samples (Closes #223)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_default_gloss() -> Dict[str, set]:
    """Load DEFAULT_GLOSS entries from sign-packs metadata. Returns {gloss_id: {expected_samples}}."""
    gloss_map = defaultdict(set)
    sign_packs_dir = DATA_DIR / "sign-packs"
    if not sign_packs_dir.is_dir():
        return dict(gloss_map)

    for fpath in sorted(sign_packs_dir.glob("*.json")):
        try:
            data = json.loads(fpath.read_text())
        except (json.JSONDecodeError, KeyError):
            continue

        frames = data.get("frames", [])
        for frame in frames:
            gloss_id = frame.get("gloss", frame.get("label", ""))
            if gloss_id:
                gloss_map[gloss_id].add(fpath.stem)

    return dict(gloss_map)


def load_available_samples() -> Dict[str, int]:
    """Load available sample files and count them per gloss."""
    samples = defaultdict(int)
    samples_dir = DATA_DIR / "samples"
    if not samples_dir.is_dir():
        return dict(samples)

    for fpath in sorted(samples_dir.glob("**/*.json")):
        try:
            data = json.loads(fpath.read_text())
        except (json.JSONDecodeError, KeyError):
            continue

        gloss_id = data.get("gloss", data.get("label", ""))
        if gloss_id:
            samples[gloss_id] += 1

    return dict(samples)


def compute_coverage(
    gloss_map: Dict[str, set],
    samples: Dict[str, int],
) -> Dict[str, dict]:
    """Compute coverage stats: covered, uncovered, sample count, expected count."""
    coverage = {}
    all_glosses = set(gloss_map.keys()) | set(samples.keys())

    for gloss_id in sorted(all_glosses):
        expected = len(gloss_map.get(gloss_id, set()))
        available = samples.get(gloss_id, 0)
        covered = available > 0
        percentage = round(available / max(expected, 1) * 100, 1) if expected > 0 else (100.0 if available > 0 else 0.0)

        coverage[gloss_id] = {
            "gloss": gloss_id,
            "expected": expected,
            "available": available,
            "covered": covered,
            "coverage_pct": min(percentage, 100.0),
        }

    return coverage


def format_heatmap(coverage: Dict[str, dict]) -> str:
    """Format coverage as a color-coded terminal heatmap."""
    if not coverage:
        return "No gloss data found."

    total = len(coverage)
    covered_count = sum(1 for v in coverage.values() if v["covered"])
    uncovered_count = total - covered_count

    lines = [
        "Gloss Coverage Heatmap",
        "=" * 60,
        "Total glosses: {} | Covered: {} | Uncovered: {}".format(total, covered_count, uncovered_count),
        "Coverage: {:.1%}".format(covered_count / max(total, 1)),
        "",
    ]

    # Color-coded bar per gloss (ASCII)
    BAR_WIDTH = 40
    for gloss_id, info in sorted(coverage.items()):
        pct = info["coverage_pct"]
        filled = int(pct / 100 * BAR_WIDTH)
        empty = BAR_WIDTH - filled
        bar = "#" * filled + "." * empty

        if pct >= 75:
            symbol = "GREEN "
        elif pct >= 25:
            symbol = "YELLOW"
        elif pct > 0:
            symbol = "ORANGE"
        else:
            symbol = "RED   "

        lines.append(
            "{} [{}] {:20s} {:3d}/{:3d} ({:5.1f}%)".format(
                symbol, bar, gloss_id[:20], info["available"], info["expected"], pct
            )
        )

    lines.append("")
    lines.append("Uncovered glosses (need samples):")
    uncovered = [g for g, v in coverage.items() if not v["covered"]]
    if uncovered:
        for g in uncovered:
            lines.append("  - {} (expected from {})".format(g, ", ".join(sorted(coverage[g].get("_packs", []))) if "_packs" in coverage[g] else ""))
    else:
        lines.append("  None — all glosses have at least one sample!")

    return "\n".join(lines)


def format_json(coverage: Dict[str, dict]) -> str:
    """Export coverage as JSON."""
    summary = {
        "total": len(coverage),
        "covered": sum(1 for v in coverage.values() if v["covered"]),
        "uncovered": sum(1 for v in coverage.values() if not v["covered"]),
        "glosses": list(coverage.values()),
    }
    return json.dumps(summary, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Gloss coverage heatmap CLI")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--uncovered-only", action="store_true", help="Show only uncovered glosses")
    args = parser.parse_args()

    gloss_map = load_default_gloss()
    samples = load_available_samples()
    coverage = compute_coverage(gloss_map, samples)

    # Attach pack info for uncovered display
    for gloss_id, info in coverage.items():
        info["_packs"] = list(gloss_map.get(gloss_id, set()))

    if args.uncovered_only:
        coverage = {k: v for k, v in coverage.items() if not v["covered"]}

    if args.json:
        print(format_json(coverage))
    else:
        print(format_heatmap(coverage))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
