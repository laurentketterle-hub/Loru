"""Gloss coverage heatmap utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from loru.models.vocab import DEFAULT_GLOSS, load_vocab

console = Console()


def _heat_color(ratio: float) -> str:
    """Return a rich color based on coverage ratio (0.0 = red, 1.0 = green)."""
    if ratio >= 1.0:
        return "green"
    if ratio >= 0.8:
        return "bright_green"
    if ratio >= 0.6:
        return "yellow"
    if ratio >= 0.4:
        return "dark_orange"
    if ratio > 0.0:
        return "red"
    return "bright_black"


def _heat_bar(ratio: float, width: int = 10) -> str:
    """Return a text heat bar for display."""
    filled = int(ratio * width)
    empty = width - filled
    bar = "\u2588" * filled + "\u2591" * empty
    return f"[{_heat_color(ratio)}]{bar}[/{_heat_color(ratio)}]"


def print_coverage_heatmap(
    samples_dir: Optional[Path] = None,
    gloss_filter: Optional[str] = None,
    vocab_file: Optional[Path] = None,
) -> dict:
    """Print a rich heatmap of gloss sample coverage and return summary dict."""
    # Lazy import to avoid numpy dependency at module level
    from loru.data.loader import list_sample_files
    from loru.config import SAMPLES_DIR

    dir_path = samples_dir or SAMPLES_DIR
    files = {p.stem for p in list_sample_files(dir_path)}
    vocab = load_vocab(vocab_file)

    # Build rows
    rows = []
    for g in vocab:
        if gloss_filter and gloss_filter.lower() not in g.lower():
            continue
        has_sample = g in files
        rows.append((g, has_sample))

    if not rows:
        console.print("[yellow]No glosses match the filter.[/yellow]")
        return {"total": 0, "covered": 0, "ratio": 0.0}

    covered = sum(1 for _, ok in rows if ok)
    total = len(rows)
    ratio = covered / total if total > 0 else 0.0

    # Heatmap grid
    table = Table(
        title=f"Gloss Coverage Heatmap ({covered}/{total} = {ratio:.1%})",
        title_style="bold",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Gloss", style="dim")
    table.add_column("Coverage", justify="left")
    table.add_column("Status", justify="center")

    for gloss, has_sample in rows:
        bar = _heat_bar(1.0 if has_sample else 0.0, width=15)
        status = "[green]Y[/green]" if has_sample else "[red]N[/red]"
        table.add_row(gloss, bar, status)

    console.print(table)

    # Summary panel
    if ratio >= 0.9:
        panel_style = "green"
        status_text = "EXCELLENT"
    elif ratio >= 0.5:
        panel_style = "yellow"
        status_text = "MODERATE"
    else:
        panel_style = "red"
        status_text = "LOW"

    summary = f"{status_text}: {covered}/{total} glosses have samples ({ratio:.1%})"
    console.print(Panel(summary, style=panel_style))
    console.print()

    return {"total": total, "covered": covered, "ratio": round(ratio, 4)}


def compute_coverage_stats(
    samples_dir: Optional[Path] = None,
    vocab_file: Optional[Path] = None,
) -> dict:
    """Compute coverage statistics without printing."""
    # Lazy import to avoid numpy dependency at module level
    from loru.data.loader import list_sample_files
    from loru.config import SAMPLES_DIR

    dir_path = samples_dir or SAMPLES_DIR
    files = {p.stem for p in list_sample_files(dir_path)}
    vocab = load_vocab(vocab_file)
    covered = sum(1 for g in vocab if g in files)
    total = len(vocab)
    return {
        "total_glosses": total,
        "covered_glosses": covered,
        "missing_glosses": total - covered,
        "coverage_ratio": round(covered / total, 4) if total > 0 else 0.0,
        "missing": [g for g in vocab if g not in files],
    }
