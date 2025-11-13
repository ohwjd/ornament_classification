"""Visualise four_step_comment distributions for every fourstep CSV.

The script walks through the ``output`` directory, finds every ``fourstep``
subdirectory, and counts how many sequences contain each
``four_step_comment`` label. Every comment is counted at most once per
``sequence_id`` to avoid inflating totals when it appears multiple times in the
same sequence. A dedicated bar chart is stored for each CSV so that comment
distributions can be inspected per file.

Usage::

    python -m scripts.plot_fourstep_comments

The plot is written to ``output/fourstep_comment_counts.png`` by default.
"""

from __future__ import annotations

import ast
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
    ) from exc

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "fourstep_comment_plots"


def _iter_fourstep_csvs(root: Path) -> Iterable[Path]:
    """Yield every CSV contained in a ``fourstep`` folder under ``root``."""
    return sorted(root.glob("**/fourstep/*.csv"))


def _parse_comment_cell(cell: str) -> list[str]:
    """Return a list of comment labels for a single table cell."""
    if not isinstance(cell, str) or not cell.strip():
        return []

    try:
        parsed = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        return [parsed.strip()] if parsed.strip() else []

    # Fall back to simple comma splitting when parsing fails.
    stripped = cell.strip().strip("[]")
    return [chunk.strip() for chunk in stripped.split(",") if chunk.strip()]


def collect_comment_counts() -> OrderedDict[str, Counter[str]]:
    """Return ordered mapping of CSV label -> counts per comment type."""
    counts: OrderedDict[str, Counter[str]] = OrderedDict()

    for csv_path in _iter_fourstep_csvs(OUTPUT_DIR):
        try:
            df = pd.read_csv(csv_path, sep=";")
        except FileNotFoundError:
            continue

        if (
            "four_step_comment" not in df.columns
            or "sequence_id" not in df.columns
        ):
            continue

        sequence_comments: dict[str | int, set[str]] = {}
        relevant = df[["sequence_id", "four_step_comment"]].dropna(
            subset=["sequence_id", "four_step_comment"]
        )

        for sequence_id, cell in relevant.itertuples(index=False):
            parsed_comments = _parse_comment_cell(cell)
            if not parsed_comments:
                continue
            bucket = sequence_comments.setdefault(sequence_id, set())
            bucket.update(parsed_comments)

        comment_counter: Counter[str] = Counter()
        for bucket in sequence_comments.values():
            comment_counter.update(bucket)
        if not comment_counter:
            continue

        key = str(csv_path.relative_to(OUTPUT_DIR))
        counts[key] = comment_counter

    return counts


def _safe_plot_name(csv_label: str) -> str:
    """Return a filesystem-friendly plot name derived from the CSV label."""
    return csv_label.replace("/", "__").replace(" ", "_")


def plot_counts(counts: OrderedDict[str, Counter[str]]) -> None:
    """Create a horizontal bar plot for each CSV in ``counts``."""
    if not counts:
        print(
            "No fourstep CSV files with four_step_comment column found. Nothing to plot."
        )
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for csv_label, comment_counter in counts.items():
        series = pd.Series(comment_counter).sort_values(ascending=True)
        fig_height = max(3.5, 0.35 * len(series))
        fig, ax = plt.subplots(figsize=(10, fig_height))

        ax.barh(range(len(series)), series.values, color="steelblue")
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels(series.index, fontsize=9)
        ax.set_xlabel("Sequences containing comment")
        title_label = Path(csv_label).name
        ax.set_title(f"counts: {title_label}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for idx, value in enumerate(series.values):
            ax.text(value + 0.1, idx, str(int(value)), va="center", fontsize=8)

        ax.set_xlim(0, max(series.values) * 1.1 if series.values.size else 1)
        fig.tight_layout()

        output_path = PLOTS_DIR / f"{_safe_plot_name(csv_label)}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot for {csv_label} -> {output_path}")


def main() -> None:
    counts = collect_comment_counts()
    plot_counts(counts)
    print(f"Processed {len(counts)} CSV files.")


if __name__ == "__main__":
    main()
