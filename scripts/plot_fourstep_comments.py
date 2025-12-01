"""
Visualise four_step_comment distributions for every fourstep CSV.

The script walks through the ``output`` directory, finds every ``fourstep``
subdirectory, and counts how many sequences contain each
``four_step_comment`` label. Every comment is counted at most once per
``sequence_id``. A dedicated bar chart is stored for each CSV so that comment
distributions can be inspected per file.
Aggregated "perfect_fifth" counts combine all variants containing "perfect_fifth"
in their label.

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


def _extract_type_from_filename(csv_label: str) -> str:
    """Grab the ornament type token from the CSV filename."""
    stem = Path(csv_label).stem
    if stem.endswith("_fourstep"):
        stem = stem[: -len("_fourstep")]

    if "_tab_" in stem:
        type_token = stem.rsplit("_tab_", 1)[-1]
    elif "_" in stem:
        type_token = stem.rsplit("_", 1)[-1]
    else:
        type_token = stem

    return type_token or "unknown"


def _title_components(csv_label: str) -> tuple[str, str]:
    """Return the shortened stem and ornament type for labelling."""
    stem = Path(csv_label).stem or "plot"
    short_name = stem[:15] or "plot"
    ornament_type = _extract_type_from_filename(csv_label)
    return short_name, ornament_type


def _build_plot_title(csv_label: str) -> str:
    """Return shortened filename and type suitable for the plot title."""
    short_name, ornament_type = _title_components(csv_label)
    return f"{short_name} ({ornament_type})"


def _slugify(value: str) -> str:
    """Convert an arbitrary string to a filesystem-friendly slug."""
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    slug = "_".join(filter(None, slug.split("_")))
    return slug or "item"


def _build_plot_filename(csv_label: str) -> str:
    """Return a compact, readable filename for the generated plot image."""
    short_name, ornament_type = _title_components(csv_label)
    title_slug = _slugify(short_name)
    type_slug = _slugify(ornament_type)
    return f"{title_slug}_{type_slug}.png"


def _combine_perfect_fifth(comment_counter: Counter[str]) -> Counter[str]:
    """Aggregate perfect-fifth variants into a single bucket for plotting."""
    combined: Counter[str] = Counter()
    perfect_total = 0

    for label, value in comment_counter.items():
        if "perfect_fifth" in label:
            perfect_total += value
        else:
            combined[label] = value

    if perfect_total:
        combined["perfect_fifth"] = perfect_total

    return combined


def plot_counts(counts: OrderedDict[str, Counter[str]]) -> None:
    """Create a horizontal bar plot for each CSV in ``counts``."""
    if not counts:
        print(
            "No fourstep CSV files with four_step_comment column found. Nothing to plot."
        )
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for csv_label, comment_counter in counts.items():
        display_counts = _combine_perfect_fifth(comment_counter)
        series = pd.Series(display_counts).sort_values(ascending=True)
        fig_height = max(3.5, 0.35 * len(series))
        fig, ax = plt.subplots(figsize=(10, fig_height))

        ax.barh(range(len(series)), series.values, color="steelblue")
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels(series.index, fontsize=9)
        ax.set_xlabel("Sequences containing comment")
        title_label = _build_plot_title(csv_label)
        ax.set_title(title_label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for idx, value in enumerate(series.values):
            ax.text(value + 0.1, idx, str(int(value)), va="center", fontsize=8)

        ax.set_xlim(0, max(series.values) * 1.1 if series.values.size else 1)
        fig.tight_layout()

        output_path = PLOTS_DIR / _build_plot_filename(csv_label)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot for {csv_label} -> {output_path}")


def main() -> None:
    counts = collect_comment_counts()
    plot_counts(counts)
    print(f"Processed {len(counts)} CSV files.")


if __name__ == "__main__":
    main()
