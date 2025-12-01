"""
Plot ornament statistics aggregated from *_summary.txt files under the output directory.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_COUNTS_FIGURE_PATH = OUTPUT_DIR / "ornament_counts_breakdown.png"

CM_TO_IN = 1 / 2.54
BASE_SUBPLOT_HEIGHT_IN = 5.0
ADDITIONAL_SUBPLOT_HEIGHT_CM = 2.0
TOP_PADDING_MM = 3.0

NUMERIC_FIELDS = {
    "bar_num",
    "tab",
    "tab_fourstep",
    "tab_raw",
    "tab_raw_fourstep",
    "abtab",
    "abtab_fourstep",
    "abtab_count_abrupt_duration_changes",
    "abtab_count_consonant_beginning_sequences",
    "abtab_non_chord",
    "raw",
    "raw_fourstep",
    "raw_count_abrupt_duration_changes",
    "raw_count_consonant_beginning_sequences",
    "raw_non_chord",
}

COUNT_METRICS = {
    "abtab": {
        "Count": "abtab",
        "Abrupt dur changes": "abtab_count_abrupt_duration_changes",
        "Non-consonant start": "abtab_non_consonant_start",
        "Fourstep": "abtab_fourstep",
        "Non-chord": "abtab_non_chord",
    },
    "raw": {
        "Count": "raw",
        "Abrupt dur changes": "raw_count_abrupt_duration_changes",
        "Non-consonant start": "raw_non_consonant_start",
        "Fourstep": "raw_fourstep",
        "Non-chord": "raw_non_chord",
    },
}


def _add_non_consonant_start_column(
    frame: pd.DataFrame,
    *,
    total_col: str,
    consonant_col: str,
    target_col: str,
) -> None:
    """Derive non-consonant count from total and consonant-start counts."""
    if total_col in frame.columns and consonant_col in frame.columns:
        total = frame[total_col].fillna(0)
        consonant = frame[consonant_col].fillna(0)
        frame[target_col] = (total - consonant).clip(lower=0)


def parse_summary_file(summary_path: Path) -> Dict[str, int | float | str]:
    """Parse a summary file into a flat dictionary of metrics."""
    metrics: Dict[str, int | float | str] = {"piece": summary_path.parent.name}
    header_consumed = False

    with summary_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("----"):
                continue
            if line.startswith("Summary for") and not header_consumed:
                metrics["label"] = line.replace("Summary for", "").strip()
                header_consumed = True
                continue
            if ":" not in line:
                continue
            key, value = (token.strip() for token in line.split(":", 1))
            if not key:
                continue
            metrics[key] = value

    for field in NUMERIC_FIELDS:
        if field not in metrics:
            continue
        raw_value = metrics[field]
        if raw_value in {None, "", "None", "--"}:
            metrics[field] = None
            continue
        try:
            metrics[field] = int(raw_value)
        except (TypeError, ValueError):
            try:
                metrics[field] = float(raw_value)
            except (TypeError, ValueError):
                metrics[field] = None

    metrics.pop("diminution", None)
    return metrics


def collect_metrics(output_root: Path) -> pd.DataFrame:
    """Collect metrics from every *_summary.txt under the output tree."""
    summary_files = sorted(output_root.glob("*/**/*_summary.txt"))
    records: List[Dict[str, int | float | str]] = []

    for summary_path in summary_files:
        record = parse_summary_file(summary_path)
        if record:
            records.append(record)

    if not records:
        raise RuntimeError(f"No summary files found under {output_root}")

    frame = pd.DataFrame(records)
    required_columns = {"piece", "bar_num"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Missing expected columns: {', '.join(sorted(missing))}"
        )

    numeric_columns = [col for col in frame.columns if col in NUMERIC_FIELDS]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["bar_num"])
    frame["bar_num"] = frame["bar_num"].astype(int)
    frame = frame.sort_values("bar_num").reset_index(drop=True)

    _add_non_consonant_start_column(
        frame,
        total_col="abtab",
        consonant_col="abtab_count_consonant_beginning_sequences",
        target_col="abtab_non_consonant_start",
    )
    _add_non_consonant_start_column(
        frame,
        total_col="raw",
        consonant_col="raw_count_consonant_beginning_sequences",
        target_col="raw_non_consonant_start",
    )

    safe_bars = frame["bar_num"].replace(0, pd.NA)

    if "tab_raw" in frame.columns:
        frame["mean_tab_raw_per_bar"] = frame["tab_raw"] / safe_bars
    if "tab_raw_fourstep" in frame.columns:
        frame["mean_tab_raw_fourstep_per_bar"] = (
            frame["tab_raw_fourstep"] / safe_bars
        )
    if "abtab" in frame.columns:
        frame["mean_abtab_per_bar"] = frame["abtab"] / safe_bars
    if "abtab_fourstep" in frame.columns:
        frame["mean_abtab_fourstep_per_bar"] = (
            frame["abtab_fourstep"] / safe_bars
        )
    if "raw" in frame.columns:
        frame["mean_raw_per_bar"] = frame["raw"] / safe_bars
    if "raw_fourstep" in frame.columns:
        frame["mean_raw_fourstep_per_bar"] = frame["raw_fourstep"] / safe_bars

    return frame


def plot_count_breakdown(
    frame: pd.DataFrame,
    figure_path: Path,
    title_suffix: str | None = None,
) -> None:
    """Plot aggregate counts for the main extraction sources."""

    sources = list(COUNT_METRICS.items())
    if not sources:
        raise RuntimeError("No count metric definitions available.")

    subplot_height_in = (
        BASE_SUBPLOT_HEIGHT_IN + ADDITIONAL_SUBPLOT_HEIGHT_CM * CM_TO_IN
    )
    fig_height = subplot_height_in * len(sources)
    fig, axes = plt.subplots(
        len(sources), 1, figsize=(12, fig_height), sharex=False
    )
    if len(sources) == 1:
        axes = [axes]

    figure_title = "Summary of Ornament Counts"
    if title_suffix:
        figure_title = f"{figure_title} ({title_suffix})"
    fig.suptitle(figure_title, fontsize=16)

    for ax, (source_name, metric_map) in zip(axes, sources):
        labels: List[str] = []
        values: List[float] = []

        for display_label, column_name in metric_map.items():
            labels.append(display_label)
            if column_name not in frame.columns:
                values.append(0.0)
                continue
            numeric_series = pd.to_numeric(frame[column_name], errors="coerce")
            values.append(float(numeric_series.fillna(0).sum()))

        if not any(values):
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(source_name.replace("_", " ").title())
            ax.set_ylabel("Total count")
            ax.grid(True, linestyle="--", alpha=0.3)
            continue

        bar_positions = range(len(labels))
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(labels))]
        bars = ax.bar(bar_positions, values, color=colors, alpha=0.85)
        ax.set_xticks(list(bar_positions))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Total count")
        ax.set_title(source_name.replace("_", " ").title())
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        if values:
            y_max = max(values)
            ax.set_ylim(0, y_max * 1.15)
            label_offset = y_max * 0.04
        else:
            label_offset = 0.0

        for rect, value in zip(bars, values):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + label_offset,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    top_padding_in = (TOP_PADDING_MM / 10.0) * CM_TO_IN
    top_rect = max(0.0, 1 - (top_padding_in / fig_height if fig_height else 0))
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ornament counts versus bar counts from summary files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root directory containing subfolders with summary files.",
    )
    parser.add_argument(
        "--counts-figure-path",
        type=Path,
        default=DEFAULT_COUNTS_FIGURE_PATH,
        help="Path to the aggregate counts PNG file.",
    )
    args = parser.parse_args()

    frame = collect_metrics(args.output_dir)

    plot_count_breakdown(frame, args.counts_figure_path)
    print(f"Saved counts figure to {args.counts_figure_path}")


if __name__ == "__main__":
    main()
