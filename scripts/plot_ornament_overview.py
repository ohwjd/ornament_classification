#!/usr/bin/env python3
"""Plot ornament statistics aggregated from *_summary.txt files under the output directory."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_FIGURE_PATH = OUTPUT_DIR / "ornament_overview.png"
NUMERIC_FIELDS = {
    "bar_num",
    "tab",
    "tab_fourstep",
    "tab_raw",
    "tab_raw_fourstep",
}


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
        if raw_value in {None, "", "None"}:
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
    required_columns = {"piece", "bar_num", "tab_raw", "tab_raw_fourstep"}
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

    frame["mean_tab_raw_per_bar"] = frame["tab_raw"] / frame["bar_num"]
    frame["mean_tab_raw_fourstep_per_bar"] = (
        frame["tab_raw_fourstep"] / frame["bar_num"]
    )

    return frame


def plot_overview(frame: pd.DataFrame, figure_path: Path) -> None:
    """Create the overview figure with the requested subplots."""
    metrics = [
        ("tab", "Tab ornaments (absolute)", "Total ornaments (tab)", "#1f77b4"),
        (
            "tab_raw",
            "Tab raw ornaments (absolute)",
            "Total ornaments (tab_raw)",
            "#ff7f0e",
        ),
        (
            "tab_raw_fourstep",
            "Tab raw fourstep ornaments (absolute)",
            "Total ornaments (tab_raw_fourstep)",
            "#2ca02c",
        ),
        (
            "mean_tab_raw_per_bar",
            "Average ornaments per bar (tab_raw)",
            "Ornaments per bar",
            "#d62728",
        ),
        (
            "mean_tab_raw_fourstep_per_bar",
            "Average ornaments per bar (tab_raw_fourstep)",
            "Ornaments per bar",
            "#9467bd",
        ),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 28), sharex=True)
    fig.suptitle("Ornament metrics across pieces", fontsize=16)

    for ax, (column, title, ylabel, color) in zip(axes, metrics):
        if column not in frame.columns:
            ax.text(
                0.5,
                0.5,
                "Metric not available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            continue

        subset = frame.dropna(subset=["bar_num", column])
        if subset.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            continue
        ax.scatter(
            subset["bar_num"],
            subset[column],
            color=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
        )

        data_max = subset[column].max()
        data_min = subset[column].min()
        span = max(data_max - data_min, 0.1)

        for _, row in subset.iterrows():
            label_text = row.get("label") or row.get("piece") or ""
            if not label_text:
                continue
            label_text = label_text[:12]
            x_val = float(row["bar_num"])
            y_val = float(row[column])
            ax.annotate(
                label_text,
                (x_val, y_val),
                textcoords="offset points",
                xytext=(3, -2),
                fontsize=6,
                rotation=15,
                ha="left",
                va="bottom",
            )

        headroom = span * 0.15
        ymin, ymax = ax.get_ylim()
        target_top = data_max + headroom
        if target_top > ymax:
            ax.set_ylim(ymin, target_top)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Number of bars (bar_num)")
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=2.0)

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
        "--figure-path",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help="Path to the plotted PNG file.",
    )
    args = parser.parse_args()

    frame = collect_metrics(args.output_dir)
    plot_overview(frame, args.figure_path)
    print(f"Saved figure to {args.figure_path}")


if __name__ == "__main__":
    main()
