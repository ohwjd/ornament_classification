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
DEFAULT_LATEX_PATH = OUTPUT_DIR / "ornament_top_tables.tex"
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


def _escape_latex(value: str) -> str:
    """Escape a string for safe use in LaTeX tabular output."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = value
    for original, replacement in replacements.items():
        escaped = escaped.replace(original, replacement)
    return escaped


def _slugify(value: str) -> str:
    """Return a file-system friendly slug derived from the given value."""
    if not value:
        return "unknown"
    slug_chars = [ch.lower() if ch.isalnum() else "_" for ch in value]
    slug = "".join(slug_chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def generate_latex_tables(frame: pd.DataFrame, tex_path: Path) -> None:
    """Create individual LaTeX tables with the top ten entries per metric."""
    metrics = [
        (
            "tab",
            "Top 10 pieces by tab ornaments",
            "tab ornaments",
        ),
        (
            "tab_raw",
            "Top 10 pieces by tab raw ornaments",
            "tab raw ornaments",
        ),
        (
            "tab_raw_fourstep",
            "Top 10 pieces by four-step raw ornaments",
            "tab raw fourstep ornaments",
        ),
        (
            "mean_tab_raw_per_bar",
            "Top 10 pieces by ornaments per bar (tab raw)",
            "mean tab raw per bar",
        ),
        (
            "mean_tab_raw_fourstep_per_bar",
            "Top 10 pieces by ornaments per bar (tab raw fourstep)",
            "mean tab raw fourstep per bar",
        ),
    ]

    lines: List[str] = [
        "% Auto-generated LaTeX tables listing ornament statistics.",
        "% Generated by plot_ornament_overview.py",
        "",
    ]

    def _format_value(value: float) -> str:
        if pd.isna(value):
            return "--"
        if float(value).is_integer():
            return f"{int(value)}"
        return f"{value:.3f}"

    if "category" in frame.columns:
        categories = [
            str(category)
            for category in sorted(frame["category"].dropna().unique())
        ]
    else:
        categories = [None]

    for category in categories:
        category_frame = (
            frame.copy()
            if category is None
            else frame[frame["category"] == category].copy()
        )

        if category_frame.empty:
            continue

        if category is not None:
            lines.extend([f"% Category: {category}", ""])

        for column, caption, col_label in metrics:
            if column not in category_frame.columns:
                continue
            subset = category_frame[["piece", "bar_num", column]].dropna()
            if subset.empty:
                continue
            top_ten = subset.sort_values(column, ascending=False).head(10)

            caption_text = (
                caption if category is None else f"{caption} ({category})"
            )

            lines.extend(
                [
                    r"\begin{table}[htbp]",
                    r"  \centering",
                    f"  \\caption{{{_escape_latex(caption_text)}}}",
                    r"  \begin{tabular}{lrr}",
                    r"    \toprule",
                    r"    Piece & Bars & " + _escape_latex(col_label) + r" \\",
                    r"    \midrule",
                ]
            )

            for _, row in top_ten.iterrows():
                piece = _escape_latex(str(row["piece"]))
                bars = _format_value(row["bar_num"])
                metric_value = _format_value(row[column])
                lines.append(
                    "    {} & {} & {} \\\\".format(piece, bars, metric_value)
                )

            lines.extend(
                [
                    r"    \bottomrule",
                    r"  \end{tabular}",
                    r"\end{table}",
                    "",
                ]
            )

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def plot_overview(
    frame: pd.DataFrame,
    figure_path: Path,
    title_suffix: str | None = None,
) -> None:
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
    figure_title = "Ornament metrics across pieces"
    if title_suffix:
        figure_title = f"{figure_title} ({title_suffix})"
    fig.suptitle(figure_title, fontsize=16)

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
    parser.add_argument(
        "--latex-path",
        type=Path,
        default=DEFAULT_LATEX_PATH,
        help="Destination path for the generated LaTeX tables.",
    )
    args = parser.parse_args()

    frame = collect_metrics(args.output_dir)
    categories_present = (
        "category" in frame.columns and not frame["category"].isna().all()
    )

    if categories_present:
        categories = [
            str(category)
            for category in sorted(frame["category"].dropna().unique())
        ]
        for category in categories:
            category_frame = frame[frame["category"] == category]
            if category_frame.empty:
                continue
            category_figure_path = args.figure_path.with_name(
                f"{args.figure_path.stem}_{_slugify(category)}{args.figure_path.suffix}"
            )
            plot_overview(
                category_frame,
                category_figure_path,
                title_suffix=f"Category: {category}",
            )
            print(f"Saved figure to {category_figure_path}")
    else:
        plot_overview(frame, args.figure_path)
        print(f"Saved figure to {args.figure_path}")

    generate_latex_tables(frame, args.latex_path)
    print(f"Wrote LaTeX tables to {args.latex_path}")


if __name__ == "__main__":
    main()
