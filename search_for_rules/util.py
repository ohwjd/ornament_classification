import os

from fractions import Fraction


def extract_tbp_basic_metadata(path):
    """Read the beginning of a .tbp file and return (tuning, meter_info, diminution).

    Assumes header lines look like {TUNING:G}, {METER_INFO:2/2 (1-56)}, {DIMINUTION:2}.
    Stops scanning once musical content appears (line starting with '|', or a dot-containing syllable line),
    keeping implementation intentionally minimal.
    """
    tuning = None
    meter_info = None
    meter_info_raw = None
    diminution = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                # Heuristic: music/content begins
                if (
                    line.startswith("|")
                    or line.startswith(".")
                    or line.startswith("br.")
                ):
                    break
                if not (line.startswith("{") and line.endswith("}")):
                    continue
                inner = line[1:-1]
                if ":" not in inner:
                    continue
                key, val = inner.split(":", 1)
                key = key.strip().upper()
                val = val.strip()
                if key == "TUNING":
                    tuning = val
                elif key in ("METER_INFO", "METER"):
                    meter_info_raw = val
                elif key == "DIMINUTION":
                    # Try int, else keep raw
                    try:
                        diminution = int(val)
                    except ValueError:
                        diminution = val
        meter_info, bar_num, meter_raw = parse_meter_info(meter_info_raw)
    except FileNotFoundError:
        return None, None, None, None, None
    return tuning, meter_info, bar_num, diminution, meter_raw


def parse_meter_info(meter_info_raw):
    """Given a raw meter_info string (e.g., "2/2 (1-56)"), extract the meter and bar number.

    Returns (meter_info, bar_num) where meter_info is a Fraction constructed from
    the meter numerator/denominator (e.g., Fraction(2, 2)) and bar_num is an integer
    if found (for ranges like "1-56" the second value, 56, is returned), else None.
    """
    if not meter_info_raw:
        return None, None, None

    parts = meter_info_raw.split("(")
    meter_str = parts[0].strip()
    meter_raw = meter_str
    bar_num = None

    # Extract bar number if present. For ranges like "1-56" return the end (56).
    if len(parts) > 1:
        tail = parts[1].strip(" )")
        if "-" in tail:
            # take the right-hand side of the range
            bar_part = tail.split("-")[-1]
            try:
                bar_num = int(bar_part)
            except ValueError:
                bar_num = None
        else:
            try:
                bar_num = int(tail)
            except Exception:
                bar_num = None

    # Parse meter fraction into a Fraction object using the explicit numerator/denominator
    try:
        if "/" in meter_str:
            num_str, den_str = meter_str.split("/")
            num = int(num_str.strip())
            den = int(den_str.strip())
            meter_frac = Fraction(num, den)
        else:
            meter_frac = Fraction(int(meter_str.strip()), 1)
    except Exception:
        # Fallback: try to construct Fraction directly from the string
        try:
            meter_frac = Fraction(meter_str)
        except Exception:
            meter_frac = None

    return meter_frac, bar_num, meter_raw


def get_matching_file_path(
    file_name,
    files_list,
    first_replace_str="-mapping",
    second_replace_str="-score",
    folder_path="",
):
    # Remove "-score" and extension for matching
    base_name = os.path.splitext(file_name.replace(first_replace_str, ""))[0]
    for file_name in files_list:
        base = os.path.splitext(file_name.replace(second_replace_str, ""))[0]
        try:
            if base == base_name:
                return os.path.join(folder_path, file_name)
        except Exception as e:
            print("Error while matching:", e)
            return None


def write_summary(
    file_output_dir,
    base_name,
    tuning,
    meter_raw,
    bar_num,
    diminution,
    summary_counts,
    category,
):
    """Write the per-file summary and its LaTeX table into one text file."""

    header_info = {
        "tuning": tuning,
        "meter_info": meter_raw,
        "bar_num": bar_num,
        "diminution": diminution,
        "category": category,
    }

    summary_lines = [f"Summary for {base_name}"]
    summary_lines.append(f"tuning: {tuning}")
    summary_lines.append(f"meter_info: {meter_raw}")
    summary_lines.append(f"bar_num: {bar_num}")
    summary_lines.append(f"diminution: {diminution}")
    summary_lines.append(f"category: {category}")
    summary_lines.append("--------------------------------")
    for key, val in sorted(summary_counts.items()):
        summary_lines.append(f"{key}: {val}")

    latex_table = create_latex_table(
        base_name=base_name,
        header_info=header_info,
        summary_counts=summary_counts,
    )

    summary_path = os.path.join(file_output_dir, f"{base_name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("\n".join(summary_lines))
        sf.write("\n\n")
        sf.write(latex_table)
        sf.write("\n")
    return summary_path


def create_latex_table(
    *,
    base_name,
    header_info,
    summary_counts,
    target_rows=None,
):
    """Build the LaTeX table string from in-memory summary data."""

    if target_rows is None:
        target_rows = ["abtab", "raw", "voices_merged"]

    def latex_escape(value):
        text = str(value)
        replacements = {
            "\\": "\\textbackslash{}",
            "_": "\\_",
            "%": "\\%",
            "&": "\\&",
            "#": "\\#",
            "{": "\\{",
            "}": "\\}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    columns = ["count"]
    pretty_names = {
        "count": "count",
        "eighth": "8th",
        "eighth_fourstep": "8th 4-step",
        "fourstep": "4-step",
        "count_abrupt_duration_changes": "abrupt",
        "abrupt_duration_change_sequence_ids": None,
        "count_consonant_beginning_sequences": "cons start",
        "non_chord": "non-chord",
        "starting_chord": "start chord",
    }

    row_values = {row: {} for row in target_rows}

    for full_key in sorted(summary_counts):
        val = summary_counts[full_key]
        for row in target_rows:
            if full_key == row:
                row_values[row]["count"] = val
                break
            prefix = f"{row}_"
            if full_key.startswith(prefix):
                suffix = full_key[len(prefix) :]
                if suffix == "non_consonant_beginning_sequence_ids":
                    break
                row_values[row][suffix] = val
                if pretty_names.get(suffix, suffix) and suffix not in columns:
                    columns.append(suffix)
                break

    for row in target_rows:
        row_values[row].setdefault("count", "--")

    display_columns = [col for col in columns if pretty_names.get(col, col)]

    col_spec = "l" + "l" * len(display_columns)
    latex_lines = ["\\begin{table}[ht]", "\\centering"]

    caption_parts = []
    if base_name:
        caption_parts.append(f"Summary for {latex_escape(base_name)}")
    for key in ("category", "tuning", "meter_info", "bar_num", "diminution"):
        value = header_info.get(key)
        if value not in (None, ""):
            caption_parts.append(
                f"{latex_escape(key)}: {latex_escape(value)}"
            )

    if caption_parts:
        latex_lines.append(f"\\caption{{{' ; '.join(caption_parts)}}}")

    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\hline")

    header_cells = [""]
    for col in display_columns:
        pretty = pretty_names.get(col, col)
        if not pretty:
            continue
        header_cells.append(latex_escape(pretty))

    row_end = " \\\\"  # space then LaTeX line break
    latex_lines.append(" & ".join(header_cells) + row_end)
    latex_lines.append("\\hline")

    for row in target_rows:
        row_cells = [latex_escape(row)]
        for col in display_columns:
            value = row_values[row].get(col, "--")
            if value == 0:
                value = "0"
            elif value in (None, ""):
                value = "--"
            row_cells.append(latex_escape(value))
        latex_lines.append(" & ".join(row_cells) + row_end)

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)
