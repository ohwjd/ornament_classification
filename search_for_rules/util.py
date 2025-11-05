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
