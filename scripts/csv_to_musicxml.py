import os
from typing import List, Optional

import pandas as pd

from music21 import stream, note, chord, duration, meter, clef, key


COMMON_PITCH_COLS = ["pitch_midi", "midi", "pitch"]
COMMON_ONSET_COLS = ["onset", "time", "offset"]
COMMON_DUR_COLS = ["duration", "dur", "ql"]
COMMON_VOICE_COLS = ["voice", "voice_order", "part", "staff"]


def infer_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_numeric(value: object) -> float:
    """Parse numbers that may be floats, ints, or fractional strings like '3/4'."""
    from fractions import Fraction

    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    # Try plain float first
    try:
        return float(s)
    except Exception:
        pass

    # Try fractional form (e.g., '3/4')
    try:
        return float(Fraction(s))
    except Exception as e:
        raise ValueError(f"Cannot parse numeric value: {value}") from e


def build_score_from_csv(
    df: pd.DataFrame,
    pitch_col: str,
    onset_col: str,
    dur_col: str,
    voice_col: Optional[str],
    time_signature: str,
    use_tab_clef: bool,
) -> stream.Score:
    score = stream.Score(id="CSV_Reconstruction")

    # Use a single staff/part for all notes
    part = stream.Part(id="Staff1")
    part.append(meter.TimeSignature(time_signature))
    part.append(key.KeySignature(-1))
    if use_tab_clef:
        part.insert(0, clef.TabClef())

    # Insert events
    for _, row in df.iterrows():
        # Multiply onset values by 4 to match duration scaling (quarterLength units)
        onset_val = _parse_numeric(row[onset_col]) * 4.0
        dur_raw = _parse_numeric(row[dur_col])
        # Multiply all durations by 4 (quarterLength units)
        dur_val = (dur_raw if dur_raw > 0 else 0.25) * 4.0

        # Support Note or Chord: allow comma-separated MIDI list in pitch column
        pitch_raw = row[pitch_col]
        if isinstance(pitch_raw, str) and "," in pitch_raw:
            midi_pitches = [
                int(x.strip()) for x in pitch_raw.split(",") if x.strip()
            ]
            n = chord.Chord(midi_pitches)
        else:
            n = note.Note(int(pitch_raw))

        n.duration = duration.Duration(dur_val)
        # Use absolute offset for placement in the single part
        part.insert(onset_val, n)

    score.insert(0, part)

    return score


def main() -> None:
    # Fixed configuration: update INPUT_CSV if needed
    # INPUT_CSV = "/Users/jaklin/Desktop/bachelor-thesis/ornament_classification/output/4400_45_ach_unfall_was/4400_45_ach_unfall_was_ornament_sequences_voice_tab_chordcontext_fourstep.csv"
    INPUT_CSV = "output/4400_45_ach_unfall_was/4400_45_ach_unfall_was_ornament_sequences_voice_tab_chordcontext.csv"
    TIME_SIGNATURE = "2/2"
    USE_TAB_CLEF = False

    df = pd.read_csv(INPUT_CSV)

    pitch_col = infer_column(df, COMMON_PITCH_COLS)
    onset_col = infer_column(df, COMMON_ONSET_COLS)
    dur_col = infer_column(df, COMMON_DUR_COLS)
    voice_col = infer_column(df, COMMON_VOICE_COLS)

    if not pitch_col or not onset_col or not dur_col:
        missing = [
            name
            for name, col in [
                ("pitch", pitch_col),
                ("onset", onset_col),
                ("duration", dur_col),
            ]
            if col is None
        ]
        raise SystemExit(
            f"Missing required columns: {', '.join(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    score = build_score_from_csv(
        df,
        pitch_col=pitch_col,
        onset_col=onset_col,
        dur_col=dur_col,
        voice_col=voice_col,
        time_signature=TIME_SIGNATURE,
        use_tab_clef=USE_TAB_CLEF,
    )

    base_no_ext, _ = os.path.splitext(INPUT_CSV)
    out_musicxml = f"{base_no_ext}.musicxml"

    os.makedirs(os.path.dirname(out_musicxml) or ".", exist_ok=True)
    score.write("musicxml", fp=out_musicxml)

    print(f"Wrote MusicXML to {out_musicxml}")


if __name__ == "__main__":
    main()
