"""
Re-export diplomatic MEI files to common music notation (CMN) format using music21 for ease of reading.
This script was originally intended to try to create mei files from tabmapper CSV outputs, but this was not acomplished.

"""

import os
import sys
import tempfile

from fractions import Fraction

import music21 as m21
import pandas as pd
import xml.etree.ElementTree as ET


score_mei_folder_path = "data/tabmapper_output/voiced_meis/"
dipl_mei_folder_path = "data/meis_dipl/"
csv_folder_path = "data/tabmapper_output/mapping_csvs/"


score_mei_file_names = sorted(
    [f for f in os.listdir(score_mei_folder_path) if f.endswith(".mei")]
)

dipl_mei_file_names = sorted(
    [f for f in os.listdir(dipl_mei_folder_path) if f.endswith(".mei")]
)

csv_file_names = sorted(
    [f for f in os.listdir(csv_folder_path) if f.endswith(".csv")]
)

MEI_NS = "http://www.music-encoding.org/ns/mei"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _safe_unlink(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def augment_mei_dir_durations(mei_path):
    ET.register_namespace("", MEI_NS)
    ET.register_namespace("xml", "http://www.w3.org/XML/1998/namespace")
    tree = ET.parse(mei_path)
    root = tree.getroot()

    id_lookup = {}
    for element in root.iter():
        elem_id = element.attrib.get(XML_ID)
        if elem_id:
            id_lookup[elem_id] = element

    changed = False

    for dir_elem in root.findall(f".//{{{MEI_NS}}}dir"):
        startid = dir_elem.attrib.get("startid")
        if not startid:
            continue
        ref_id = startid.lstrip("#")
        target = id_lookup.get(ref_id)
        if target is None:
            continue

        dot_count = 0

        for symbol in dir_elem.findall(f"{{{MEI_NS}}}symbol"):
            glyph_name = symbol.attrib.get("glyph.name")
            if glyph_name == "augmentationDot":
                dot_count += 1
                continue

        if dot_count > 0:
            if target.get("dots") != str(dot_count):
                target.set("dots", str(dot_count))
                changed = True
        elif "dots" in target.attrib:
            del target.attrib["dots"]
            changed = True

    if not changed:
        return mei_path, None

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mei")
    with os.fdopen(tmp_fd, "wb") as tmp_file:
        tmp_file.write(
            ET.tostring(root, encoding="utf-8", xml_declaration=True)
        )
    return tmp_path, lambda: _safe_unlink(tmp_path)


def parse_mei_with_dir_durations(mei_path):
    prepared_path, cleanup = augment_mei_dir_durations(mei_path)
    try:
        return m21.converter.parse(prepared_path)
    finally:
        if cleanup is not None:
            cleanup()


def filter_score_files(score_files, args):
    if not args:
        raise ValueError(
            "Provide at least one MEI file path, diplomatic file, or prefix."
        )

    resolved = []
    for raw_arg in args:
        arg = os.path.basename(raw_arg)

        if arg in score_files:
            resolved.append(arg)
            continue

        if arg.endswith("-dipl.mei"):
            base = os.path.splitext(arg)[0]
            if base.endswith("-dipl"):
                base = base[: -len("-dipl")]

            candidate_variants = [
                f"{base}-score.mei",
                f"{base}.mei",
            ]
            for candidate in candidate_variants:
                if candidate in score_files:
                    resolved.append(candidate)
                    break
            else:
                raise ValueError(
                    "No score MEI file found for diplomatic file "
                    f"'{raw_arg}'."
                )
            continue

        prefix_matches = [f for f in score_files if f.startswith(arg)]
        if prefix_matches:
            resolved.extend(prefix_matches)
            continue

        raise ValueError(f"No score MEI file found matching '{raw_arg}'.")

    # Preserve argument order while removing duplicates
    seen = set()
    ordered_unique = []
    for name in resolved:
        if name not in seen:
            ordered_unique.append(name)
            seen.add(name)

    return ordered_unique


def merged_df_to_stream(merged_df):
    part = m21.stream.Part(id="merged_from_csv")
    for _, row in (
        merged_df.dropna(subset=["c_pitch", "c_onset"])
        .sort_values("c_onset", kind="stable")
        .iterrows()
    ):
        try:
            onset = Fraction(row["c_onset"]).limit_denominator(1024)
        except Exception:
            continue
        try:
            pitch_val = int(row["c_pitch"])
        except Exception:
            continue
        note_obj = m21.note.Note(pitch_val)
        duration_val = row.get("duration_quartered")
        if pd.notna(duration_val):
            try:
                note_obj.duration = m21.duration.Duration(duration_val)
            except Exception:
                pass
        part.insert(float(onset), note_obj)
    return part


def write_stream_to_musicxml(stream_obj, destination):
    try:
        stream_obj.write("musicxml", fp=destination)
    except Exception as err:
        raise RuntimeError(f"MusicXML export failed: {err}") from err


def get_matching_file_path(
    mei_file_name,
    files_list,
    base_replace_str="-mapping",
    score_or_dipl_replace_str="-score",
    folder_path="",
):
    # Remove "-score" and extension for matching
    base_name = os.path.splitext(
        mei_file_name.replace(score_or_dipl_replace_str, "")
    )[0]
    for file_name in files_list:
        base = os.path.splitext(file_name.replace(base_replace_str, ""))[0]
        try:
            if base == base_name:
                return os.path.join(folder_path, file_name)
        except Exception as e:
            print("No matching file found:", e)
            return None


def parse_and_chordify(mei_path):
    parsed = parse_mei_with_dir_durations(mei_path)
    return parsed.chordify()


def extract_notes(m21_score):
    measures = []
    chords = []
    notes_data = []

    for m in m21_score:
        if isinstance(m, m21.stream.Measure):
            measures.append(m)
            for chord in m.notes:
                chords.append(chord)
                try:
                    absolute_offset = Fraction(
                        chord.getOffsetInHierarchy(m21_score)
                    ).limit_denominator(1024)
                except Exception:
                    absolute_offset = Fraction(chord.offset).limit_denominator(
                        1024
                    )
                onset_fraction = absolute_offset / 4
                for i, note in enumerate(chord.notes):
                    voice_id = None
                    voice_ctx = note.getContextByClass(m21.stream.Voice)
                    if voice_ctx is not None and voice_ctx.id is not None:
                        try:
                            voice_id = int(str(voice_ctx.id))
                        except ValueError:
                            voice_id = str(voice_ctx.id)
                    notes_data.append(
                        {
                            "pitch": note.pitch.nameWithOctave,
                            "duration": note.quarterLength,
                            "measure": chord.measureNumber,
                            "chord_pos": i,
                            # "copm": chord.offset, # chord_offset_per_measure
                            # "accidental": note.pitch.accidental,
                            "midi_pitch": note.pitch.midi,
                            "pitch_step": note.pitch.step,  # relevant for rule with stepwise-motion
                            "onset_fraction": onset_fraction,
                            "absolute_offset": absolute_offset,
                            "voice": voice_id,
                            # "pitchClass": note.pitch.pitchClass,
                            # "octave": note.pitch.octave,
                        }
                    )
    return measures, chords, pd.DataFrame(notes_data)


def merge_with_csv(notes_df, csv_df):

    # Prepare CSV columns
    if "note" in csv_df.columns:
        csv_df = csv_df.rename(columns={"note": "note_number"})
    if "mapped voice" in csv_df.columns:
        csv_df = csv_df.rename(columns={"mapped voice": "voice"})
    csv_df = csv_df.rename(columns=lambda x: f"c_{x}")

    if notes_df.empty or csv_df.empty:
        return pd.DataFrame()

    notes_df = notes_df.reset_index()
    csv_df = csv_df.reset_index(drop=True)

    def to_fraction(value):
        try:
            return Fraction(str(value)).limit_denominator(1024)
        except Exception:
            return None

    def normalize_voice(value):
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return int(value)
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return str(value).strip()

    csv_df["c_onset_fraction"] = csv_df["c_onset"].apply(to_fraction)
    if "c_voice" in csv_df.columns:
        csv_df["c_voice_normalized"] = csv_df["c_voice"].apply(normalize_voice)
    else:
        csv_df["c_voice_normalized"] = None

    if "onset_fraction" in notes_df.columns:
        notes_df["onset_fraction"] = notes_df["onset_fraction"].apply(
            lambda val: (
                to_fraction(val) if not isinstance(val, Fraction) else val
            )
        )
    else:
        notes_df["onset_fraction"] = None
    if "voice" in notes_df.columns:
        notes_df["voice_normalized"] = notes_df["voice"].apply(normalize_voice)
    else:
        notes_df["voice_normalized"] = None

    note_records = []
    for idx, row in notes_df.iterrows():
        onset = row.get("onset_fraction")
        pitch = row.get("midi_pitch")
        if onset is None or pd.isna(onset) or pd.isna(pitch):
            continue
        try:
            midi_pitch = int(pitch)
        except (TypeError, ValueError):
            continue
        note_records.append(
            {
                "note_index": row["index"],
                "data_index": idx,
                "onset": onset,
                "midi": midi_pitch,
                "voice": row.get("voice_normalized"),
                "matched": False,
            }
        )

    offset_tolerance = Fraction(1, 1024)
    merged_rows = []

    for _, csv_row in csv_df.iterrows():
        onset_fraction = csv_row.get("c_onset_fraction")
        pitch_val = csv_row.get("c_pitch")
        if (
            onset_fraction is None
            or pd.isna(onset_fraction)
            or pd.isna(pitch_val)
        ):
            continue
        try:
            midi_pitch = int(pitch_val)
        except (TypeError, ValueError):
            continue
        voice_key = csv_row.get("c_voice_normalized")

        match_record = None
        for candidate in note_records:
            if candidate["matched"]:
                continue
            if candidate["midi"] != midi_pitch:
                continue
            if not (
                candidate["onset"] == onset_fraction
                or abs(candidate["onset"] - onset_fraction) <= offset_tolerance
            ):
                continue
            if (
                voice_key is not None
                and candidate["voice"] is not None
                and candidate["voice"] != voice_key
            ):
                continue
            match_record = candidate
            break

        if match_record is None:
            for candidate in note_records:
                if candidate["matched"]:
                    continue
                if candidate["midi"] != midi_pitch:
                    continue
                if not (
                    candidate["onset"] == onset_fraction
                    or abs(candidate["onset"] - onset_fraction)
                    <= offset_tolerance
                ):
                    continue
                match_record = candidate
                break

        if match_record is None:
            continue

        match_record["matched"] = True
        note_series = notes_df.loc[match_record["data_index"]]
        merged_row = pd.concat([note_series, csv_row])
        merged_rows.append(merged_row)

    if not merged_rows:
        return pd.DataFrame()

    merged_df = pd.DataFrame(merged_rows)
    if "index" in merged_df.columns:
        merged_df = merged_df.set_index("index")
    merged_df = merged_df.sort_values("c_onset_fraction", kind="stable")
    return merged_df


def find_error_in_df(df):
    midi_pitches = df["midi_pitch"].tolist()
    pitch_csvs = df["c_pitch"].tolist()
    i, j = 0, 0
    while i < len(midi_pitches) and j < len(pitch_csvs):
        # Skip NaNs in pitch_csv
        if pd.isna(pitch_csvs[j]):
            i += 1
            j += 1
            continue
        # Compare values
        if midi_pitches[i] != pitch_csvs[j]:
            print(f"*** Divergence at DataFrame index: {df.index[i]}")
            start = max(i - 5, 0)
            end = min(
                i + 6, len(df)
            )  # +6 to include the divergent row and 5 after
            print(df.iloc[start:end])
            return df.iloc[i]
        i += 1
        j += 1
    return None  # No divergence found


try:
    selected_score_files = filter_score_files(
        score_mei_file_names, sys.argv[1:]
    )
except ValueError as err:
    print(err)
    sys.exit(1)

if not selected_score_files:
    print("No score MEI files matched the provided arguments.")
    sys.exit(1)

for f in selected_score_files:
    score_mei_path = os.path.join(score_mei_folder_path, f)

    chordified_score = parse_and_chordify(score_mei_path)
    score_measures, score_chords, score_notes_df = extract_notes(
        chordified_score
    )

    dipl_mei_path = get_matching_file_path(
        f,
        dipl_mei_file_names,
        base_replace_str="-dipl",
        folder_path=dipl_mei_folder_path,
    )
    if not dipl_mei_path:
        print(f"### No matching diplomatic MEI file found for {f}.")
        continue

    try:
        dipl_stream = parse_mei_with_dir_durations(dipl_mei_path)
    except Exception as parse_err:
        print("Diplomatic parse failed; skipping annotation:", parse_err)
        continue

    dipl_base = os.path.splitext(os.path.basename(dipl_mei_path))[0]

    dipl_measures, dipl_chords, dipl_notes_df = extract_notes(
        dipl_stream.chordify()
    )
    if "duration" in dipl_notes_df.columns:
        dipl_notes_df["duration_quartered"] = dipl_notes_df["duration"].apply(
            lambda d: (
                (Fraction(d).limit_denominator(1024) / 4)
                if pd.notna(d)
                else pd.NA
            )
        )

    try:
        dipl_output_dir = os.path.join("output", "dipl_reexports")
        os.makedirs(dipl_output_dir, exist_ok=True)
        reexport_musicxml_path = os.path.join(
            dipl_output_dir, f"{dipl_base}_reexport.musicxml"
        )
        reexport_pdf_path = os.path.join(
            dipl_output_dir, f"{dipl_base}_reexport.pdf"
        )
        dipl_stream.write("musicxml", fp=reexport_musicxml_path)
        try:
            dipl_stream.write("musicxml.pdf", fp=reexport_pdf_path)
        except Exception as reexport_pdf_err:
            print(
                "Diplomatic PDF export failed; continuing without PDF:",
                reexport_pdf_err,
            )
    except Exception as reexport_err:
        print("Diplomatic re-export failed:", reexport_err)

    csv_file_path = get_matching_file_path(
        f,
        csv_file_names,
        base_replace_str="-mapping",
        score_or_dipl_replace_str="-dipl",
        folder_path=csv_folder_path,
    )
    if not csv_file_path:
        print(f"*** No matching CSV file found for {f}.\n")
        continue

    csv_df = pd.read_csv(csv_file_path)
