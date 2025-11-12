import music21 as m21
import pandas as pd

import os

# diminution
# bar: zero-based and visual indication of where the note is in the bar
# bars themselves are one based!
# so 3 7/8 is third bar and the 7/8th beat within that bar

# combine dipl_mei and voice_separated_mei to get the voicing info
# one note in dipl can belong to multiple voices
# go over voice_separated_mei and match with dipl_mei voice by voice


# tomorrow and on tuesday meeting again

# tabmapper-output with csv with all notes
# dipl_mei from tablature_input and voice_separated_mei in separate voices (what is called <layer> in README
# csv: mapped voice has the info from the voice_separated_mei (or is this enough?) some notes belong to multiple voices??

# tabmapper has the info of how many voices a piece has (not necessarily the chord with the most notes)

# note before and after is included into the ornamentation


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
                return folder_path + file_name
        except Exception as e:
            print("No matching file found:", e)
            return None


def parse_and_chordify(mei_path):
    parsed = m21.converter.parse(mei_path)
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
                for i, note in enumerate(chord.notes):
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
                            # "pitchClass": note.pitch.pitchClass,
                            # "octave": note.pitch.octave,
                        }
                    )
    return measures, chords, pd.DataFrame(notes_data)


def merge_with_csv(notes_df, csv_df):

    # must be smarter!

    # Prepare CSV columns
    if "note" in csv_df.columns:
        csv_df = csv_df.rename(columns={"note": "note_number"})
    if "mapped voice" in csv_df.columns:
        csv_df = csv_df.rename(columns={"mapped voice": "voice"})
    csv_df = csv_df.rename(columns=lambda x: f"c_{x}")

    notes_df = notes_df.reset_index()
    csv_df = csv_df.reset_index(drop=True)

    merged_rows = []
    notes_idx = 0
    csv_idx = 0

    while notes_idx < len(notes_df) and csv_idx < len(csv_df):
        midi_pitch = notes_df.loc[notes_idx, "midi_pitch"]
        c_pitch = csv_df.loc[csv_idx, "c_pitch"]
        if midi_pitch == c_pitch:
            # Merge the rows
            merged_row = pd.concat(
                [notes_df.loc[notes_idx], csv_df.loc[csv_idx]]
            )
            merged_rows.append(merged_row)
            notes_idx += 1
            csv_idx += 1
        else:
            # Skip this note, try next note with same csv row
            notes_idx += 1

    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        # Restore original index if needed
        if "index" in merged_df.columns:
            merged_df = merged_df.set_index("index")
        return merged_df
    else:
        return pd.DataFrame()  # No matches found


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
    print("No divergence found.")
    return None  # No divergence found


for f in score_mei_file_names[:10]:
    score_mei_path = score_mei_folder_path + f

    chordified_score = parse_and_chordify(score_mei_path)
    score_measures, score_chords, score_notes_df = extract_notes(
        chordified_score
    )

    print("\n\n#############################\n")

    print(f"Score MEI file: {score_mei_path}.")
    dipl_mei_path = get_matching_file_path(
        f,
        dipl_mei_file_names,
        base_replace_str="-dipl",
        folder_path=dipl_mei_folder_path,
    )
    if not dipl_mei_path:
        print(f"### No matching diplomatic MEI file found for {f}.")
        continue

    dipl_measures, dipl_chords, dipl_notes_df = extract_notes(
        parse_and_chordify(dipl_mei_path)
    )

    print(f"Diplomatic MEI file: {dipl_mei_path}.")

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
    print(f"Using csv-file: {csv_file_path}.")
    merged_df = merge_with_csv(dipl_notes_df, csv_df)
    # print(merged_df.head(30))

    error = find_error_in_df(merged_df)
    if error is not None:
        print(f"*** Error found in {f}: {error}\n")
        continue

    print(f"# rows in Score:   {len(score_notes_df)}")
    print(f"# rows in CSV:     {len(csv_df)}")
    print(f"# rows in Dipl:    {len(dipl_notes_df)}")
    print(f"# rows in merged:  {len(merged_df)}")

    print(merged_df.head())

    print("\n#############################\n\n")
