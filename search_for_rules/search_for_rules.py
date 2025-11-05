# import music21 as m21

import pandas as pd
import os

from fractions import Fraction
from search_for_rules.util import (
    extract_tbp_basic_metadata,
    get_matching_file_path,
)
from search_for_rules.search_operations import (
    find_ornament_sequences,
    filter_four_note_step_sequences,
)

from search_for_rules.preprocess import (
    expand_and_split_voices,
    squish_to_one_staff,
    split_df_by_voice,
)


def preprocess_df(df):

    # convert string durations to numbers
    df["duration"] = df["duration"].apply(Fraction)
    # df["dur_float"] = df["duration"].apply(float)
    df["onset"] = df["onset"].apply(Fraction)

    # remove space from column name
    df.rename(columns={"mapped voice": "voice"}, inplace=True)

    df = df.drop(
        columns=["bar"]
    )  # remove bar column, could contain wrong information (when inconsistencies in meter occur (e.g., 3/2 bar in 2/2 meter))
    df = df.drop(columns=["cost"])  # remove cost column

    # Expand multi-voice rows before computing vertical ordering so each duplicated row can receive an appropriate rank (otherwise a combined "0 and 1" row would be treated as a single note at its onset and incorrectly get order 0 only).
    df = expand_and_split_voices(df)

    # Add per-onset vertical ordering index (voice_order)
    df = squish_to_one_staff(df)
    return df


def add_chord_context_to_sequences(
    sequences_no_context: pd.DataFrame, original_df: pd.DataFrame
):
    """Given sequences WITHOUT context (is_context column present but all False or absent),
    add the full chord (all notes) at the immediate next distinct onset after the sequence
    as context. The previous chord (before the sequence) is NOT added.

    Returns a new DataFrame with is_context boolean properly set.
    """
    if sequences_no_context.empty:
        return sequences_no_context

    # Ensure is_context column exists
    if "is_context" not in sequences_no_context.columns:
        sequences_no_context = sequences_no_context.copy()
        sequences_no_context["is_context"] = False

    # Work on a copy to avoid mutating inputs
    if "voice_squished" in original_df.columns:
        original_sorted = original_df.sort_values(
            ["onset", "voice_squished"], kind="stable"
        )
    elif "voice_order" in original_df.columns:
        original_sorted = original_df.sort_values(
            ["onset", "voice_order"], kind="stable"
        )
    else:
        original_sorted = original_df.sort_values("onset", kind="stable")
    all_onsets = original_sorted["onset"].drop_duplicates().tolist()

    augmented_groups = []
    for seq_id, group in sequences_no_context.groupby("sequence_id"):
        group = group.sort_values("onset", kind="stable")
        end_onset = group["onset"].max()
        # Locate the next onset value only (post-sequence). Do NOT include previous onset.
        next_onset_candidates = [o for o in all_onsets if o > end_onset]
        next_onset = next_onset_candidates[0] if next_onset_candidates else None

        context_rows = []
        if next_onset is not None:
            next_rows = original_sorted[original_sorted["onset"] == next_onset]
            if not next_rows.empty:
                next_rows_ctx = next_rows.copy()
                next_rows_ctx["sequence_id"] = seq_id
                next_rows_ctx["is_context"] = True
                context_rows.append(next_rows_ctx)

        seq_core = group.copy()
        seq_core["is_context"] = False  # ensure

        if context_rows:
            combined = pd.concat(
                [seq_core] + context_rows, axis=0, ignore_index=True
            )
        else:
            combined = seq_core

        # Remove duplicates if any (same original row appearing twice as context/core)
        if "pitch" in combined.columns:
            combined = combined.drop_duplicates(
                subset=["sequence_id", "onset", "pitch", "duration", "voice"],
                keep="first",
            )
        else:
            combined = combined.drop_duplicates(keep="first")

        augmented_groups.append(combined)

    result = pd.concat(augmented_groups, axis=0, ignore_index=True)
    # Order by sequence then onset then voice_order if available
    sort_cols = ["sequence_id", "onset"]
    if "voice_squished" in result.columns:
        sort_cols.append("voice_squished")
    elif "voice_order" in result.columns:
        sort_cols.append("voice_order")
    result = result.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return result


csv_folder_path = "data/tabmapper_output/mapping_csvs/"
score_mei_folder_path = "data/tabmapper_output/voiced_meis/"
dipl_mei_folder_path = "data/meis_dipl/"
tab_folder_path = "data/tabs/"

csv_file_names = sorted(
    [f for f in os.listdir(csv_folder_path) if f.endswith(".csv")]
)

score_mei_file_names = sorted(
    [f for f in os.listdir(score_mei_folder_path) if f.endswith(".mei")]
)

dipl_mei_file_names = sorted(
    [f for f in os.listdir(dipl_mei_folder_path) if f.endswith(".mei")]
)

tab_file_names = sorted(
    [f for f in os.listdir(tab_folder_path) if f.endswith(".tbp")]
)


for f in [
    f
    for f in csv_file_names
    if f.startswith("4400_45_ach_unfall") or f.startswith("4481_49_ach_unfal")
]:
    csv_file_path = csv_folder_path + f
    print(f"Processing file {csv_file_path}:")
    csv_df = pd.read_csv(csv_file_path)
    base_name = os.path.splitext(f)[0].replace("-mapping", "")

    matching_tab_file = get_matching_file_path(
        f,
        tab_file_names,
        second_replace_str="",
        folder_path=tab_folder_path,
    )

    matching_score_mei_file = get_matching_file_path(
        f,
        score_mei_file_names,
        second_replace_str="-score",
        folder_path=score_mei_folder_path,
    )

    matching_dipl_mei_file = get_matching_file_path(
        f,
        dipl_mei_file_names,
        second_replace_str="-dipl",
        folder_path=dipl_mei_folder_path,
    )
    if matching_tab_file is None:
        print(
            "  No matching .tbp file found, skipping TBP metadata extraction."
        )
        continue

    tuning, meter_info, bar_num, diminution, meter_raw = (
        extract_tbp_basic_metadata(matching_tab_file)
    )

    # initial preprocessing
    preprocessed_df = preprocess_df(csv_df)

    # Prepare per-file output subfolder: output/<base_name>/
    output_root = "output"
    file_output_dir = os.path.join(output_root, base_name)
    os.makedirs(file_output_dir, exist_ok=True)

    # Preprocessing already expanded multi-voice rows and added voice_order
    voice_dfs = split_df_by_voice(preprocessed_df)

    # Also process the combined dataframe (all voices) using voice_order / voice_squished
    # max_ornament_duration_threshold is set to a quarter of the meter_info or two levels below beat level (in accordance to JosquinTab data set conventions)
    combined_sequences = find_ornament_sequences(
        preprocessed_df,
        allow_variable_durations=False,
        max_ornament_duration_threshold=meter_info / 4,
    )
    # Track counts for summary
    summary_counts = {}

    if not combined_sequences.empty:
        combined_path = os.path.join(
            file_output_dir, f"{base_name}_sequences_allvoices.csv"
        )
        combined_sequences.to_csv(combined_path, index=False)
        summary_counts["allvoices"] = combined_sequences[
            "sequence_id"
        ].nunique()
        # Four-note variant for combined
        combined_four = filter_four_note_step_sequences(
            combined_sequences, original_df=preprocessed_df
        )
        if not combined_four.empty:
            combined_four_path = os.path.join(
                file_output_dir, f"{base_name}_sequences_fourstep_allvoices.csv"
            )
            combined_four.to_csv(combined_four_path, index=False)
            summary_counts["allvoices_fourstep"] = combined_four[
                "sequence_id"
            ].nunique()
        else:
            summary_counts["allvoices_fourstep"] = 0

        # Chord context alternative: first strip existing context then add chord context
        core_only = combined_sequences[~combined_sequences["is_context"]].copy()
        chord_context_sequences = add_chord_context_to_sequences(
            core_only, preprocessed_df
        )
        chord_context_path = os.path.join(
            file_output_dir, f"{base_name}_sequences_allvoices_chordcontext.csv"
        )
        chord_context_sequences.to_csv(chord_context_path, index=False)
        summary_counts["allvoices_chordcontext"] = chord_context_sequences[
            "sequence_id"
        ].nunique()
        chord_context_four = filter_four_note_step_sequences(
            chord_context_sequences, original_df=preprocessed_df
        )
        if not chord_context_four.empty:
            chord_context_four_path = os.path.join(
                file_output_dir,
                f"{base_name}_sequences_fourstep_allvoices_chordcontext.csv",
            )
            chord_context_four.to_csv(chord_context_four_path, index=False)
            summary_counts["allvoices_chordcontext_fourstep"] = (
                chord_context_four["sequence_id"].nunique()
            )
        else:
            summary_counts["allvoices_chordcontext_fourstep"] = 0
    else:
        summary_counts.update(
            {
                "allvoices": 0,
                "allvoices_fourstep": 0,
                "allvoices_chordcontext": 0,
                "allvoices_chordcontext_fourstep": 0,
            }
        )

    for voice, df_voice in voice_dfs.items():
        output_path = os.path.join(
            file_output_dir, f"{base_name}_sequences_{voice}.csv"
        )
        sequences_df = find_ornament_sequences(
            df_voice, allow_variable_durations=False
        )

        # Save only if there are sequences
        if not sequences_df.empty:
            sequences_df.to_csv(output_path, index=False)
            summary_counts[f"voice_{voice}"] = sequences_df[
                "sequence_id"
            ].nunique()
            # Second step: filter for four-note small-step sequences
            filtered_four = filter_four_note_step_sequences(
                sequences_df, original_df=preprocessed_df
            )
            filtered_path = os.path.join(
                file_output_dir, f"{base_name}_sequences_fourstep_{voice}.csv"
            )
            if not filtered_four.empty:
                filtered_four.to_csv(filtered_path, index=False)
                summary_counts[f"voice_{voice}_fourstep"] = filtered_four[
                    "sequence_id"
                ].nunique()
            else:
                summary_counts[f"voice_{voice}_fourstep"] = 0
        else:
            summary_counts[f"voice_{voice}"] = 0
            summary_counts[f"voice_{voice}_fourstep"] = 0

    # Write per-file summary (include basic TBP header info)
    summary_lines = [f"Summary for {base_name}"]
    summary_lines.append(f"tuning: {tuning}")
    summary_lines.append(f"meter_info: {meter_raw}")
    summary_lines.append(f"bar_num: {bar_num}")
    summary_lines.append(f"diminution: {diminution}")
    for key, val in sorted(summary_counts.items()):
        summary_lines.append(f"{key}: {val}")
    summary_path = os.path.join(file_output_dir, f"{base_name}_summary.txt")
    with open(summary_path, "w") as sf:
        sf.write("\n".join(summary_lines) + "\n")
    print(f"Summary written to {summary_path}")
