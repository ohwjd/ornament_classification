# import music21 as m21

import pandas as pd
import os

from search_for_rules.util import (
    extract_tbp_basic_metadata,
    get_matching_file_path,
    write_summary,
)
from search_for_rules.search_operations import (
    find_ornament_sequences_abtab,
    find_ornament_sequences_raw,
    filter_four_note_step_sequences,
)

from search_for_rules.preprocess import (
    preprocess_df,
    split_df_by_voice,
)


csv_folder_path = "data/tabmapper_output/mapping_csvs/"
# score_mei_folder_path = "data/tabmapper_output/voiced_meis/"
dipl_mei_folder_path = "data/meis_dipl/"
tab_folder_path = "data/tabs/"

csv_file_names = sorted(
    [f for f in os.listdir(csv_folder_path) if f.endswith(".csv")]
)

# score_mei_file_names = sorted(
#     [f for f in os.listdir(score_mei_folder_path) if f.endswith(".mei")]
# )

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
    print(f"Processing file {csv_file_path}.")
    csv_df = pd.read_csv(csv_file_path)
    base_name = os.path.splitext(f)[0].replace("-mapping", "")

    matching_tab_file = get_matching_file_path(
        f,
        tab_file_names,
        second_replace_str="",
        folder_path=tab_folder_path,
    )

    # matching_score_mei_file = get_matching_file_path(
    #     f,
    #     score_mei_file_names,
    #     second_replace_str="-score",
    #     folder_path=score_mei_folder_path,
    # )

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

    fourstep_dir = os.path.join(file_output_dir, "fourstep")
    os.makedirs(fourstep_dir, exist_ok=True)

    voices_dir = os.path.join(file_output_dir, "voices")
    os.makedirs(voices_dir, exist_ok=True)

    # Preprocessing already expanded multi-voice rows and added voice_order
    voice_dfs = split_df_by_voice(preprocessed_df)

    # max_ornament_duration_threshold is set to a quarter of the meter_info or two levels below beat level (in accordance to JosquinTab data set conventions)
    combined_sequences = find_ornament_sequences_abtab(
        preprocessed_df,
        allow_variable_durations=False,
        max_ornament_duration_threshold=meter_info / 4,
    )
    # Track counts for summary
    summary_counts = {}

    raw_sequences = find_ornament_sequences_raw(preprocessed_df)

    if not combined_sequences.empty:
        voice_tab_path = os.path.join(file_output_dir, f"{base_name}_tab.csv")
        combined_sequences.to_csv(voice_tab_path, index=False)
        summary_counts["tab"] = combined_sequences["sequence_id"].nunique()
        # Four-note variant for combined
        voice_tab_four = filter_four_note_step_sequences(combined_sequences)
        if not voice_tab_four.empty:
            voice_tab_four_path = os.path.join(
                fourstep_dir,
                f"{base_name}_tab_fourstep.csv",
            )
            voice_tab_four.to_csv(voice_tab_four_path, index=False, sep=";")
            summary_counts["tab_fourstep"] = voice_tab_four[
                "sequence_id"
            ].nunique()
        else:
            summary_counts["tab_fourstep"] = 0

    else:
        summary_counts.update(
            {
                "tab": 0,
                "tab_fourstep": 0,
            }
        )

    if not raw_sequences.empty:
        raw_tab_path = os.path.join(file_output_dir, f"{base_name}_tab_raw.csv")
        raw_sequences.to_csv(raw_tab_path, index=False)
        summary_counts["tab_raw"] = raw_sequences["sequence_id"].nunique()
        raw_four = filter_four_note_step_sequences(raw_sequences)
        if not raw_four.empty:
            raw_four_path = os.path.join(
                fourstep_dir,
                f"{base_name}_tab_raw_fourstep.csv",
            )
            raw_four.to_csv(raw_four_path, index=False, sep=";")
            summary_counts["tab_raw_fourstep"] = raw_four[
                "sequence_id"
            ].nunique()
        else:
            summary_counts["tab_raw_fourstep"] = 0
    else:
        summary_counts["tab_raw"] = 0
        summary_counts["tab_raw_fourstep"] = 0

    merged_by_voice_parts = []
    merged_sequence_offset = 0

    for voice, df_voice in voice_dfs.items():
        output_path = os.path.join(voices_dir, f"{base_name}_voice-{voice}.csv")
        sequences_df = find_ornament_sequences_abtab(
            df_voice, allow_variable_durations=False
        )

        # Save only if there are sequences
        if not sequences_df.empty:
            sequences_df.to_csv(output_path, index=False)
            summary_counts[f"voice_{voice}"] = sequences_df[
                "sequence_id"
            ].nunique()
            # Second step: filter for four-note small-step sequences
            filtered_four = filter_four_note_step_sequences(sequences_df)
            filtered_path = os.path.join(
                voices_dir,
                f"{base_name}_voice-{voice}_fourstep.csv",
            )
            if not filtered_four.empty:
                filtered_four.to_csv(filtered_path, index=False, sep=";")
                summary_counts[f"voice_{voice}_fourstep"] = filtered_four[
                    "sequence_id"
                ].nunique()
            else:
                summary_counts[f"voice_{voice}_fourstep"] = 0

            voice_part = sequences_df.copy()
            voice_part["sequence_id_voice"] = voice_part["sequence_id"]
            voice_part["sequence_id"] = (
                voice_part["sequence_id"] + merged_sequence_offset
            )
            merged_sequence_offset = (
                voice_part["sequence_id"].max() + 1
                if not voice_part.empty
                else merged_sequence_offset
            )
            voice_part["voice"] = str(voice)
            merged_by_voice_parts.append(voice_part)
        else:
            summary_counts[f"voice_{voice}"] = 0
            summary_counts[f"voice_{voice}_fourstep"] = 0

    if merged_by_voice_parts:
        merged_by_voice = pd.concat(
            merged_by_voice_parts, axis=0, ignore_index=True
        )
        sort_cols = ["voice", "sequence_id"]
        if "onset" in merged_by_voice.columns:
            sort_cols.append("onset")
        merged_by_voice = merged_by_voice.sort_values(
            sort_cols, kind="stable"
        ).reset_index(drop=True)
        merged_path = os.path.join(
            file_output_dir,
            f"{base_name}_voices_merged.csv",
        )
        merged_by_voice.to_csv(merged_path, index=False)
        summary_counts["voices_merged"] = merged_by_voice[
            "sequence_id"
        ].nunique()

        merged_four = filter_four_note_step_sequences(merged_by_voice)
        if not merged_four.empty:
            merged_four_path = os.path.join(
                fourstep_dir,
                f"{base_name}_voices_merged_fourstep.csv",
            )
            merged_four.to_csv(merged_four_path, index=False, sep=";")
            summary_counts["voices_merged_fourstep"] = merged_four[
                "sequence_id"
            ].nunique()
        else:
            summary_counts["voices_merged_fourstep"] = 0
    else:
        summary_counts["voices_merged"] = 0
        summary_counts["voices_merged_fourstep"] = 0

    write_summary(
        file_output_dir,
        base_name,
        tuning,
        meter_raw,
        bar_num,
        diminution,
        summary_counts,
    )
