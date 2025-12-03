import os
from typing import Dict, List

import pandas as pd

from search_for_rules.util import (
    extract_tbp_basic_metadata,
    get_matching_file_path,
    write_summary,
)
from search_for_rules.search_operations import (
    find_ornament_sequences_abtab,
    find_ornament_sequences_raw,
)

from search_for_rules.filters import (
    filter_four_note_step_sequences,
    filter_non_chord_sequences,
    # only_starting_chord_and_then_non_chord_sequences,
)

from search_for_rules.counters import (
    count_ornaments_by_length,
    count_ornaments_by_duration,
    count_abrupt_duration_changes,
    count_consonant_beginning_sequences,
)

from search_for_rules.preprocess import (
    preprocess_df,
    split_df_by_voice,
)


csv_folder_path = "data/tabmapper_output/mapping_csvs/"
# score_mei_folder_path = "data/tabmapper_output/voiced_meis/"
dipl_mei_folder_path = "data/meis_dipl/"
tab_folder_path = "data/tabs/"

category_csv_path = "data/josquintab_file_categories.csv"
category_lookup_df = pd.read_csv(category_csv_path)

csv_file_names = sorted(
    [f for f in os.listdir(csv_folder_path) if f.endswith(".csv")]
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
    # if f.startswith("4400_45_ach_unfall") or f.startswith("4481_49_ach_unfal")
]:
    csv_file_path = csv_folder_path + f
    print(f"Processing file {csv_file_path}.")
    csv_df = pd.read_csv(csv_file_path)
    base_name = os.path.splitext(f)[0].replace("-mapping", "")
    category_row = category_lookup_df[category_lookup_df["file_name"] == f]
    category = (
        category_row["category"].values[0]
        if not category_row.empty
        else "Unknown"
    )
    matching_tab_file = get_matching_file_path(
        f,
        tab_file_names,
        second_replace_str="",
        folder_path=tab_folder_path,
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

    # Track counts for summary
    summary_counts = {}
    length_summary_records: List[Dict[str, object]] = []
    duration_summary_records: List[Dict[str, object]] = []

    def record_length_summary(
        output_path: str, sequences_df: pd.DataFrame
    ) -> None:
        counts, length_seq_ids = count_ornaments_by_length(sequences_df)
        if counts:
            for seq_length, count in sorted(counts.items()):
                seq_ids = length_seq_ids.get(seq_length, [])
                length_summary_records.append(
                    {
                        "output_csv": output_path,
                        "sequence_length": int(seq_length),
                        "count": int(count),
                        "sequence_ids": ",".join(
                            [str(seq_id) for seq_id in seq_ids]
                        ),
                    }
                )
        else:
            length_summary_records.append(
                {
                    "output_csv": output_path,
                    "sequence_length": pd.NA,
                    "count": 0,
                    "sequence_ids": "",
                }
            )

    def record_duration_summary(
        output_path: str, sequences_df: pd.DataFrame
    ) -> None:
        counts, duration_seq_ids = count_ornaments_by_duration(sequences_df)
        if counts:
            for duration_value, count in sorted(
                counts.items(),
                key=lambda item: (
                    float(item[0]) if hasattr(item[0], "__float__") else item[0]
                ),
            ):
                seq_ids = duration_seq_ids.get(duration_value, [])
                duration_summary_records.append(
                    {
                        "output_csv": output_path,
                        "duration": str(duration_value),
                        "count": int(count),
                        "sequence_ids": ",".join(
                            [str(seq_id) for seq_id in seq_ids]
                        ),
                    }
                )
        else:
            duration_summary_records.append(
                {
                    "output_csv": output_path,
                    "duration": pd.NA,
                    "count": 0,
                    "sequence_ids": "",
                }
            )

    #####################################################

    # abtab
    # max_ornament_duration_threshold is set to a quarter of the meter_info or two levels below beat level (in accordance to JosquinTab data set conventions)
    ornament_sequences_abtab = find_ornament_sequences_abtab(
        preprocessed_df,
        max_ornament_duration_threshold=meter_info / 4,
    )
    if not ornament_sequences_abtab.empty:
        voice_tab_path = os.path.join(
            file_output_dir, f"{base_name}_tab_abtab.csv"
        )
        ornament_sequences_abtab.to_csv(voice_tab_path, index=False)
        record_length_summary(voice_tab_path, ornament_sequences_abtab)
        record_duration_summary(voice_tab_path, ornament_sequences_abtab)
        summary_counts["abtab"] = ornament_sequences_abtab[
            "sequence_id"
        ].nunique()
        abrupt_count, abrupt_ids = count_abrupt_duration_changes(
            ornament_sequences_abtab
        )
        (
            consonant_count,
            non_consonant_ids,
        ) = count_consonant_beginning_sequences(ornament_sequences_abtab)
        summary_counts["abtab_count_abrupt_duration_changes"] = abrupt_count
        summary_counts["abtab_count_consonant_beginning_sequences"] = (
            consonant_count
        )
        if abrupt_count > 0:
            summary_counts["abtab_abrupt_duration_change_sequence_ids"] = (
                "[" + ",".join(str(seq_id) for seq_id in abrupt_ids) + "]"
            )
        if non_consonant_ids:
            summary_counts["abtab_non_consonant_beginning_sequence_ids"] = (
                "["
                + ",".join(str(seq_id) for seq_id in non_consonant_ids)
                + "]"
            )
        voice_tab_four = filter_four_note_step_sequences(
            ornament_sequences_abtab
        )
        if not voice_tab_four.empty:
            voice_tab_four_path = os.path.join(
                fourstep_dir,
                f"{base_name}_tab_abtab_fourstep.csv",
            )
            voice_tab_four.to_csv(voice_tab_four_path, index=False, sep=";")
            record_length_summary(voice_tab_four_path, voice_tab_four)
            record_duration_summary(voice_tab_four_path, voice_tab_four)
            summary_counts["abtab_fourstep"] = voice_tab_four[
                "sequence_id"
            ].nunique()
        else:
            summary_counts["abtab_fourstep"] = 0

        abtab_non_chord = filter_non_chord_sequences(ornament_sequences_abtab)
        if not abtab_non_chord.empty:
            abtab_non_chord_path = os.path.join(
                fourstep_dir,
                f"{base_name}_tab_abtab_non_chord.csv",
            )
            abtab_non_chord.to_csv(abtab_non_chord_path, index=False, sep=";")
            record_length_summary(abtab_non_chord_path, abtab_non_chord)
            record_duration_summary(abtab_non_chord_path, abtab_non_chord)
            summary_counts["abtab_non_chord"] = abtab_non_chord[
                "sequence_id"
            ].nunique()
        else:
            summary_counts["abtab_non_chord"] = 0

    else:
        summary_counts.update(
            {
                "abtab": 0,
                "abtab_fourstep": 0,
                "abtab_count_abrupt_duration_changes": 0,
                "abtab_count_consonant_beginning_sequences": 0,
                "abtab_non_chord": 0,
            }
        )

    #############################################
    # raw (not abtab based)

    ornament_sequences_raw = find_ornament_sequences_raw(preprocessed_df)

    if not ornament_sequences_raw.empty:
        raw_tab_path = os.path.join(file_output_dir, f"{base_name}_tab_raw.csv")
        ornament_sequences_raw.to_csv(raw_tab_path, index=False)
        record_length_summary(raw_tab_path, ornament_sequences_raw)
        record_duration_summary(raw_tab_path, ornament_sequences_raw)
        summary_counts["raw"] = ornament_sequences_raw["sequence_id"].nunique()
        raw_abrupt_count, raw_abrupt_ids = count_abrupt_duration_changes(
            ornament_sequences_raw
        )
        (
            raw_consonant_count,
            raw_non_consonant_ids,
        ) = count_consonant_beginning_sequences(ornament_sequences_raw)
        summary_counts["raw_count_abrupt_duration_changes"] = raw_abrupt_count
        summary_counts["raw_count_consonant_beginning_sequences"] = (
            raw_consonant_count
        )
        if raw_abrupt_count > 0:
            summary_counts["raw_abrupt_duration_change_sequence_ids"] = (
                "[" + ",".join(str(seq_id) for seq_id in raw_abrupt_ids) + "]"
            )
        if raw_non_consonant_ids:
            summary_counts["raw_non_consonant_beginning_sequence_ids"] = (
                "["
                + ",".join(str(seq_id) for seq_id in raw_non_consonant_ids)
                + "]"
            )
        raw_four = filter_four_note_step_sequences(ornament_sequences_raw)
        if not raw_four.empty:
            raw_four_path = os.path.join(
                fourstep_dir,
                f"{base_name}_tab_raw_fourstep.csv",
            )
            raw_four.to_csv(raw_four_path, index=False, sep=";")
            record_length_summary(raw_four_path, raw_four)
            record_duration_summary(raw_four_path, raw_four)
            summary_counts["raw_fourstep"] = raw_four["sequence_id"].nunique()
        else:
            summary_counts["raw_fourstep"] = 0
        raw_non_chord = filter_non_chord_sequences(ornament_sequences_raw)
        if not raw_non_chord.empty:
            raw_non_chord_path = os.path.join(
                fourstep_dir,
                f"{base_name}_tab_raw_non_chord.csv",
            )
            raw_non_chord.to_csv(raw_non_chord_path, index=False, sep=";")
            record_length_summary(raw_non_chord_path, raw_non_chord)
            record_duration_summary(raw_non_chord_path, raw_non_chord)
            summary_counts["raw_non_chord"] = raw_non_chord[
                "sequence_id"
            ].nunique()
        else:
            summary_counts["raw_non_chord"] = 0

    else:
        summary_counts["raw"] = 0
        summary_counts["raw_fourstep"] = 0
        summary_counts["raw_count_abrupt_duration_changes"] = 0
        summary_counts["raw_count_consonant_beginning_sequences"] = 0

    merged_by_voice_parts = []
    merged_sequence_offset = 0
    merged_eighth_sequence_offset = 0

    for voice, df_voice in voice_dfs.items():
        output_path = os.path.join(voices_dir, f"{base_name}_voice-{voice}.csv")
        sequences_df = find_ornament_sequences_abtab(df_voice)

        if not sequences_df.empty:
            sequences_df.to_csv(output_path, index=False)
            record_length_summary(output_path, sequences_df)
            record_duration_summary(output_path, sequences_df)
            summary_counts[f"voice_{voice}"] = sequences_df[
                "sequence_id"
            ].nunique()
            filtered_four = filter_four_note_step_sequences(sequences_df)
            filtered_path = os.path.join(
                voices_dir,
                f"{base_name}_voice-{voice}_fourstep.csv",
            )
            if not filtered_four.empty:
                filtered_four.to_csv(filtered_path, index=False, sep=";")
                record_length_summary(filtered_path, filtered_four)
                record_duration_summary(filtered_path, filtered_four)
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
        record_length_summary(merged_path, merged_by_voice)
        record_duration_summary(merged_path, merged_by_voice)
        summary_counts["voices_merged"] = merged_by_voice[
            "sequence_id"
        ].nunique()
        summary_counts["voices_merged_count_abrupt_duration_changes"] = 0

        merged_abrupt_count, merged_abrupt_ids = count_abrupt_duration_changes(
            merged_by_voice
        )
        summary_counts["voices_merged_count_abrupt_duration_changes"] = (
            merged_abrupt_count
        )
        if merged_abrupt_ids:
            summary_counts[
                "voices_merged_abrupt_duration_change_sequence_ids"
            ] = (
                "["
                + ",".join(str(seq_id) for seq_id in merged_abrupt_ids)
                + "]"
            )

        merged_four = filter_four_note_step_sequences(merged_by_voice)
        if not merged_four.empty:
            merged_four_path = os.path.join(
                fourstep_dir,
                f"{base_name}_voices_merged_fourstep.csv",
            )
            merged_four.to_csv(merged_four_path, index=False, sep=";")
            record_length_summary(merged_four_path, merged_four)
            record_duration_summary(merged_four_path, merged_four)
            summary_counts["voices_merged_fourstep"] = merged_four[
                "sequence_id"
            ].nunique()
        else:
            summary_counts["voices_merged_fourstep"] = 0
    else:
        summary_counts["voices_merged"] = 0
        summary_counts["voices_merged_fourstep"] = 0

    summary_path = write_summary(
        file_output_dir,
        base_name,
        tuning,
        meter_raw,
        bar_num,
        diminution,
        summary_counts,
        category,
    )

    summary_csv_path = os.path.join(
        file_output_dir, f"{base_name}_length_summary.csv"
    )
    if length_summary_records:
        length_summary_df = pd.DataFrame(length_summary_records)
    else:
        length_summary_df = pd.DataFrame(
            columns=[
                "output_csv",
                "sequence_length",
                "count",
                "sequence_ids",
            ]
        )
    length_summary_df.to_csv(summary_csv_path, index=False)

    duration_summary_csv_path = os.path.join(
        file_output_dir, f"{base_name}_duration_summary.csv"
    )
    if duration_summary_records:
        duration_summary_df = pd.DataFrame(duration_summary_records)
    else:
        duration_summary_df = pd.DataFrame(
            columns=[
                "output_csv",
                "duration",
                "count",
                "sequence_ids",
            ]
        )
    duration_summary_df.to_csv(duration_summary_csv_path, index=False)
