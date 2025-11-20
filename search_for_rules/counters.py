import pandas as pd
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


def _normalize_duration(value) -> Optional[Fraction]:
    """Return a Fraction representation of a duration value when possible."""
    if pd.isna(value):
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, Fraction)):
        try:
            return Fraction(value)
        except Exception:
            return None
    if isinstance(value, float):
        try:
            return Fraction(value).limit_denominator()
        except Exception:
            return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        if "/" in text:
            num_str, den_str = text.split("/", 1)
            return Fraction(int(num_str.strip()), int(den_str.strip()))
        return Fraction(float(text)).limit_denominator()
    except Exception:
        return None


def count_ornaments_by_length(
    sequences_df: pd.DataFrame,
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """
    Count the number of unique ornament sequences by their core ornament note length.

    Parameters
    ----------
    sequences_df : pd.DataFrame
        Output of find_ornament_sequences (may contain multiple sequence_id values).

    Returns
    -------
        Tuple[Dict[int, int], Dict[int, List[int]]]
                A pair where:
                        - The first dictionary maps core ornament note lengths to counts of unique
                            sequences exhibiting that length.
                        - The second dictionary maps each core-length bucket to the list of
                            sequence_ids contributing to that length.
    """
    length_counts: Dict[int, int] = {}
    length_to_sequence_ids: Dict[int, List[int]] = {}

    if sequences_df.empty:
        return length_counts, length_to_sequence_ids

    for seq_id, group in sequences_df.groupby("sequence_id"):
        non_context = group[~group["is_context"]]
        if non_context.empty:
            continue
        if "onset" in non_context.columns:
            core_onsets = non_context[non_context["onset"].notna()]["onset"]
            core_length = int(core_onsets.nunique())

            if core_length == 0:
                # No valid onset values were available; deduplicate by onset to ensure
                # only one representative note per onset (NaNs collapse together).
                core_length = int(
                    non_context.drop_duplicates(subset=["onset"]).shape[0]
                )
        else:
            # Without onset information, fall back to unique row indices (already
            # filtered to non-context notes).
            core_length = int(non_context.index.nunique())

        if core_length == 0:
            continue

        if core_length not in length_counts:
            length_counts[core_length] = 0
        length_counts[core_length] += 1
        if core_length not in length_to_sequence_ids:
            length_to_sequence_ids[core_length] = []
        length_to_sequence_ids[core_length].append(int(seq_id))

    for seq_list in length_to_sequence_ids.values():
        seq_list.sort()

    return length_counts, length_to_sequence_ids


def count_ornaments_by_duration(
    sequences_df: pd.DataFrame,
) -> Tuple[Dict[Fraction, int], Dict[Fraction, List[int]]]:
    """
    Count the number of unique ornament sequences by their common duration value.

    Parameters
    ----------
    sequences_df : pd.DataFrame
        Output of find_ornament_sequences (may contain multiple sequence_id values).

    Returns
    -------
        Tuple[Dict[Fraction, int], Dict[Fraction, List[int]]]
                A pair where:
                        - The first dictionary maps common duration values to counts of
                          unique sequences exhibiting that duration.
                        - The second dictionary maps each duration bucket to the list of
                          sequence_ids contributing to that duration.
    """
    duration_counts: Dict[Fraction, int] = {}
    duration_to_sequence_ids: Dict[Fraction, List[int]] = {}

    if sequences_df.empty:
        return duration_counts, duration_to_sequence_ids

    for seq_id, group in sequences_df.groupby("sequence_id"):
        non_context = group[~group["is_context"]]
        if non_context.empty or "duration" not in non_context.columns:
            continue
        durations = non_context["duration"].dropna().unique()
        if len(durations) != 1:
            continue
        duration_value = durations[0]
        if duration_value not in duration_counts:
            duration_counts[duration_value] = 0
        duration_counts[duration_value] += 1
        if duration_value not in duration_to_sequence_ids:
            duration_to_sequence_ids[duration_value] = []
        duration_to_sequence_ids[duration_value].append(int(seq_id))

    for seq_list in duration_to_sequence_ids.values():
        seq_list.sort()

    return duration_counts, duration_to_sequence_ids


def count_abrupt_duration_changes(
    sequences_df: pd.DataFrame,
) -> Tuple[int, List[int]]:
    """
    Count the number of unique ornament sequences exhibiting abrupt duration changes
    between the pre-context-onset-time duration and the duration of the first ornament note.

    An abrupt duration change is defined as a change in duration between the pre-context
    note and the first ornament note that spans at least two standard note-value steps
    (i.e., a factor of four or greater difference between their durations).

    Parameters
    ----------
    sequences_df : pd.DataFrame
        Output of find_ornament_sequences (may contain multiple sequence_id values).

    Returns
    -------
        Tuple[int, List[int]]
            Number of unique sequences exhibiting at least one abrupt duration change
            and the list of their sequence_id values.
    """
    abrupt_change_count = 0
    matching_sequence_ids: List[int] = []

    if sequences_df.empty:
        return abrupt_change_count, matching_sequence_ids

    for seq_id, group in sequences_df.groupby("sequence_id"):
        if "duration" not in group.columns:
            continue
        non_context = group[~group["is_context"]]
        if non_context.empty:
            continue

        first_core_idx = non_context.index.min()
        first_core_duration = _normalize_duration(
            group.loc[first_core_idx, "duration"]
        )
        if first_core_duration is None or first_core_duration <= 0:
            continue

        pre_context = group[
            (group["is_context"]) & (group.index < first_core_idx)
        ]
        pre_context = pre_context.dropna(subset=["duration"])
        if pre_context.empty:
            continue

        pre_context_duration = _normalize_duration(
            pre_context.loc[pre_context.index.max(), "duration"]
        )
        if pre_context_duration is None or pre_context_duration <= 0:
            continue

        longer = max(first_core_duration, pre_context_duration)
        shorter = min(first_core_duration, pre_context_duration)
        if shorter == 0:
            continue
        ratio = longer / shorter
        if ratio >= 8:
            # print(
            #     "[abrupt_duration_match] seq_id=%s pre_context=%s first_core=%s longer=%s shorter=%s ratio=%s"
            #     % (
            #         seq_id,
            #         pre_context_duration,
            #         first_core_duration,
            #         longer,
            #         shorter,
            #         ratio,
            #     )
            # )
            abrupt_change_count += 1
            matching_sequence_ids.append(int(seq_id))

    matching_sequence_ids.sort()
    return abrupt_change_count, matching_sequence_ids


def count_consonant_beginning_sequences(
    sequences_df: pd.DataFrame,
) -> Tuple[int, List[int]]:
    """
    Count the number of unique ornament sequences that begin with a consonant pitch span (or with a single note).

    A consonant span is defined as a unison, minor third, major third, perfect fourth,
    perfect fifth, minor sixth, major sixth, or octave, evaluated via MIDI semitone
    differences (absolute span and mod-12 class).

    Parameters
    ----------
    sequences_df : pd.DataFrame
        Output of find_ornament_sequences (may contain multiple sequence_id values).

    Returns
    -------
        Tuple[int, List[int]]
            Number of unique sequences that begin with a consonant interval and the list
            of sequence_id values that do not meet the consonant criterion.
    """
    consonant_intervals = {0, 3, 4, 5, 7, 8, 9, 12}
    consonant_count = 0
    non_consonant_sequence_ids: List[int] = []

    if sequences_df.empty:
        return consonant_count, non_consonant_sequence_ids

    for seq_id, group in sequences_df.groupby("sequence_id"):
        non_context = group[~group["is_context"]]
        if non_context.empty:
            continue

        first_core_idx = non_context.index.min()
        first_block = non_context.loc[[first_core_idx]]
        same_onset_rows = first_block

        if "onset" in non_context.columns:
            onset_series = non_context["onset"]
            block_onset = (
                first_block["onset"].iloc[0]
                if "onset" in first_block.columns
                else None
            )

            if pd.isna(block_onset) and onset_series.notna().any():
                block_onset = onset_series.dropna().min()

            if pd.isna(block_onset):
                same_onset_rows = first_block
            else:
                same_onset_rows = non_context[onset_series == block_onset]
                if same_onset_rows.empty:
                    same_onset_rows = first_block

        note_count = int(same_onset_rows.shape[0])

        interval_classes = set()
        if "pitch" in same_onset_rows.columns:
            pitch_series = same_onset_rows["pitch"].dropna()
            if not pitch_series.empty:
                pitches = [int(round(p)) for p in pitch_series.tolist()]
                if len(pitches) == 1:
                    interval_classes.add(0)
                else:
                    for i in range(len(pitches)):
                        for j in range(i + 1, len(pitches)):
                            diff = abs(pitches[i] - pitches[j])
                            interval_classes.add(diff)
                            interval_classes.add(diff % 12)

        # Evaluate all notes at the first onset to capture chord consonance modulo 12.
        is_consonant = any(
            interval_class in consonant_intervals
            for interval_class in interval_classes
        )

        if not is_consonant and not interval_classes and note_count == 1:
            is_consonant = True

        if is_consonant:
            consonant_count += 1
        else:
            non_consonant_sequence_ids.append(int(seq_id))

    non_consonant_sequence_ids.sort()
    return consonant_count, non_consonant_sequence_ids
