import pandas as pd
from fractions import Fraction
from typing import Dict, List, Tuple


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
