import pandas as pd
from fractions import Fraction
from typing import Any, Dict, List, Set


def find_ornament_sequences_raw(
    df: pd.DataFrame,
    onset_col: str = "onset",
    duration_col: str = "duration",
    min_sequence_length: int = 4,
    max_ornament_duration_threshold: Fraction = Fraction(1, 4),
) -> pd.DataFrame:
    """
    Identify ornament sequences purely from onset ordering and duration uniformity.

    A sequence is defined as at least `min_sequence_length` consecutive onsets (after
    sorting by onset) that all share the same duration value. Any onset in a
    qualifying span may contain multiple notes (a chord) provided every note at that
    onset shares the same duration. Durations are required to be homogeneous within
    each onset (all notes at that onset share the same duration).
    Candidate sequences are further restricted to those whose shared duration is
    strictly shorter than `max_ornament_duration_threshold`.

    Context handling:
      - the full chord (all rows) at the onsets immediately preceding and
        following each detected sequence is added as context where available.
      - Context rows are marked with `is_context = True`; core sequence rows remain
        `False`.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed tab DataFrame containing at least onset and duration columns.
    onset_col : str
        Column name holding onset positions (expected sortable, e.g., Fraction).
    duration_col : str
        Column containing duration values (Fractions or numbers).
    min_sequence_length : int
        Minimum number of consecutive onsets required to form a sequence.
    max_ornament_duration_threshold : Fraction or numeric, default Fraction(1, 4)
        Upper bound for acceptable shared duration across a candidate sequence.

    Returns
    -------
    pd.DataFrame
        Rows belonging to detected sequences with added columns:
          - sequence_id (int)
          - is_context (bool)
        An empty DataFrame (preserving schema) is returned when no sequences are found.
    """

    if df.empty:
        empty = df.head(0).copy()
        empty["sequence_id"] = pd.Series(dtype="Int64")
        empty["is_context"] = pd.Series(dtype="boolean")
        return empty

    if onset_col not in df.columns or duration_col not in df.columns:
        raise ValueError(
            f"Required columns '{onset_col}' and '{duration_col}' must be present."
        )

    sort_columns: List[str] = [onset_col]
    for candidate in ("voice_squished", "voice_order", "voice_tab", "voice"):
        if candidate in df.columns and candidate not in sort_columns:
            sort_columns.append(candidate)

    df = df.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    group_infos: List[Dict[str, Any]] = []
    for onset_value, group in df.groupby(onset_col, sort=True):
        durations = group[duration_col].dropna().unique()
        if len(durations) != 1:
            info = {
                "onset": onset_value,
                "indices": tuple(group.index),
                "duration": None,
                "homogeneous": False,
                "size": len(group),
            }
        else:
            info = {
                "onset": onset_value,
                "indices": tuple(group.index),
                "duration": durations[0],
                "homogeneous": True,
                "size": len(group),
            }
        group_infos.append(info)

    sequences: List[Dict[str, Any]] = []
    num_groups = len(group_infos)
    i = 0
    while i < num_groups:
        info = group_infos[i]
        duration_value = info["duration"]
        if not info["homogeneous"] or pd.isna(duration_value):
            i += 1
            continue

        try:
            duration_ok = duration_value < max_ornament_duration_threshold
        except TypeError:
            duration_ok = False

        if not duration_ok:
            i += 1
            continue

        run: List[Dict[str, Any]] = [info]
        j = i + 1
        while j < num_groups:
            next_info = group_infos[j]
            next_duration = next_info["duration"]
            if (
                not next_info["homogeneous"]
                or pd.isna(next_duration)
                or next_duration != duration_value
            ):
                break
            run.append(next_info)
            j += 1

        if len(run) >= min_sequence_length:
            sequences.append(
                {
                    "groups": run,
                    "start_group_idx": i,
                    "end_group_idx": j - 1,
                    "duration": duration_value,
                }
            )
            i = j
        else:
            i += 1

    if not sequences:
        empty = df.head(0).copy()
        empty["sequence_id"] = pd.Series(dtype="Int64")
        empty["is_context"] = pd.Series(dtype="boolean")
        return empty

    output_rows: List[pd.DataFrame] = []

    for seq_id, seq in enumerate(sequences):
        core_indices: List[int] = []
        for grp in seq["groups"]:
            core_indices.extend(grp["indices"])

        core_index_set: Set[int] = set(core_indices)
        context_indices: Set[int] = set()

        post_idx = seq["end_group_idx"] + 1
        if post_idx < num_groups:
            post_indices = set(group_infos[post_idx]["indices"])
            context_indices.update(
                idx for idx in post_indices if idx not in core_index_set
            )

        pre_idx = seq["start_group_idx"] - 1
        if pre_idx >= 0:
            pre_indices = set(group_infos[pre_idx]["indices"])
            context_indices.update(
                idx for idx in pre_indices if idx not in core_index_set
            )

        selected_indices = sorted(core_index_set | context_indices)

        seq_df = df.loc[selected_indices].copy()
        seq_df["sequence_id"] = seq_id
        seq_df["is_context"] = seq_df.index.isin(context_indices)
        output_rows.append(seq_df)

    combined = pd.concat(output_rows).reset_index(drop=True)
    sort_cols = ["sequence_id", onset_col]
    ascending_flags = [True, True]
    if "voice_squished" in combined.columns:
        sort_cols.append("voice_squished")
        ascending_flags.append(True)
    elif "voice_order" in combined.columns:
        sort_cols.append("voice_order")
        ascending_flags.append(True)
    elif "voice_tab" in combined.columns:
        sort_cols.append("voice_tab")
        ascending_flags.append(False)
    combined = combined.sort_values(
        sort_cols, ascending=ascending_flags, kind="stable"
    ).reset_index(drop=True)
    return combined


def find_ornament_sequences_abtab(
    df,
    max_ornament_duration_threshold=Fraction(1, 4),
    voice_col: str = "voice",
):
    """
    Identify ornament sequences and return a single DataFrame containing only the
    notes that belong to these sequences. Each note gains:
      - 'sequence_id' (zero-based)
      - 'is_context' (bool) True for added context notes (before/after), False otherwise.

    Definition of a sequence:
        - One or more consecutive notes where for at least one note the category == 'ornamentation' AND
            duration < max_ornament_duration_threshold.
        - All ornament notes in a sequence must share the same duration value.
        - Two ornament runs of the same duration separated
            by exactly one non-ornament note of that SAME duration are merged into a single sequence;
            that bridging note is treated as part of the sequence (is_context = False).
        - After a sequence is delimited, include the entire onset
            immediately preceding its first element and the entire onset
            immediately following its last element as context (is_context=True).
        - Any adjacent notes sharing the base ornament duration are absorbed into the sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Voice-specific DataFrame (assumed already filtered per voice and preprocessed).
    max_ornament_duration_threshold : Fraction or numeric, default 1/4
        Upper bound for ornament note durations.
    voice_col : str, default "voice"
        Column name indicating the voice label (kept for API symmetry; not modified).

    Returns
    -------
    pd.DataFrame
        Rows belonging to sequences (ornament + optional context) with added columns:
        sequence_id (int), is_context (bool). Empty DataFrame if none found.
    """

    if df.empty:
        empty = df.head(0).copy()
        empty["sequence_id"] = pd.Series(dtype="Int64")
        empty["is_context"] = pd.Series(dtype="boolean")
        return empty

    # Ensure natural ordering by onset if present; otherwise keep current order
    if "onset" in df.columns:
        df = df.sort_values("onset").reset_index(drop=True)

    def is_ornament(idx):
        row = df.loc[idx]
        if row.get("category") != "ornamentation":
            return False
        dur = row["duration"]
        return dur < max_ornament_duration_threshold

    sequences = []  # list of dicts with keys: indices, base_duration
    in_sequence = False
    seq_indices = []
    base_duration = None

    n = len(df)
    for i in range(n):
        row = df.loc[i]
        row_cat = row.get("category")
        row_dur = row["duration"]
        is_orn = (
            row_cat == "ornamentation"
            and row_dur < max_ornament_duration_threshold
        )
        is_same_duration_extra = (
            in_sequence
            and row_cat != "ornamentation"
            and (base_duration is not None and row_dur == base_duration)
        )

        if is_orn:
            # Starting a new sequence
            if not in_sequence:
                in_sequence = True
                seq_indices = [i]
                base_duration = df.loc[i, "duration"]
            else:
                # Continue only if duration matches base_duration
                if df.loc[i, "duration"] == base_duration:
                    seq_indices.append(i)
                else:
                    if seq_indices:
                        sequences.append(
                            {
                                "indices": seq_indices.copy(),
                                "base_duration": base_duration,
                            }
                        )
                    seq_indices = [i]
                    base_duration = df.loc[i, "duration"]
        elif is_same_duration_extra:
            # Include same-duration non-ornament notes as part of the active sequence
            seq_indices.append(i)
        else:
            if in_sequence:
                can_bridge = (
                    base_duration is not None
                    and row_dur == base_duration
                    and i + 1 < n
                    and is_ornament(i + 1)
                    and df.loc[i + 1, "duration"] == base_duration
                )
                if can_bridge:
                    # Treat this non-ornament same-duration note as part of the sequence
                    seq_indices.append(i)
                    continue  # Next loop iteration will handle following ornament(s)
                else:
                    # Close sequence here (without including this non-ornament)
                    if seq_indices:
                        sequences.append(
                            {
                                "indices": seq_indices.copy(),
                                "base_duration": base_duration,
                            }
                        )
                    in_sequence = False
                    seq_indices = []
                    base_duration = None
            # else: outside sequence, ignore non-ornament

    # Tail close
    if in_sequence and seq_indices:
        sequences.append(
            {
                "indices": seq_indices.copy(),
                "base_duration": base_duration,
            }
        )

    if not sequences:
        empty = df.head(0).copy()
        empty["sequence_id"] = pd.Series(dtype="Int64")
        empty["is_context"] = pd.Series(dtype="boolean")
        return empty

    output_rows = []
    for seq_id, seq in enumerate(sequences):
        idxs = sorted(set(seq["indices"]))
        if not idxs:
            continue
        base_duration = seq.get("base_duration")
        context_idxs = set()

        left_idx = idxs[0] - 1
        left_context_candidate = None
        while left_idx >= 0:
            row_left = df.loc[left_idx]
            row_left_dur = row_left.get("duration")
            can_absorb_left = (
                pd.notna(row_left_dur)
                and base_duration is not None
                and row_left_dur == base_duration
            )
            if can_absorb_left:
                idxs.insert(0, left_idx)
                left_idx -= 1
                continue
            left_context_candidate = left_idx
            break

        right_idx = idxs[-1] + 1
        while right_idx < n:
            row_right = df.loc[right_idx]
            row_right_dur = row_right.get("duration")
            can_absorb_right = (
                pd.notna(row_right_dur)
                and base_duration is not None
                and row_right_dur == base_duration
            )
            if can_absorb_right:
                idxs.append(right_idx)
                right_idx += 1
                continue
            break
        right_context_candidate = right_idx

        idxs = sorted(set(idxs))

        if "onset" in df.columns:
            onset_series = df.loc[idxs, "onset"]
            valid_onsets = onset_series[pd.notna(onset_series)]
            if not valid_onsets.empty:
                first_onset = valid_onsets.min()
                last_onset = valid_onsets.max()
            else:
                first_onset = None
                last_onset = None
        else:
            first_onset = None
            last_onset = None

        if left_context_candidate is not None:
            pre_idx = left_context_candidate
            while pre_idx is not None and pre_idx >= 0:
                if pre_idx in idxs:
                    pre_idx -= 1
                    continue
                row_pre = df.loc[pre_idx]
                row_pre_dur = row_pre.get("duration")
                can_absorb_pre = (
                    pd.notna(row_pre_dur)
                    and base_duration is not None
                    and row_pre_dur == base_duration
                )
                if can_absorb_pre:
                    idxs.insert(0, pre_idx)
                    if "onset" in df.columns:
                        pre_onset = row_pre.get("onset")
                        if pd.notna(pre_onset):
                            if first_onset is None or pre_onset < first_onset:
                                first_onset = pre_onset
                    pre_idx -= 1
                    continue
                added_context = False
                if "onset" in df.columns:
                    pre_onset = row_pre.get("onset")
                    if pd.notna(pre_onset):
                        if first_onset is None or pre_onset < first_onset:
                            same_onset_mask = df["onset"] == pre_onset
                            same_onset_indices = df.index[same_onset_mask]
                            for idx_candidate in same_onset_indices:
                                if idx_candidate not in idxs:
                                    context_idxs.add(idx_candidate)
                            added_context = True
                        else:
                            pre_idx -= 1
                            continue
                    else:
                        context_idxs.add(pre_idx)
                        added_context = True
                else:
                    context_idxs.add(pre_idx)
                    added_context = True

                if added_context:
                    break

                pre_idx -= 1

            idxs = sorted(set(idxs))

            if "onset" in df.columns:
                onset_series = df.loc[idxs, "onset"]
                valid_onsets = onset_series[pd.notna(onset_series)]
                if not valid_onsets.empty:
                    first_onset = valid_onsets.min()
                    last_onset = valid_onsets.max()
                else:
                    first_onset = None
                    last_onset = None
            else:
                first_onset = None
                last_onset = None

        if right_context_candidate is not None:
            post_idx = right_context_candidate
            while post_idx is not None and post_idx < n:
                if post_idx in idxs:
                    post_idx += 1
                    continue
                row_post = df.loc[post_idx]
                row_post_dur = row_post.get("duration")
                can_absorb_post = (
                    pd.notna(row_post_dur)
                    and base_duration is not None
                    and row_post_dur == base_duration
                )
                if can_absorb_post:
                    idxs.append(post_idx)
                    if "onset" in df.columns:
                        post_onset = row_post.get("onset")
                        if pd.notna(post_onset):
                            if last_onset is None or post_onset > last_onset:
                                last_onset = post_onset
                    post_idx += 1
                    continue
                added_context = False
                if "onset" in df.columns:
                    post_onset = row_post.get("onset")
                    if pd.notna(post_onset):
                        if last_onset is None or post_onset > last_onset:
                            same_onset_mask = df["onset"] == post_onset
                            same_onset_indices = df.index[same_onset_mask]
                            for idx_candidate in same_onset_indices:
                                if idx_candidate not in idxs:
                                    context_idxs.add(idx_candidate)
                            added_context = True
                        else:
                            post_idx += 1
                            continue
                    else:
                        context_idxs.add(post_idx)
                        added_context = True
                else:
                    context_idxs.add(post_idx)
                    added_context = True

                if added_context:
                    break

                post_idx += 1

        idxs = sorted(set(idxs))
        context_idxs.difference_update(idxs)

        full_indices = sorted(set(idxs) | context_idxs)
        seq_df = df.loc[full_indices].copy()
        seq_df["sequence_id"] = seq_id
        seq_df["is_context"] = seq_df.index.isin(context_idxs)
        output_rows.append(seq_df)

    combined = pd.concat(output_rows).reset_index(drop=True)
    sort_cols = ["sequence_id"]
    ascending_flags = [True]
    if "onset" in combined.columns:
        sort_cols.append("onset")
        ascending_flags.append(True)
    if "voice_tab" in combined.columns:
        sort_cols.append("voice_tab")
        ascending_flags.append(False)
    combined = combined.sort_values(
        sort_cols, ascending=ascending_flags, kind="stable"
    ).reset_index(drop=True)
    return combined
