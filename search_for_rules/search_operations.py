from fractions import Fraction
from typing import Any, Dict, List, Sequence

import pandas as pd


def find_ornament_sequences_raw(
    df: pd.DataFrame,
    onset_col: str = "onset",
    duration_col: str = "duration",
    min_sequence_length: int = 4,
    add_context: bool = True,
) -> pd.DataFrame:
    """
    Identify ornament sequences purely from onset ordering and duration uniformity.

    A sequence is defined as at least `min_sequence_length` consecutive onsets (after
    sorting by onset) that all share the same duration value. The first onset in a
    qualifying span may contain multiple notes (a chord); every subsequent onset in
    the span must be represented by exactly one note row. Durations are required to be
    homogeneous within each onset (all notes at that onset share the same duration).

        Context handling:
            - If `add_context` is True, the full chord (all rows) at the immediate onset
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
    add_context : bool
        Whether to append immediate pre/post onset chords as context.

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
        if not info["homogeneous"] or pd.isna(info["duration"]):
            i += 1
            continue

        run: List[Dict[str, Any]] = [info]
        j = i + 1
        while j < num_groups:
            next_info = group_infos[j]
            if (
                not next_info["homogeneous"]
                or pd.isna(next_info["duration"])
                or next_info["duration"] != info["duration"]
            ):
                break
            if next_info["size"] != 1:
                break
            run.append(next_info)
            j += 1

        if len(run) >= min_sequence_length:
            sequences.append(
                {
                    "groups": run,
                    "start_group_idx": i,
                    "end_group_idx": j - 1,
                    "duration": info["duration"],
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

        context_indices: Sequence[int] = ()
        if add_context:
            post_idx = seq["end_group_idx"] + 1
            if post_idx < num_groups:
                post_indices = set(group_infos[post_idx]["indices"])
                core_index_set = set(core_indices)
                context_indices = tuple(
                    sorted(
                        idx for idx in post_indices if idx not in core_index_set
                    )
                )
        selected_indices = sorted(set(core_indices) | set(context_indices))

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
    merge_single_bridge: bool = True,
    add_context: bool = True,
    allow_variable_durations: bool = False,
    voice_col: str = "voice",
    same_duration_categories=None,
):
    """
    Identify ornament sequences and return a single DataFrame containing only the
    notes that belong to these sequences. Each note gains:
      - 'sequence_id' (zero-based)
      - 'is_context' (bool) True for added context notes (before/after), False otherwise.

        Definition of a sequence:
            - One or more consecutive notes where category == 'ornamentation' AND
                duration < (or <= if inclusive=True) max_ornament_duration_threshold.
            - All ornament notes in a sequence must share the same duration value.
            - If merge_single_bridge is True: two ornament runs of the same duration separated
                by exactly one non-ornament note of that SAME duration are merged into a single sequence;
                that bridging note is treated as part of the sequence (is_context = False).
            - After a sequence is delimited, optionally (add_context=True) include the entire onset
                immediately following its last element as context (is_context=True).
            - Any adjacent notes sharing the base ornament duration are absorbed into the sequence
                (regardless of category unless `same_duration_categories` explicitly restricts them).

    Parameters
    ----------
    df : pd.DataFrame
        Voice-specific DataFrame (assumed already filtered per voice and preprocessed).
    max_ornament_duration_threshold : Fraction or numeric, default 1/4
        Upper bound for ornament note durations.
    merge_single_bridge : bool, default True
        Merge ornament runs split by exactly one non-ornament of the same duration.
    add_context : bool, default True
        Whether to append the immediate post notes of each sequence.
    allow_variable_durations : bool, default False
        Allow ornament sequences to contain notes with varying durations.
    same_duration_categories : iterable or None, default None
        Optional whitelist of non-ornament categories that may be merged into the sequence.
        When None, any category is eligible if it satisfies the duration rules.

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

    if same_duration_categories is None:
        allowed_same_duration_categories = None
    else:
        allowed_same_duration_categories = set(same_duration_categories)

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
        category_allowed = (
            allowed_same_duration_categories is None
            or row_cat in allowed_same_duration_categories
        )
        is_same_duration_extra = (
            in_sequence
            and row_cat != "ornamentation"
            and category_allowed
            and (
                allow_variable_durations
                or (base_duration is not None and row_dur == base_duration)
            )
        )

        if is_orn:
            # Starting a new sequence
            if not in_sequence:
                in_sequence = True
                seq_indices = [i]
                base_duration = df.loc[i, "duration"]
            else:
                # Continue only if duration matches base_duration, unless variable durations allowed
                if (
                    allow_variable_durations
                    or df.loc[i, "duration"] == base_duration
                ):
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
                # Potential bridging logic
                can_bridge = False
                if merge_single_bridge and (
                    allow_variable_durations
                    or (base_duration is not None and row_dur == base_duration)
                ):
                    # Look ahead one note for an ornament continuing with same duration
                    if (
                        i + 1 < n
                        and is_ornament(i + 1)
                        and (
                            allow_variable_durations
                            or df.loc[i + 1, "duration"] == base_duration
                        )
                    ):
                        can_bridge = True
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

        # Phase 1: expand the sequence boundaries to absorb adjacent same-duration notes
        def _category_allowed(row_cat: str) -> bool:
            return (
                row_cat == "ornamentation"
                or allowed_same_duration_categories is None
                or row_cat in allowed_same_duration_categories
            )

        left_idx = idxs[0] - 1
        while left_idx >= 0:
            row_left = df.loc[left_idx]
            row_left_cat = row_left.get("category")
            row_left_dur = row_left.get("duration")
            if allow_variable_durations:
                can_absorb_left = _category_allowed(row_left_cat) and pd.notna(
                    row_left_dur
                )
            else:
                can_absorb_left = (
                    _category_allowed(row_left_cat)
                    and pd.notna(row_left_dur)
                    and base_duration is not None
                    and row_left_dur == base_duration
                )
            if can_absorb_left:
                idxs.insert(0, left_idx)
                left_idx -= 1
                continue
            break

        right_idx = idxs[-1] + 1
        while right_idx < n:
            row_right = df.loc[right_idx]
            row_right_cat = row_right.get("category")
            row_right_dur = row_right.get("duration")
            if allow_variable_durations:
                can_absorb_right = _category_allowed(
                    row_right_cat
                ) and pd.notna(row_right_dur)
            else:
                can_absorb_right = (
                    _category_allowed(row_right_cat)
                    and pd.notna(row_right_dur)
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
            last_onset = valid_onsets.max() if not valid_onsets.empty else None
        else:
            last_onset = None

        if add_context and right_context_candidate is not None:
            post_idx = right_context_candidate
            while post_idx is not None and post_idx < n:
                if post_idx in idxs:
                    post_idx += 1
                    continue
                row_post = df.loc[post_idx]
                row_post_cat = row_post.get("category")
                row_post_dur = row_post.get("duration")
                if allow_variable_durations:
                    can_absorb_post = _category_allowed(
                        row_post_cat
                    ) and pd.notna(row_post_dur)
                else:
                    can_absorb_post = (
                        _category_allowed(row_post_cat)
                        and pd.notna(row_post_dur)
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


def filter_four_note_step_sequences(
    sequences_df: pd.DataFrame,
    allowed_intervals=(1, 2),
    exact_ornament_note_count: int = 4,
    ascending_or_descending_only: bool = False,
    require_perfect_fifth_span: bool = False,
):
    """
    Filter ornament sequences to those whose core ornamentation notes consist of exactly
    `exact_ornament_note_count` notes moving only by allowed semitone steps (default: half or whole steps).

        Assumptions:
            - Core notes are determined solely from the ordering of non-context rows.
                The function inspects contiguous blocks of
                `exact_ornament_note_count` non-context notes (ignoring the `category`
                column entirely).
            - Intervals are computed between successive core notes using absolute pitch differences.
        - All successive intervals must be in `allowed_intervals`.
            - If ascending_or_descending_only is True, the core notes must be monotonic (all up or all down).
            - If require_perfect_fifth_span is True, only sequences spanning exactly a perfect fifth (7 semitones) or exhibiting a perfect-fifth context match are retained.
            - Each qualifying four-step sequence receives a `four_step_comment` list rendered as a string, e.g.:
                    * `[monotonic_0_1]` for monotonic runs where a context note completes a perfect fifth with a first-onset note (voice_tab identifiers shown)
                    * `[perfect_fifth_monotonic]` when a perfect fifth is achieved but voice identifiers are unavailable
                    * `[four_step_monotonic_other_span]` for monotonic runs without a perfect fifth
                    * `[four_step_non_monotonic]` when the contour changes direction

    Parameters
    ----------
    sequences_df : pd.DataFrame
        Output of find_ornament_sequences (may contain multiple sequence_id values).
    allowed_intervals : iterable
        Set of permitted absolute semitone distances between successive core ornamentation notes.
    exact_ornament_note_count : int
        Require exactly this number of ornamentation (non-context) notes.
    ascending_or_descending_only : bool
        If True, reject sequences whose core ornamentation contour changes direction.
    require_perfect_fifth_span : bool, default False
        If True, only keep sequences whose overall span is exactly 7 semitones; otherwise include all four-step sequences and classify them.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with same columns plus original sequence_id. The
        four_step_comment column stores list-formatted strings (e.g., "[monotonic_0_1]")
        describing monotonic behavior. Perfect-fifth matches inspect every first-onset
        note (to accommodate opening chords) against post-context notes and encode
        successful combinations using their `voice_tab` identifiers.
    """
    if sequences_df.empty:
        return sequences_df.head(0).copy()

    def _format_voice_label(val):
        if pd.isna(val):
            return "unknown"
        try:
            as_int = int(val)
            if float(as_int) == float(val):
                return str(as_int)
        except (ValueError, TypeError):
            pass
        return str(val)

    def _format_comment_output(labels):
        unique_labels: List[str] = []
        for label in labels:
            if label not in unique_labels:
                unique_labels.append(label)
        return "[" + ", ".join(unique_labels) + "]"

    qualifying_seq_ids = []
    seq_comments = {}

    for seq_id, group in sequences_df.groupby("sequence_id"):
        non_context = group[~group["is_context"]]
        if non_context.empty:
            continue

        ordered = (
            non_context.sort_values("onset", kind="stable")
            if "onset" in non_context.columns
            else non_context.sort_index()
        )

        max_start = len(ordered) - exact_ornament_note_count
        if max_start < 0:
            continue

        comments_for_sequence: List[str] = []
        for start in range(max_start + 1):
            core = ordered.iloc[start : start + exact_ornament_note_count]
            pitches = core["pitch"].tolist()
            if any(pd.isna(p) for p in pitches):
                continue
            try:
                diffs = [
                    abs(pitches[i + 1] - pitches[i])
                    for i in range(len(pitches) - 1)
                ]
            except Exception:
                continue
            if not all(d in allowed_intervals for d in diffs):
                continue

            non_decreasing = all(
                pitches[i + 1] >= pitches[i] for i in range(len(pitches) - 1)
            )
            non_increasing = all(
                pitches[i + 1] <= pitches[i] for i in range(len(pitches) - 1)
            )
            monotonic = non_decreasing or non_increasing
            if ascending_or_descending_only and not monotonic:
                continue

            span_target_pitch = pitches[-1]
            post_context_sorted = pd.DataFrame()
            if "onset" in group.columns and "onset" in core.columns:
                last_core_onset_series = core["onset"].dropna()
                if not last_core_onset_series.empty:
                    last_core_onset = last_core_onset_series.max()
                    post_context_sorted = group[
                        (group["is_context"])
                        & (group["onset"] > last_core_onset)
                    ].sort_values("onset", kind="stable")
                    if not post_context_sorted.empty:
                        span_target_pitch = post_context_sorted.iloc[-1][
                            "pitch"
                        ]
            else:
                last_core_idx = core.index.max()
                post_context_sorted = group[
                    (group["is_context"]) & (group.index > last_core_idx)
                ].sort_index()
                if not post_context_sorted.empty:
                    span_target_pitch = post_context_sorted.iloc[-1]["pitch"]

            if pd.isna(span_target_pitch):
                continue

            try:
                span = abs(span_target_pitch - pitches[0])
            except Exception:
                continue

            perfect_fifth_labels: List[str] = []
            if (
                monotonic
                and not post_context_sorted.empty
                and "pitch" in group.columns
            ):
                if "onset" in core.columns:
                    first_onset_series = core["onset"].dropna()
                    if not first_onset_series.empty:
                        first_onset = first_onset_series.min()
                        first_rows = core[core["onset"] == first_onset]
                    else:
                        first_rows = core.iloc[[0]]
                else:
                    first_rows = core.iloc[[0]]

                if first_rows.empty:
                    first_rows = core.iloc[[0]]

                for _, first_row in first_rows.iterrows():
                    first_pitch = first_row.get("pitch")
                    if pd.isna(first_pitch):
                        continue
                    first_voice = (
                        first_row.get("voice_tab")
                        if "voice_tab" in first_row.index
                        else None
                    )
                    for _, ctx_row in post_context_sorted.iterrows():
                        post_pitch = ctx_row.get("pitch")
                        if pd.isna(post_pitch):
                            continue
                        try:
                            interval = abs(post_pitch - first_pitch)
                        except TypeError:
                            continue
                        if interval == 7:
                            ctx_voice = (
                                ctx_row.get("voice_tab")
                                if "voice_tab" in ctx_row.index
                                else None
                            )
                            label = (
                                "monotonic_"
                                + _format_voice_label(first_voice)
                                + "_"
                                + _format_voice_label(ctx_voice)
                            )
                            perfect_fifth_labels.append(label)

            if require_perfect_fifth_span and not perfect_fifth_labels:
                if span != 7:
                    continue

            if monotonic:
                if perfect_fifth_labels:
                    comments_for_sequence = perfect_fifth_labels
                elif span == 7:
                    comments_for_sequence = ["perfect_fifth_monotonic"]
                else:
                    comments_for_sequence = ["four_step_monotonic_other_span"]
            else:
                comments_for_sequence = ["four_step_non_monotonic"]

            if comments_for_sequence:
                break

        if not comments_for_sequence:
            continue

        seq_comments[seq_id] = _format_comment_output(comments_for_sequence)
        qualifying_seq_ids.append(seq_id)

    if not qualifying_seq_ids:
        return sequences_df.head(0).copy()

    filtered = sequences_df[
        sequences_df["sequence_id"].isin(qualifying_seq_ids)
    ].copy()
    # Attach comment per sequence_id
    filtered["four_step_comment"] = filtered["sequence_id"].map(seq_comments)

    # Preserve ordering
    sort_cols = ["sequence_id"]
    if "onset" in filtered.columns:
        sort_cols.append("onset")
    filtered = filtered.sort_values(sort_cols, kind="stable").reset_index(
        drop=True
    )

    return filtered
