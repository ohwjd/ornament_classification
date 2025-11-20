import pandas as pd
from typing import List


def filter_four_note_step_sequences(
    sequences_df: pd.DataFrame,
    allowed_intervals=(1, 2),
):
    """
    Filter ornament sequences to those whose core ornamentation notes consist of
    four unique onsets moving only by allowed semitone steps (default: half or whole steps).

        Assumptions:
            - Core notes are determined solely from the ordering of non-context rows.
                The function requires exactly four unique non-context onsets.
                Additional simultaneous notes at the same onset (chords) are allowed but flagged in the output comment.
            - Intervals are computed between successive core notes using absolute pitch differences.
        - All successive intervals must be in `allowed_intervals`.
            - Perfect-fifth checks respect melodic direction: ascending runs accept a fifth up (+7) or fourth down (-5) from the opening pitch, descending runs accept a fifth down (-7) or fourth up (+5).
        - Each qualifying sequence receives a `four_step_comment` list rendered as a string, e.g.:
                * `[monotonic_perfect_fifth_0_1]` for directional perfect-fifth matches annotated with voice IDs
                * `[monotonic_perfect_fifth]` when a directional perfect fifth is present but voice identifiers are unavailable
                * `[monotonic]` for monotonic runs that do not have a perfect-fifth between first note and post-context notes
                * `[non_monotonic]` when the contour changes direction
                * Comments gain a `_chords` suffix when any core onset contains multiple notes.

    Parameters
    ----------
    sequences_df : pd.DataFrame
        Output of find_ornament_sequences (may contain multiple sequence_id values).
    allowed_intervals : iterable
        Set of permitted absolute semitone distances between successive core ornamentation notes.
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with same columns plus original sequence_id. The
        four_step_comment column stores list-formatted strings (e.g., "[monotonic_perfect_fifth_0_1]")
        describing monotonic behavior. Perfect-fifth matches inspect every first-onset
        note (to accommodate opening chords) against post-context notes and encode
        successful combinations using their `voice_tab` identifiers.
    """

    ORNAMENT_ONSET_COUNT = 4

    if sequences_df.empty:
        return sequences_df.head(0).copy()

    allowed_step_values = set()
    for interval in allowed_intervals:
        try:
            if pd.isna(interval):
                continue
        except TypeError:
            pass
        try:
            allowed_step_values.add(abs(float(interval)))
        except (TypeError, ValueError):
            continue

    if not allowed_step_values:
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

    def _normalize_pitch(val):
        if pd.isna(val):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _generate_stepwise_paths(pitch_options):
        paths: List[List[float]] = []

        def backtrack(index: int, current: List[float]):
            if index == len(pitch_options):
                paths.append(current.copy())
                return
            for pitch in pitch_options[index]:
                if (
                    not current
                    or abs(pitch - current[-1]) in allowed_step_values
                ):
                    current.append(pitch)
                    backtrack(index + 1, current)
                    current.pop()

        backtrack(0, [])
        return paths

    def _is_non_decreasing(values: List[float]) -> bool:
        return all(values[i + 1] >= values[i] for i in range(len(values) - 1))

    def _is_non_increasing(values: List[float]) -> bool:
        return all(values[i + 1] <= values[i] for i in range(len(values) - 1))

    def _is_strictly_increasing(values: List[float]) -> bool:
        return all(values[i + 1] > values[i] for i in range(len(values) - 1))

    def _is_strictly_decreasing(values: List[float]) -> bool:
        return all(values[i + 1] < values[i] for i in range(len(values) - 1))

    qualifying_seq_ids = []
    seq_comments = {}

    for seq_id, group in sequences_df.groupby("sequence_id"):
        non_context = group[~group["is_context"]]
        if non_context.empty:
            continue

        if "onset" not in non_context.columns:
            continue

        if non_context["onset"].isna().any():
            continue

        core_onsets = non_context["onset"]
        if core_onsets.nunique() != ORNAMENT_ONSET_COUNT:
            continue

        ordered = (
            non_context.sort_values("onset", kind="stable")
            if "onset" in non_context.columns
            else non_context.sort_index()
        )
        onset_counts = core_onsets.value_counts()
        has_chords = any(count > 1 for count in onset_counts)

        core = (
            ordered.drop_duplicates(subset=["onset"], keep="first")
            .sort_values("onset", kind="stable")
            .reset_index(drop=True)
        )
        if len(core) != ORNAMENT_ONSET_COUNT:
            continue

        pitch_options: List[List[float]] = []
        invalid_pitch_found = False
        for onset_value in core["onset"].tolist():
            onset_rows = ordered[ordered["onset"] == onset_value]
            normalized_pitches = [
                normalized
                for normalized in (
                    _normalize_pitch(val)
                    for val in onset_rows["pitch"].tolist()
                )
                if normalized is not None
            ]
            if not normalized_pitches:
                invalid_pitch_found = True
                break
            pitch_options.append(normalized_pitches)
        if invalid_pitch_found:
            continue

        stepwise_paths = _generate_stepwise_paths(pitch_options)
        if not stepwise_paths:
            continue

        strictly_increasing_paths = [
            path for path in stepwise_paths if _is_strictly_increasing(path)
        ]
        strictly_decreasing_paths = [
            path for path in stepwise_paths if _is_strictly_decreasing(path)
        ]
        non_decreasing_paths = [
            path for path in stepwise_paths if _is_non_decreasing(path)
        ]
        non_increasing_paths = [
            path for path in stepwise_paths if _is_non_increasing(path)
        ]

        if strictly_increasing_paths:
            pitches = strictly_increasing_paths[0]
        elif strictly_decreasing_paths:
            pitches = strictly_decreasing_paths[0]
        elif non_decreasing_paths:
            pitches = non_decreasing_paths[0]
        elif non_increasing_paths:
            pitches = non_increasing_paths[0]
        else:
            pitches = stepwise_paths[0]

        try:
            diffs = [
                abs(pitches[i + 1] - pitches[i])
                for i in range(len(pitches) - 1)
            ]
        except Exception:
            continue

        if len(pitches) != ORNAMENT_ONSET_COUNT:
            continue

        if not all(d in allowed_step_values for d in diffs):
            continue

        non_decreasing = _is_non_decreasing(pitches)
        non_increasing = _is_non_increasing(pitches)
        monotonic = non_decreasing or non_increasing

        strictly_increasing = _is_strictly_increasing(pitches)
        strictly_decreasing = _is_strictly_decreasing(pitches)

        if strictly_increasing:
            directional_allowed = {7, -5}
        elif strictly_decreasing:
            directional_allowed = {-7, 5}
        elif monotonic:
            directional_allowed = {7, -7, 5, -5}
        else:
            directional_allowed = {7, -7}

        span_target_pitch = pitches[-1]
        post_context_sorted = pd.DataFrame()
        last_core_onset = core["onset"].max()
        post_context_sorted = group[
            (group["is_context"]) & (group["onset"] > last_core_onset)
        ].sort_values("onset", kind="stable")
        if not post_context_sorted.empty:
            span_target_pitch = post_context_sorted.iloc[-1]["pitch"]

        if pd.isna(span_target_pitch):
            continue

        try:
            span_diff = span_target_pitch - pitches[0]
        except Exception:
            continue

        perfect_fifth_labels: List[str] = []
        if (
            monotonic
            and not post_context_sorted.empty
            and "pitch" in group.columns
        ):
            first_onset = core["onset"].iloc[0]
            first_rows = non_context[non_context["onset"] == first_onset]
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
                        interval = post_pitch - first_pitch
                    except TypeError:
                        continue
                    if interval in directional_allowed:
                        ctx_voice = (
                            ctx_row.get("voice_tab")
                            if "voice_tab" in ctx_row.index
                            else None
                        )
                        label = (
                            "monotonic_perfect_fifth_"
                            + _format_voice_label(first_voice)
                            + "_"
                            + _format_voice_label(ctx_voice)
                        )
                        perfect_fifth_labels.append(label)

        if monotonic:
            if perfect_fifth_labels:
                comments_for_sequence = perfect_fifth_labels
            elif span_diff in {7, -7}:
                comments_for_sequence = ["monotonic_perfect_fifth"]
            else:
                comments_for_sequence = ["monotonic"]
        else:
            comments_for_sequence = ["non_monotonic"]

        if has_chords:
            comments_for_sequence = [
                label + ("" if label.endswith("_chords") else "_chords")
                for label in comments_for_sequence
            ]

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


def filter_non_chord_sequences(
    sequences_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter ornament sequences to those whose core ornamentation notes do not form chords
    (i.e., only one note per onset in the core sequence).

        Definition:
            - Core notes are determined solely from the ordering of non-context rows.
            - If any two core notes share the same onset value, the sequence is excluded.

    Parameters
    ----------
    sequences_df : pd.DataFrame
        Output of find_ornament_sequences (may contain multiple sequence_id values).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with same columns plus original sequence_id.
    """
    if sequences_df.empty:
        return sequences_df.head(0).copy()

    qualifying_seq_ids = []

    for seq_id, group in sequences_df.groupby("sequence_id"):
        non_context = group[~group["is_context"]]
        if non_context.empty:
            continue

        if "onset" in non_context.columns:
            onset_counts = non_context["onset"].value_counts()
            if all(count == 1 for count in onset_counts):
                qualifying_seq_ids.append(seq_id)
        else:
            qualifying_seq_ids.append(seq_id)

    if not qualifying_seq_ids:
        return sequences_df.head(0).copy()

    filtered = sequences_df[
        sequences_df["sequence_id"].isin(qualifying_seq_ids)
    ].copy()

    # Preserve ordering
    sort_cols = ["sequence_id"]
    if "onset" in filtered.columns:
        sort_cols.append("onset")
    filtered = filtered.sort_values(sort_cols, kind="stable").reset_index(
        drop=True
    )

    return filtered


def only_starting_chord_and_then_non_chord_sequences(
    sequences_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter ornament sequences to those whose core ornamentation notes start with a chord
    (i.e., more than one note at the first onset in the core sequence) and are followed
    only by non-chord notes.

        Definition:
            - Core notes are determined solely from the ordering of non-context rows.
            - If the first onset in the core sequence contains more than one note, and
              all subsequent onsets contain only one note, the sequence is retained.

    Parameters
    ----------
    sequences_df : pd.DataFrame
        Output of find_ornament_sequences (may contain multiple sequence_id values).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with same columns plus original sequence_id.
    """
    if sequences_df.empty:
        return sequences_df.head(0).copy()

    qualifying_seq_ids = []

    for seq_id, group in sequences_df.groupby("sequence_id"):
        non_context = group[~group["is_context"]]
        if non_context.empty:
            continue

        if "onset" in non_context.columns:
            onset_counts = non_context["onset"].value_counts().sort_index()
            first_onset_count = onset_counts.iloc[0]
            subsequent_counts = onset_counts.iloc[1:]
            if first_onset_count > 1 and all(
                count == 1 for count in subsequent_counts
            ):
                qualifying_seq_ids.append(seq_id)

    if not qualifying_seq_ids:
        return sequences_df.head(0).copy()

    filtered = sequences_df[
        sequences_df["sequence_id"].isin(qualifying_seq_ids)
    ].copy()

    # Preserve ordering
    sort_cols = ["sequence_id"]
    if "onset" in filtered.columns:
        sort_cols.append("onset")
    filtered = filtered.sort_values(sort_cols, kind="stable").reset_index(
        drop=True
    )

    return filtered
