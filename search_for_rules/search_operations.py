from fractions import Fraction
import pandas as pd


def find_ornament_sequences_abtab(
    df,
    max_ornament_duration_threshold=Fraction(1, 4),
    merge_single_bridge: bool = True,
    add_context: bool = True,
    allow_variable_durations: bool = False,
    voice_col: str = "voice",
    same_duration_categories=("adaptation", "repetition", "ficta"),
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
            - After a sequence is delimited, optionally (add_context=True) include the note
                immediately after its last element (if it exists) and the note immediately before
                the sequence (only when that earlier note ends exactly at the first sequence onset)
                marked with is_context=True. Any earlier notes with a gap are ignored.
            - Notes whose category is one of `same_duration_categories` and whose duration matches
                the base ornament duration are treated as regular sequence members (not context).

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
    same_duration_categories : iterable or None, default ("adaptation", "repetition", "ficta")
        Additional categories that, when matching the base ornament duration, are treated as
        sequence members rather than context. Pass None to allow any non-ornament category.

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

    sequences = []  # list of dicts with keys: indices, context_indices
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
                        sequences.append({"indices": seq_indices})
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
                        sequences.append({"indices": seq_indices})
                    in_sequence = False
                    seq_indices = []
                    base_duration = None
            # else: outside sequence, ignore non-ornament

    # Tail close
    if in_sequence and seq_indices:
        sequences.append({"indices": seq_indices})

    if not sequences:
        empty = df.head(0).copy()
        empty["sequence_id"] = pd.Series(dtype="Int64")
        empty["is_context"] = pd.Series(dtype="boolean")
        return empty

    output_rows = []
    for seq_id, seq in enumerate(sequences):
        idxs = seq["indices"]
        context_idxs = set()
        # Only include the immediate post-sequence note as context (if requested).
        if add_context:
            pre = idxs[0] - 1
            if pre >= 0:
                if {"onset", "duration"}.issubset(df.columns):
                    pre_row = df.loc[pre]
                    pre_onset = pre_row.get("onset")
                    pre_duration = pre_row.get("duration")
                    first_onset = df.loc[idxs[0], "onset"]
                    if (
                        pd.notna(pre_onset)
                        and pd.notna(pre_duration)
                        and pd.notna(first_onset)
                        and pre_onset + pre_duration == first_onset
                    ):
                        context_idxs.add(pre)
                else:
                    context_idxs.add(pre)
            post = idxs[-1] + 1
            if post < n:
                context_idxs.add(post)
        # Reclassify adaptation context notes that have same duration as ornaments into sequence
        if context_idxs:
            # Determine base ornament duration (first ornamentation note inside idxs)
            ornament_durations = [
                df.loc[i, "duration"]
                for i in idxs
                if df.loc[i, "category"] == "ornamentation"
            ]
            base_orn_dur = ornament_durations[0] if ornament_durations else None
            # Work on a copy since we may modify context_idxs during iteration
            for c_idx in list(context_idxs):
                if base_orn_dur is None:
                    break
                row_c = df.loc[c_idx]
                if (
                    row_c.get("category") != "ornamentation"
                    and (
                        allowed_same_duration_categories is None
                        or row_c.get("category")
                        in allowed_same_duration_categories
                    )
                    and (
                        allow_variable_durations
                        or row_c["duration"] == base_orn_dur
                    )
                ):
                    # Move this note from context to sequence
                    context_idxs.remove(c_idx)
                    if c_idx not in idxs:
                        # Insert at beginning or end based on position
                        if c_idx < min(idxs):
                            idxs = [c_idx] + idxs
                        else:
                            idxs = idxs + [c_idx]
                    # Add replacement context on the outside if possible
                    # Determine if it was originally pre or post relative to original bounds
                    orig_min = min(idxs)
                    orig_max = max(idxs)
                    # If c_idx was pre we try one more to the left
                    if c_idx == orig_min:
                        candidate = c_idx - 1
                        if (
                            candidate >= 0
                            and candidate not in idxs
                            and candidate not in context_idxs
                        ):
                            context_idxs.add(candidate)
                    # If c_idx was post we try one more to the right
                    if c_idx == orig_max:
                        candidate = c_idx + 1
                        if (
                            candidate < n
                            and candidate not in idxs
                            and candidate not in context_idxs
                        ):
                            context_idxs.add(candidate)

        full_indices = sorted(set(idxs) | context_idxs)
        seq_df = df.loc[full_indices].copy()
        seq_df["sequence_id"] = seq_id
        seq_df["is_context"] = seq_df.index.isin(context_idxs)
        output_rows.append(seq_df)

    combined = pd.concat(output_rows).reset_index(drop=True)
    sort_cols = ["sequence_id"]
    if "onset" in combined.columns:
        sort_cols.append("onset")
    combined = combined.sort_values(sort_cols, kind="stable").reset_index(
        drop=True
    )
    return combined


def filter_four_note_step_sequences(
    sequences_df: pd.DataFrame,
    allowed_intervals=(1, 2),
    exact_ornament_note_count: int = 4,
    include_context: bool = True,
    ascending_or_descending_only: bool = False,
    include_same_duration_categories=("adaptation", "repetition", "ficta"),
    require_perfect_fifth_span: bool = False,
):
    """
    Filter ornament sequences to those whose core ornamentation notes consist of exactly
    `exact_ornament_note_count` notes moving only by allowed semitone steps (default: half or whole steps).

    Assumptions:
      - Core notes are those with is_context == False and:
          * category == 'ornamentation'
          * PLUS (if include_adaptation_same_duration) category == 'adaptation' AND duration == base ornament duration
      - A sequence qualifies if it has exactly `exact_ornament_note_count` core notes.
      - Intervals are computed between successive core notes using absolute pitch differences.
    - All successive intervals must be in `allowed_intervals`.
      - If ascending_or_descending_only is True, the core notes must be monotonic (all up or all down).
      - If require_perfect_fifth_span is True, only sequences spanning exactly a perfect fifth (7 semitones) are retained.
      - Regardless, this function now annotates each qualifying four-step sequence with a 'four_step_comment':
          * 'perfect_fifth_monotonic' (monotonic span where a post-context note supplies the perfect fifth)
          * 'four_step_monotonic_other_span' (monotonic but span != 7)
          * 'four_step_non_monotonic' (direction changes)
      - If include_context is True, return the entire sequence rows (including context notes and any non-ornament bridging notes); otherwise only the core ornamentation rows.

    Parameters
    ----------
    sequences_df : pd.DataFrame
        Output of find_ornament_sequences (may contain multiple sequence_id values).
    allowed_intervals : iterable
        Set of permitted absolute semitone distances between successive core ornamentation notes.
    exact_ornament_note_count : int
        Require exactly this number of ornamentation (non-context) notes.
    include_context : bool
        Whether to include context notes in the returned rows.
    ascending_or_descending_only : bool
        If True, reject sequences whose core ornamentation contour changes direction.
    include_same_duration_categories : tuple, default ("adaptation","repetition","ficta")
        Additional categories to count as core if same duration as base ornament notes.
    require_perfect_fifth_span : bool, default False
        If True, only keep sequences whose overall span is exactly 7 semitones; otherwise include all four-step sequences and classify them.

    Returns
    -------
    pd.DataFrame
    Filtered DataFrame with same columns plus original sequence_id. The
    four_step_comment value `perfect_fifth_monotonic` is only assigned when a post-context
    note provides the perfect-fifth span when measured against the first core note; otherwise
    monotonic sequences receive the
        `four_step_monotonic_other_span` label. Empty if none match.
    """
    if sequences_df.empty:
        return sequences_df.head(0).copy()

    qualifying_seq_ids = []
    seq_comments = {}

    for seq_id, group in sequences_df.groupby("sequence_id"):
        # Determine base ornament duration (first ornamentation note among non-context rows)
        non_context = group[~group["is_context"]]
        if len(non_context) != exact_ornament_note_count:
            continue
        ornament_rows = non_context[non_context["category"] == "ornamentation"]
        if ornament_rows.empty:
            continue
        base_duration = ornament_rows.iloc[0]["duration"]
        # Build core: ornamentation plus same-duration selected extra categories
        core = non_context[(non_context["category"] == "ornamentation")]
        if include_same_duration_categories:
            extra = non_context[
                (non_context["category"].isin(include_same_duration_categories))
                & (non_context["duration"] == base_duration)
            ]
            if not extra.empty:
                core = (
                    pd.concat([core, extra], axis=0)
                    .drop_duplicates()
                    .sort_values("onset", kind="stable")
                )
        if len(core) != exact_ornament_note_count:
            continue
        # Order core by onset if present to compute melodic intervals
        if "onset" in core.columns:
            core = core.sort_values("onset", kind="stable")
        pitches = core["pitch"].tolist()
        # Skip if any pitch is NaN or non-numeric
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
        # Compute span: between first core note and the (post) context note if it exists, else last core
        span_target_pitch = pitches[-1]
        post_context_sorted = pd.DataFrame()
        if "onset" in group.columns:
            # Identify post-context note(s): context notes with onset greater than last core onset
            last_core_onset = (
                core["onset"].max() if "onset" in core.columns else None
            )
            if last_core_onset is not None:
                post_context_sorted = group[
                    (group["is_context"]) & (group["onset"] > last_core_onset)
                ].sort_values("onset", kind="stable")
                if not post_context_sorted.empty:
                    span_target_pitch = post_context_sorted.iloc[-1]["pitch"]
        else:
            last_core_idx = core.index.max()
            post_context_sorted = group[
                (group["is_context"]) & (group.index > last_core_idx)
            ].sort_index()
            if not post_context_sorted.empty:
                span_target_pitch = post_context_sorted.iloc[-1]["pitch"]
        span = abs(span_target_pitch - pitches[0])
        if ascending_or_descending_only and not monotonic:
            continue
        if require_perfect_fifth_span and span != 7:
            continue
        perfect_fifth_from_context = False
        if (
            monotonic
            and not post_context_sorted.empty
            and "pitch" in group.columns
        ):
            first_core_pitch = pitches[0]
            last_core_pitch = pitches[-1]
            if pd.notna(first_core_pitch) and pd.notna(last_core_pitch):
                for _, ctx_row in post_context_sorted.iterrows():
                    post_pitch = ctx_row.get("pitch")
                    if pd.isna(post_pitch):
                        continue
                    try:
                        interval = abs(post_pitch - first_core_pitch)
                        relation_to_tail = abs(post_pitch - last_core_pitch)
                    except TypeError:
                        continue
                    if interval == 7 and relation_to_tail != 0:
                        perfect_fifth_from_context = True
                        break
        if monotonic and perfect_fifth_from_context and span == 7:
            comment = "perfect_fifth_monotonic"
        elif monotonic:
            comment = "four_step_monotonic_other_span"
        else:
            comment = "four_step_non_monotonic"
        seq_comments[seq_id] = comment
        qualifying_seq_ids.append(seq_id)

    if not qualifying_seq_ids:
        return sequences_df.head(0).copy()

    filtered = sequences_df[
        sequences_df["sequence_id"].isin(qualifying_seq_ids)
    ].copy()
    # Attach comment per sequence_id
    filtered["four_step_comment"] = filtered["sequence_id"].map(seq_comments)
    if not include_context:
        filtered = filtered[
            (~filtered["is_context"])
            & (filtered["category"] == "ornamentation")
        ]

    # Preserve ordering
    sort_cols = ["sequence_id"]
    if "onset" in filtered.columns:
        sort_cols.append("onset")
    filtered = filtered.sort_values(sort_cols, kind="stable").reset_index(
        drop=True
    )

    return filtered
