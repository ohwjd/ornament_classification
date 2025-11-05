from fractions import Fraction
import pandas as pd


def find_ornament_sequences(
    df,
    max_ornament_duration_threshold=Fraction(1, 4),
    inclusive: bool = False,
    merge_single_bridge: bool = True,
    add_context: bool = True,
    allow_variable_durations: bool = False,
    voice_col: str = "voice",
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
            - After a sequence is delimited, optionally (add_context=True) include only the note
                immediately after its last element (if it exists) marked with is_context=True. The
                note before the sequence will NOT be included as context.

    Parameters
    ----------
    df : pd.DataFrame
        Voice-specific DataFrame (assumed already filtered per voice and preprocessed).
    max_ornament_duration_threshold : Fraction or numeric, default 1/4
        Upper bound for ornament note durations.
    inclusive : bool, default False
        If True, duration <= threshold counts; else strictly <.
    merge_single_bridge : bool, default True
        Merge ornament runs split by exactly one non-ornament of the same duration.
    add_context : bool, default True
        Whether to append the immediate pre/post notes of each sequence.

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
        if inclusive:
            return dur <= max_ornament_duration_threshold
        return dur < max_ornament_duration_threshold

    sequences = []  # list of dicts with keys: indices, context_indices
    in_sequence = False
    seq_indices = []
    base_duration = None

    n = len(df)
    for i in range(n):
        if is_ornament(i):
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
        else:
            if in_sequence:
                # Potential bridging logic
                can_bridge = False
                if merge_single_bridge and (
                    allow_variable_durations
                    or df.loc[i, "duration"] == base_duration
                ):
                    # Look ahead one note for an ornament continuing with same duration
                    if (
                        i + 1 < n
                        and is_ornament(i + 1)
                        and df.loc[i + 1, "duration"] == base_duration
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
                    row_c.get("category")
                    in {"adaptation", "repetition", "ficta"}
                    and row_c["duration"] == base_orn_dur
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

    # Stable sort: by sequence then onset if available, else by sequence then original index order
    sort_cols = ["sequence_id"]
    if "onset" in combined.columns:
        sort_cols.append("onset")
    combined = combined.sort_values(sort_cols, kind="stable").reset_index(
        drop=True
    )
    return combined


def filter_four_note_step_sequences(
    sequences_df: pd.DataFrame,
    original_df: pd.DataFrame = None,
    allowed_intervals=(1, 2),
    exact_ornament_note_count: int = 4,
    include_context: bool = True,
    ascending_or_descending_only: bool = False,
    include_same_duration_categories=("adaptation", "repetition", "ficta"),
    require_perfect_fifth_span: bool = False,
    use_all_chord_pitches: bool = True,
):
    """
    Filter ornament sequences to those whose core ornamentation notes consist of exactly
    `exact_ornament_note_count` notes moving only by allowed semitone steps (default: half or whole steps).

    Assumptions / Logic:
      - Core notes are those with is_context == False and:
          * category == 'ornamentation'
          * PLUS (if include_adaptation_same_duration) category == 'adaptation' AND duration == base ornament duration
      - A sequence qualifies if it has exactly `exact_ornament_note_count` core notes.
      - Intervals are computed between successive core notes using absolute pitch differences.
    - All successive intervals must be in `allowed_intervals`.
      - If ascending_or_descending_only is True, the core notes must be monotonic (all up or all down).
      - If require_perfect_fifth_span is True, only sequences spanning exactly a perfect fifth (7 semitones) are retained.
      - Regardless, this function now annotates each qualifying four-step sequence with a 'four_step_comment':
          * 'perfect_fifth_monotonic' (monotonic and span=7)
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
        Filtered DataFrame with same columns plus original sequence_id. Empty if none match.
    """
    if sequences_df.empty:
        return sequences_df.head(0).copy()

    qualifying_seq_ids = []
    seq_comments = {}
    # Track whether the chord-context (all context rows for the sequence) contains
    # any pitch a perfect fifth (7 semitones) from the first selected ornament pitch.
    seq_post_ctx_has_fifth = {}
    # Prepare fast lookup of all pitches per onset from the full texture
    onset_to_pitches = {}
    if (
        original_df is not None
        and use_all_chord_pitches
        and "onset" in original_df.columns
    ):
        for onset, g in original_df.groupby("onset"):
            if "pitch" in g.columns:
                onset_to_pitches[onset] = [
                    p for p in g["pitch"].tolist() if pd.notna(p)
                ]

    def exists_allowed_path(
        onset_list,
        base_duration_rows,
        require_monotonic=False,
        require_intervals=True,
    ):
        """
        Given ordered onsets, return one pitch-per-onset path complying with allowed_intervals, if any.
        If chord sets are unavailable for an onset, fall back to the sequence's own pitch at that onset.
        """
        # Build candidate sets per onset
        candidates = []
        for o in onset_list:
            if use_all_chord_pitches and onset_to_pitches.get(o):
                candidates.append(sorted(set(onset_to_pitches[o])))
            else:
                # Fallback: use pitches from sequence rows at that onset
                rows_here = base_duration_rows[base_duration_rows["onset"] == o]
                pitches_here = [
                    p for p in rows_here["pitch"].tolist() if pd.notna(p)
                ]
                if not pitches_here:
                    return None
                candidates.append(sorted(set(pitches_here)))

        # DFS over small depth (typically 4)
        path = []

        def dfs(i):
            if i == len(candidates):
                # If caller requested monotonicity, only accept paths that are
                # non-decreasing or non-increasing.
                if require_monotonic:
                    non_decreasing = all(
                        path[j + 1] >= path[j] for j in range(len(path) - 1)
                    )
                    non_increasing = all(
                        path[j + 1] <= path[j] for j in range(len(path) - 1)
                    )
                    return non_decreasing or non_increasing
                return True
            if i == 0:
                for p in candidates[0]:
                    path.append(p)
                    if dfs(1):
                        return True
                    path.pop()
                return False
            prev = path[-1]
            for p in candidates[i]:
                try:
                    step = abs(p - prev)
                except Exception:
                    continue
                # Only enforce allowed-intervals if requested; otherwise allow any step
                if (not require_intervals) or (step in allowed_intervals):
                    path.append(p)
                    if dfs(i + 1):
                        return True
                    path.pop()
            return False

        ok = dfs(0)
        return path.copy() if ok else None

    for seq_id, group in sequences_df.groupby("sequence_id"):
        # Determine base ornament duration (first ornamentation note among non-context rows)
        non_context = group[~group["is_context"]]
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
        # Order core by onset
        if "onset" in core.columns:
            core = core.sort_values("onset", kind="stable")
        onset_list = core["onset"].tolist()

        # Build base rows for fallback lookups
        base_rows_for_fallback = group if include_context else core

        # Try to find a valid pitch path across onsets considering all chord pitches
        # Try in order:
        # 1) monotonic path that respects allowed intervals
        # 2) any path that respects allowed intervals
        # 3) monotonic path ignoring allowed intervals (relaxed)
        # 4) any path ignoring allowed intervals
        selected_path = exists_allowed_path(
            onset_list,
            base_rows_for_fallback,
            require_monotonic=True,
            require_intervals=True,
        )
        if selected_path is None:
            selected_path = exists_allowed_path(
                onset_list,
                base_rows_for_fallback,
                require_monotonic=False,
                require_intervals=True,
            )
        if selected_path is None:
            selected_path = exists_allowed_path(
                onset_list,
                base_rows_for_fallback,
                require_monotonic=True,
                require_intervals=False,
            )
        if selected_path is None:
            selected_path = exists_allowed_path(
                onset_list,
                base_rows_for_fallback,
                require_monotonic=False,
                require_intervals=False,
            )

        # Use actual core ornament pitches (from sequence rows) to test monotonicity.
        core_pitches_from_rows = []
        if "pitch" in core.columns:
            core_pitches_from_rows = [
                p for p in core["pitch"].tolist() if pd.notna(p)
            ]

        if (
            core_pitches_from_rows
            and len(core_pitches_from_rows) == exact_ornament_note_count
        ):
            try:
                core_pitches_int = [int(p) for p in core_pitches_from_rows]
                non_decreasing = all(
                    core_pitches_int[i + 1] >= core_pitches_int[i]
                    for i in range(len(core_pitches_int) - 1)
                )
                non_increasing = all(
                    core_pitches_int[i + 1] <= core_pitches_int[i]
                    for i in range(len(core_pitches_int) - 1)
                )
                monotonic = non_decreasing or non_increasing
            except Exception:
                monotonic = False
        else:
            # Fallback to using selected_path if core pitches missing
            pitches = selected_path
            if pitches is None:
                monotonic = False
            else:
                non_decreasing = all(
                    pitches[i + 1] >= pitches[i]
                    for i in range(len(pitches) - 1)
                )
                non_increasing = all(
                    pitches[i + 1] <= pitches[i]
                    for i in range(len(pitches) - 1)
                )
                monotonic = non_decreasing or non_increasing

        if selected_path is None:
            continue

        pitches = selected_path
        non_decreasing = all(
            pitches[i + 1] >= pitches[i] for i in range(len(pitches) - 1)
        )
        non_increasing = all(
            pitches[i + 1] <= pitches[i] for i in range(len(pitches) - 1)
        )
        monotonic = non_decreasing or non_increasing

        # Compute span: from first selected pitch to any candidate at the post onset (if any), else last selected
        span_target_pitch = pitches[-1]
        last_core_onset = (
            core["onset"].max() if "onset" in core.columns else None
        )
        if last_core_onset is not None:
            # Prefer using chord set at the immediate next onset in the original texture
            next_onset_candidates = None
            if onset_to_pitches:
                next_onsets = sorted(
                    o for o in onset_to_pitches.keys() if o > last_core_onset
                )
                if next_onsets:
                    next_onset = next_onsets[0]
                    next_onset_candidates = onset_to_pitches.get(next_onset)
            if not next_onset_candidates:
                # Fallback to any post-context row inside the group
                post_context = group[
                    (group["is_context"]) & (group["onset"] > last_core_onset)
                ]
                if not post_context.empty and "pitch" in post_context.columns:
                    next_onset_candidates = [
                        p for p in post_context["pitch"].tolist() if pd.notna(p)
                    ]
            if next_onset_candidates:
                # Choose the candidate minimizing absolute span (we only need span classification)
                span_target_pitch = min(
                    next_onset_candidates, key=lambda p: abs(p - pitches[0])
                )

        span = abs(span_target_pitch - pitches[0])
        if ascending_or_descending_only and not monotonic:
            continue
        if require_perfect_fifth_span and span != 7:
            continue
        if monotonic and span == 7:
            comment = "perfect_fifth_monotonic"
        elif monotonic:
            comment = "four_step_monotonic_other_span"
        else:
            comment = "four_step_non_monotonic"
        seq_comments[seq_id] = comment
        # Collect post-context pitches from context rows that occur after the last core onset
        # (this represents the chord-context). If none are present, fall back to onset_to_pitches
        # for the immediate next onset.
        first_pitch = pitches[0]
        post_context_pitches = []
        if last_core_onset is not None and "is_context" in group.columns:
            post_context_pitches = [
                p
                for p in group[
                    (group["is_context"]) & (group["onset"] > last_core_onset)
                ]["pitch"].tolist()
                if pd.notna(p)
            ]
        # Fallback: if no explicit post-context rows, try onset_to_pitches at next onset
        if not post_context_pitches and onset_to_pitches:
            if last_core_onset is not None:
                next_onsets = sorted(
                    o for o in onset_to_pitches.keys() if o > last_core_onset
                )
                if next_onsets:
                    next_onset = next_onsets[0]
                    post_context_pitches = onset_to_pitches.get(next_onset, [])

        has_fifth = any(
            abs(int(p) - int(first_pitch)) == 7
            for p in post_context_pitches
            if pd.notna(p)
        )

        # Enforce monotonic 4-step core and require at least one post-context perfect fifth.
        if not monotonic:
            continue
        if not has_fifth:
            continue

        seq_post_ctx_has_fifth[seq_id] = bool(has_fifth)
        qualifying_seq_ids.append(seq_id)

    if not qualifying_seq_ids:
        return sequences_df.head(0).copy()

    filtered = sequences_df[
        sequences_df["sequence_id"].isin(qualifying_seq_ids)
    ].copy()
    # Attach comment per sequence_id
    filtered["four_step_comment"] = filtered["sequence_id"].map(seq_comments)
    # Attach the post-context perfect-fifth flag per sequence_id
    filtered["post_context_has_perfect_fifth"] = filtered["sequence_id"].map(
        seq_post_ctx_has_fifth
    )
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
