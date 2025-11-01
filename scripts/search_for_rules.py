# import music21 as m21

import pandas as pd
import os

from fractions import Fraction


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
    df = squish_to_one_staff(df, new_col="voice_squished")
    return df


def expand_and_split_voices(df):
    """
    Some notes are marked as belonging to multiple voices (e.g., "0 and 1").
    For rows where 'voice' contains 'and', split and duplicate the row for each voice.
    """
    expanded_rows = []
    for _, row in df.iterrows():
        voices = str(row["voice"]).split(" and ")
        for v in voices:
            new_row = row.copy()
            new_row["voice"] = v.strip()
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows).reset_index(drop=True)


def squish_to_one_staff(
    df: pd.DataFrame,
    voice_col: str = "voice",
    onset_col: str = "onset",
    new_col: str = "voice_order",
):
    """Add a per-onset vertical ordering column across all voices before splitting.

    Logic:
      - Single note at an onset -> 0
      - Multiple notes: rank unique voices numerically ascending (voice '0' highest) -> 0..N-1
      - Multiple rows sharing the same voice at that onset (if any) all get the same rank.
    """
    if voice_col not in df.columns or onset_col not in df.columns:
        return df

    def _voice_to_int(v):
        try:
            return int(v)
        except Exception:
            return 10_000

    voice_int_cache = df[voice_col].map(_voice_to_int)
    df[new_col] = pd.NA
    for _, idxs in df.groupby(onset_col).groups.items():
        if len(idxs) == 1:
            df.loc[idxs, new_col] = 0
            continue
        voices_here = voice_int_cache.loc[idxs]
        sorted_unique_voices = sorted(voices_here.unique())
        rank_map = {v: i for i, v in enumerate(sorted_unique_voices)}
        df.loc[idxs, new_col] = voices_here.map(rank_map).astype("Int64")
    # Provide an alias name if user expects a different label (voice_squished)
    if "voice_squished" not in df.columns:
        df["voice_squished"] = df[new_col]
    return df


def split_df_by_voice(df, voice_col: str = "voice_squished"):
    """
    Returns a dictionary of DataFrames, one for each unique voice bucket.
    Uses the squished voice column by default (vertical order per onset),
    so sequences are traced along the compressed staff rather than original voice labels.
    """
    if voice_col not in df.columns:
        # Fallback to original "voice" if squished not present
        voice_col = "voice"
    return {
        voice: df[df[voice_col] == voice].copy().reset_index(drop=True)
        for voice in df[voice_col].dropna().unique()
    }


def find_ornament_sequences(
    df,
    max_ornament_duration_threshold=Fraction(1, 4),
    inclusive: bool = False,
    merge_single_bridge: bool = True,
    add_context: bool = True,
    allow_variable_durations: bool = False,
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
        immediately before its first element and immediately after its last element (if they exist)
        marked with is_context=True.

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
        if add_context:
            pre = idxs[0] - 1
            post = idxs[-1] + 1
            if pre >= 0:
                context_idxs.add(pre)
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

    def exists_allowed_path(onset_list, base_duration_rows):
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
                if step in allowed_intervals:
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
        selected_path = exists_allowed_path(onset_list, base_rows_for_fallback)
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


def add_chord_context_to_sequences(
    sequences_no_context: pd.DataFrame, original_df: pd.DataFrame
):
    """Given sequences WITHOUT context (is_context column present but all False or absent),
    add full chords (all notes at previous distinct onset and next distinct onset) as context.

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
        start_onset = group["onset"].min()
        end_onset = group["onset"].max()
        # Locate previous and next onset values
        prev_onset_candidates = [o for o in all_onsets if o < start_onset]
        next_onset_candidates = [o for o in all_onsets if o > end_onset]
        prev_onset = (
            prev_onset_candidates[-1] if prev_onset_candidates else None
        )
        next_onset = next_onset_candidates[0] if next_onset_candidates else None

        context_rows = []
        if prev_onset is not None:
            prev_rows = original_sorted[original_sorted["onset"] == prev_onset]
            if not prev_rows.empty:
                prev_rows_ctx = prev_rows.copy()
                prev_rows_ctx["sequence_id"] = seq_id
                prev_rows_ctx["is_context"] = True
                context_rows.append(prev_rows_ctx)
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


def get_matching_file_path(
    file_name,
    files_list,
    first_replace_str="-mapping",
    second_replace_str="-score",
    folder_path="",
):
    # Remove "-score" and extension for matching
    base_name = os.path.splitext(file_name.replace(first_replace_str, ""))[0]
    for file_name in files_list:
        base = os.path.splitext(file_name.replace(second_replace_str, ""))[0]
        try:
            if base == base_name:
                return os.path.join(folder_path, file_name)
        except Exception as e:
            print("Error while matching:", e)
            return None


def extract_tbp_basic_metadata(path):
    """Read the beginning of a .tbp file and return (tuning, meter_info, diminution).

    Assumes header lines look like {TUNING:G}, {METER_INFO:2/2 (1-56)}, {DIMINUTION:2}.
    Stops scanning once musical content appears (line starting with '|', or a dot-containing syllable line),
    keeping implementation intentionally minimal.
    """
    tuning = None
    meter_info = None
    diminution = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                # Heuristic: music/content begins
                if (
                    line.startswith("|")
                    or line.startswith(".")
                    or line.startswith("br.")
                ):
                    break
                if not (line.startswith("{") and line.endswith("}")):
                    continue
                inner = line[1:-1]
                if ":" not in inner:
                    continue
                key, val = inner.split(":", 1)
                key = key.strip().upper()
                val = val.strip()
                if key == "TUNING":
                    tuning = val
                elif key in ("METER_INFO", "METER"):
                    meter_info = val
                elif key == "DIMINUTION":
                    # Try int, else keep raw
                    try:
                        diminution = int(val)
                    except ValueError:
                        diminution = val
                # Early exit if we've got all three
                if tuning and meter_info and (diminution is not None):
                    # keep going a tiny bit in case order differs, but we could break
                    pass
    except FileNotFoundError:
        return None, None, None
    return tuning, meter_info, diminution


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

    tuning, meter_info, diminution = extract_tbp_basic_metadata(
        matching_tab_file
    )
    print("  TBP header extracted:")
    print(f"    tuning: {tuning}")
    print(f"    meter_info: {meter_info}")
    print(f"    diminution: {diminution}")

    # initial preprocessing
    preprocessed_df = preprocess_df(csv_df)

    # Prepare per-file output subfolder: output/<base_name>/
    output_root = "output"
    file_output_dir = os.path.join(output_root, base_name)
    os.makedirs(file_output_dir, exist_ok=True)
    # Optional info
    print(f"  Writing outputs to {file_output_dir}")

    # Preprocessing already expanded multi-voice rows and added voice_order
    voice_dfs = split_df_by_voice(preprocessed_df)

    # Also process the combined dataframe (all voices) using voice_order / voice_squished
    combined_sequences = find_ornament_sequences(
        preprocessed_df,
        allow_variable_durations=False,
        max_ornament_duration_threshold=Fraction(1, 1),
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

    # Write per-file summary
    summary_lines = [f"Summary for {base_name}"]
    for key, val in sorted(summary_counts.items()):
        summary_lines.append(f"{key}: {val}")
    summary_path = os.path.join(file_output_dir, f"{base_name}_summary.txt")
    with open(summary_path, "w") as sf:
        sf.write("\n".join(summary_lines) + "\n")
    print(f"Summary written to {summary_path}")
