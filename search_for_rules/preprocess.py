import pandas as pd
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
    df = squish_to_one_staff(df)
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
):
    """Add a per-onset vertical ordering column across all voices before splitting.

    Logic:
      - Single note at an onset -> 0
      - Multiple notes: rank unique voices numerically ascending (voice '0' highest) -> 0..N-1
      - Multiple rows sharing the same voice at that onset (if any) all get the same rank.
    """
    new_col = "voice_tab"
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
    # Provide an alias name if user expects a different label (voice_tab)
    if "voice_tab" not in df.columns:
        df["voice_tab"] = df[new_col]
    return df


def split_df_by_voice(df, voice_col: str = "voice"):
    """
    Returns a dictionary of DataFrames, one for each unique voice bucket.
    """
    return {
        voice: df[df[voice_col] == voice].copy().reset_index(drop=True)
        for voice in df[voice_col].dropna().unique()
    }
