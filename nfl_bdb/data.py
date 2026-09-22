"""Data loading, sanitization, player table and play retrieval (notebook Sections 2-4).

All the tables of the project travel together in one `NFLData` object, so no function relies on notebook globals:
`get_play(data, game_id, play_id)` always reads the CURRENT (standardized) dataframes.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Columns of the auxiliary (supplementary) file used as play context
CONTEXT_COLS = [
    'game_id', 'play_id', 'quarter', 'game_clock', 'down', 'yards_to_go',
    'pass_result', 'offense_formation', 'route_of_targeted_receiver',
    'play_action', 'dropback_type', 'expected_points', 'expected_points_added',
    'possession_team', 'home_team_abbr', 'visitor_team_abbr',
    'pre_snap_home_score', 'pre_snap_visitor_score', 'defenders_in_the_box',
]

# Role uniformization (2.2): LB/MLB are the same inside-linebacker role, T is ambiguous while OT is not
POSITION_MERGES = {"LB": "ILB", "MLB": "ILB", "T": "OT"}
# Individual role corrections (2.2), found by inspecting the value_counts of player_position
PLAYER_POSITION_FIXES = {52991: "FS", 56275: "FS", 46162: "SS", 45244: "TE"}

FIELD_LENGTH = 120
FIELD_WIDTH = 53.3


@dataclass
class NFLData:
    """The tables of the project. `input_df`/`output_df` are the pre-pass / post-pass tracking data."""
    input_df: pd.DataFrame
    output_df: pd.DataFrame
    aux_df: pd.DataFrame
    players_df: Optional[pd.DataFrame] = None                # one row per player, physical attributes (3.1)
    preprocessed_players_df: Optional[pd.DataFrame] = None   # normalized + one-hot version of players_df (3.3)
    aux_df_preprocessed: Optional[pd.DataFrame] = None       # normalized + one-hot play context (6.3)


# ---------------------------------------------------------------- 2. Loading and sanitization
def load_weeks_parallel(week_nums, train_path, max_workers=8):
    """Read input/output csv of the given weeks in parallel threads and concatenate them."""
    def load_week(w):
        inp = pd.read_csv(train_path / f"input_2023_w{w:02d}.csv")
        out = pd.read_csv(train_path / f"output_2023_w{w:02d}.csv")
        return w, inp, out

    inputs, outputs = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_week, w): w for w in week_nums}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading weeks"):
            w, inp, out = future.result()
            inputs.append(inp)
            outputs.append(out)

    return pd.concat(inputs, ignore_index=True), pd.concat(outputs, ignore_index=True)


def load_raw_data(cfg):
    """Tracking data (input + output) and play context of the weeks in `cfg.WEEKS`, restricted to the plays
    that have output data. Returns an `NFLData`."""
    input_df, output_df = load_weeks_parallel(cfg.WEEKS, cfg.DATA_PATH, max_workers=8)
    print(f"\nInput Data loaded: {len(input_df)} rows, {input_df['game_id'].nunique()} games")
    print(f"Output Data loaded: {len(output_df)} rows, {output_df['game_id'].nunique()} games")

    aux = pd.read_csv(cfg.AUXILIARY_DATA_PATH)
    print(f"Auxiliary Data loaded: {len(aux)} rows, {aux['game_id'].nunique()} games")

    return NFLData(input_df, output_df, _build_aux_df(aux, output_df))


def _build_aux_df(aux, output_df):
    """Play context of the plays that appear in output_df (MultiIndex game_id, play_id)."""
    output_idx = output_df.set_index(["game_id", "play_id"]).index
    aux_context = aux[CONTEXT_COLS].set_index(["game_id", "play_id"])
    aux_df = aux_context.loc[aux_context.index.intersection(output_idx)]
    aux_df.reset_index(inplace=True)
    return aux_df


def sanitize_positions(input_df):
    """Role uniformization and individual role corrections (in place on `input_df`)."""
    input_df["player_position"] = input_df["player_position"].replace(POSITION_MERGES)
    for nfl_id, position in PLAYER_POSITION_FIXES.items():
        input_df["player_position"] = input_df["player_position"].mask(input_df["nfl_id"] == nfl_id, position)
    return input_df


def filter_valid_plays(input_df, min_players, verbose=True):
    """Plays that have a `Passer` and at least `min_players` players in every frame.
    Returns (valid, invalid): dataframes with the (game_id, play_id) of the valid plays / the diagnostics of the dropped ones."""
    flagged = input_df.assign(is_passer=input_df["player_role"] == "Passer")

    per_frame_counts = (
        flagged.groupby(["game_id", "play_id", "frame_id"])["nfl_id"]
        .nunique()
        .reset_index(name="num_players")
    )
    min_players_per_play = (
        per_frame_counts.groupby(["game_id", "play_id"])["num_players"]
        .min()
        .reset_index(name="min_players_in_play")
    )
    has_passer_per_play = (
        flagged.groupby(["game_id", "play_id"])["is_passer"]
        .any()
        .reset_index(name="has_passer")
    )

    play_validity = min_players_per_play.merge(has_passer_per_play, on=["game_id", "play_id"])
    play_validity["is_valid"] = play_validity["has_passer"] & (
        play_validity["min_players_in_play"] >= min_players
    )

    valid = play_validity.loc[play_validity["is_valid"], ["game_id", "play_id"]]
    invalid = play_validity.loc[~play_validity["is_valid"]]

    if verbose:
        no_passer = (~invalid["has_passer"]).sum()
        too_few = (invalid["min_players_in_play"] < min_players).sum()
        print(f"Valid plays: {len(valid)} | dropped: {len(invalid)} "
              f"(no Passer: {no_passer}, < {min_players} players/frame: {too_few})")
        if len(invalid) > 0:
            print(invalid)

    return valid, invalid


def apply_play_filter(data, min_players):
    """Keep only the valid plays in every table of `data` (see `filter_valid_plays`)."""
    valid_plays_idx, dropped_plays_idx = filter_valid_plays(data.input_df, min_players=min_players)
    valid_multiindex = pd.MultiIndex.from_frame(valid_plays_idx)
    print(f"\nTotal valid plays after filtering: {len(valid_multiindex)}")

    keep = lambda df: df[df.set_index(["game_id", "play_id"]).index.isin(valid_multiindex)].reset_index(drop=True)
    data.input_df, data.output_df, data.aux_df = keep(data.input_df), keep(data.output_df), keep(data.aux_df)
    return data


# ---------------------------------------------------------------- 3. Players physique
def ft_to_m(height_str):
    """'6-2' (feet-inches) -> meters."""
    try:
        feet, inches = map(int, height_str.split('-'))
        return (feet * 12 + inches) * 0.0254
    except:
        return None


def birthday_to_age(players_df, ref_date):
    """Age in years at `ref_date` (today when None)."""
    if ref_date is None:
        ref_date = pd.Timestamp.today()
    else:
        ref_date = pd.Timestamp(ref_date)

    birth_dates = pd.to_datetime(players_df["player_birth_date"])
    ages = (ref_date - birth_dates).dt.days / 365.25
    return ages


def build_players_df(input_df, ref_date="2023-11-15"):
    """One row per player (index nfl_id) with age, height (m), weight (kg) and BMI.
    The age is computed at the mean date of the 2023 season (`ref_date`)."""
    player_columns = [
        'player_name', 'player_height', 'player_weight',
        'player_birth_date', 'player_position', 'player_side'
    ]

    # Drop duplicates to keep only one row per nfl_id
    players_df = input_df.drop_duplicates(subset=['nfl_id'])[['nfl_id'] + player_columns].set_index('nfl_id')

    players_df["player_age"] = birthday_to_age(players_df, ref_date=ref_date)
    players_df['player_height_m'] = players_df["player_height"].apply(ft_to_m)
    players_df["player_weight_kg"] = players_df["player_weight"] * 0.45359237
    players_df.drop(["player_height", "player_weight", "player_birth_date"], axis=1, inplace=True)
    players_df["player_BMI"] = players_df["player_weight_kg"] / (players_df["player_height_m"] ** 2)
    return players_df


def get_player(players_df, nfl_id):
    return players_df.loc[nfl_id]


def normalize_players(df):
    """Adds, for age/height/weight/BMI, a global z-score and a z-score within the player's position."""
    numeric_cols = ["player_age", "player_height_m", "player_weight_kg", "player_BMI"]

    # --- Global normalization ---
    global_norm = df[numeric_cols].copy()
    global_norm = (global_norm - global_norm.mean()) / global_norm.std(ddof=0)
    global_norm.columns = [col + "_global_norm" for col in numeric_cols]

    # --- Positional normalization ---
    positional_norm = (
        df.groupby("player_position")[numeric_cols]
          .transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    )
    positional_norm.columns = [col + "_position_norm" for col in numeric_cols]

    # --- Concatenate all together ---
    df_out = pd.concat([df, global_norm, positional_norm], axis=1)
    return df_out


def preprocess_players(players_df):
    """Model-ready player table: normalized physical attributes + one-hot side/position (NaN -> 0)."""
    preprocessed_players_df = normalize_players(players_df.drop(["player_name"], axis=1))

    preprocessed_players_df = pd.get_dummies(
        preprocessed_players_df,
        columns=['player_side', 'player_position']
    )

    preprocessed_players_df.fillna(0, inplace=True)
    return preprocessed_players_df


# ---------------------------------------------------------------- 4. Play retrieval and standardization
def get_play(data, game_id, play_id):
    """(pre-pass, post-pass) tracking rows of one play. The post-pass `frame_id`s are shifted by the number of
    pre-pass frames, so the two dataframes describe one continuous timeline. (None, None) when a part is missing."""
    prepass_play_df = data.input_df[(data.input_df["game_id"] == game_id) & (data.input_df["play_id"] == play_id)]
    if prepass_play_df.empty:
        print(f"No input data found for game_id={game_id}, play_id={play_id}")
        return None, None

    postpass_play_df = data.output_df[(data.output_df["game_id"] == game_id) & (data.output_df["play_id"] == play_id)].copy()
    if postpass_play_df.empty:
        print(f"No output data found for game_id={game_id}, play_id={play_id}")
        return None, None

    prepass_num_frames = prepass_play_df["frame_id"].max()
    postpass_play_df["frame_id"] += prepass_num_frames

    return prepass_play_df, postpass_play_df


def standardize_plays(input_df, output_df):
    """Mirror the plays that go left so that all of them move toward the right.
    Input tracking data (before the pass) contains `play_direction`; the output (after the pass) does not.
    Returns the standardized (input_df, output_df); `input_df` is modified in place."""
    field_length = FIELD_LENGTH
    field_width = FIELD_WIDTH

    # Plays with Left Play Direction
    left_plays = (
        input_df.loc[input_df["play_direction"].str.lower() == "left", ["game_id", "play_id"]]
        .drop_duplicates()
        .assign(left_play=True)
    )

    output_df = output_df.merge(left_plays, on=["game_id", "play_id"], how="left")
    output_df["left_play"] = output_df["left_play"].fillna(False)

    # Input Standardization
    mask_left_input = input_df["play_direction"].str.lower() == "left"

    input_df.loc[mask_left_input, "x"] = field_length - input_df.loc[mask_left_input, "x"]
    input_df.loc[mask_left_input, "y"] = field_width - input_df.loc[mask_left_input, "y"]

    if "absolute_yardline_number" in input_df.columns:
        input_df.loc[mask_left_input, "absolute_yardline_number"] = (
            field_length - input_df.loc[mask_left_input, "absolute_yardline_number"]
        )

    if "ball_land_x" in input_df.columns:
        input_df.loc[mask_left_input, "ball_land_x"] = field_length - input_df.loc[mask_left_input, "ball_land_x"]
    if "ball_land_y" in input_df.columns:
        input_df.loc[mask_left_input, "ball_land_y"] = field_width - input_df.loc[mask_left_input, "ball_land_y"]

    for col in ["o", "dir"]:
        if col in input_df.columns:
            input_df.loc[mask_left_input, col] = (input_df.loc[mask_left_input, col] + 180) % 360

    # Output Standardization
    mask_left_output = output_df["left_play"]

    output_df.loc[mask_left_output, "x"] = field_length - output_df.loc[mask_left_output, "x"]
    output_df.loc[mask_left_output, "y"] = field_width - output_df.loc[mask_left_output, "y"]

    input_df.drop(columns=["play_direction"], inplace=True)
    output_df.drop(columns=["left_play"], inplace=True)

    return input_df, output_df
