"""Play-level features: targeted receiver / defender statistics (Chapters 5-6), node features, edges and
play context (Chapter 6).

Functions that need the tables receive the `NFLData` object explicitly (`get_play(data, game_id, play_id)`).
The ball statistics and the global-feature vector are notebook contributions and are injected where needed.
"""
import numpy as np
import pandas as pd

from .data import get_play

# Roles of the tracking data, in the alphabetical order that pd.get_dummies would give them
PLAYER_ROLES = ["Defensive Coverage", "Other Route Runner", "Passer", "Targeted Receiver"]

# ---------------------------------------------------------------- 5.2 Targeted receiver and defenders
def get_target_stats(prepass_play_df, postpass_play_df):

    passer = prepass_play_df[prepass_play_df["player_role"] == "Passer"]["nfl_id"].unique()[0]

    landed_at_x = prepass_play_df["ball_land_x"].iloc[0]
    landed_at_y = prepass_play_df["ball_land_y"].iloc[0]

    targeted_receiver = prepass_play_df[prepass_play_df["player_role"] == "Targeted Receiver"]["nfl_id"].unique()[0]

    target_at_throw_x = prepass_play_df[(prepass_play_df["frame_id"] == prepass_play_df["frame_id"].max()) & (prepass_play_df["nfl_id"] == targeted_receiver)]["x"].values[0]
    target_at_throw_y = prepass_play_df[(prepass_play_df["frame_id"] == prepass_play_df["frame_id"].max()) & (prepass_play_df["nfl_id"] == targeted_receiver)]["y"].values[0]
    target_at_landing_x = postpass_play_df[(postpass_play_df["frame_id"] == postpass_play_df["frame_id"].max()) & (postpass_play_df["nfl_id"] == targeted_receiver)]["x"].values[0]
    target_at_landing_y = postpass_play_df[(postpass_play_df["frame_id"] == postpass_play_df["frame_id"].max()) & (postpass_play_df["nfl_id"] == targeted_receiver)]["y"].values[0]

    erb_dx, erb_dy = landed_at_x - target_at_throw_x ,  landed_at_y - target_at_throw_y
    expected_path_dist = np.sqrt(erb_dx**2 + erb_dy**2)
    erb_dir = ((90 - np.degrees(np.arctan2(erb_dy, erb_dx)) + 180) % 360) - 180

    rrb_dx, rrb_dy = target_at_landing_x - target_at_throw_x, target_at_landing_y - target_at_throw_y
    real_path_dist = np.sqrt(rrb_dx**2 + rrb_dy**2)
    rrb_dir = ((90 - np.degrees(np.arctan2(rrb_dy, rrb_dx)) + 180) % 360) - 180

    rb_dx, rb_dy = landed_at_x - target_at_landing_x, landed_at_y - target_at_landing_y
    catch_dist = np.sqrt(rb_dx**2 + rb_dy**2)
    rb_dir = ((90 - np.degrees(np.arctan2(rb_dy, rb_dx)) + 180) % 360) - 180

    target_stats = []
    target_stats.append({

        "game_id": prepass_play_df["game_id"].iloc[0],
        "play_id": prepass_play_df["play_id"].iloc[0],
        "nfl_id": targeted_receiver,

        "player_at_throw_x": target_at_throw_x,
        "player_at_throw_y": target_at_throw_y,

        "player_at_landing_x": target_at_landing_x,
        "player_at_landing_y": target_at_landing_y,

        "expected_path_distance": expected_path_dist,
        "expected_path_dist_x" : erb_dx,
        "expected_path_dist_y" : erb_dy,
        "expected_path_dir": erb_dir,

        "real_path_distance": real_path_dist,
        "real_path_dist_x" : rrb_dx,
        "real_path_dist_y" : rrb_dy,
        "real_path_dir": rrb_dir,

        "catch_distance": catch_dist,
        "catch_dist_x" : rb_dx,
        "catch_dist_y" : rb_dy,
        "catch_dir": rb_dir,

    })

    return target_stats

def get_predicted_defenders_stats(prepass_play_df, postpass_play_df):

    landed_at_x = prepass_play_df["ball_land_x"].iloc[0]
    landed_at_y = prepass_play_df["ball_land_y"].iloc[0]

    targeted_receiver = prepass_play_df[prepass_play_df["player_role"] == "Targeted Receiver"]["nfl_id"].unique()[0]

    target_at_throw_x = prepass_play_df[(prepass_play_df["frame_id"] == prepass_play_df["frame_id"].max()) & (prepass_play_df["nfl_id"] == targeted_receiver)]["x"].values[0]
    target_at_throw_y = prepass_play_df[(prepass_play_df["frame_id"] == prepass_play_df["frame_id"].max()) & (prepass_play_df["nfl_id"] == targeted_receiver)]["y"].values[0]
    target_at_landing_x = postpass_play_df[(postpass_play_df["frame_id"] == postpass_play_df["frame_id"].max()) & (postpass_play_df["nfl_id"] == targeted_receiver)]["x"].values[0]
    target_at_landing_y = postpass_play_df[(postpass_play_df["frame_id"] == postpass_play_df["frame_id"].max()) & (postpass_play_df["nfl_id"] == targeted_receiver)]["y"].values[0]

    defenders = prepass_play_df[(prepass_play_df["player_role"] == "Defensive Coverage") & (prepass_play_df["player_to_predict"] == True)]["nfl_id"].unique()
    defenders_stats = []

    if len(defenders) > 0:
        for defender in defenders:
            defender_at_throw_x = prepass_play_df[(prepass_play_df["frame_id"] == prepass_play_df["frame_id"].max()) & (prepass_play_df["nfl_id"] == defender)]["x"].values[0]
            defender_at_throw_y = prepass_play_df[(prepass_play_df["frame_id"] == prepass_play_df["frame_id"].max()) & (prepass_play_df["nfl_id"] == defender)]["y"].values[0]
            defender_at_landing_x = postpass_play_df[(postpass_play_df["frame_id"] == postpass_play_df["frame_id"].max()) & (postpass_play_df["nfl_id"] == defender)]["x"].values[0]
            defender_at_landing_y = postpass_play_df[(postpass_play_df["frame_id"] == postpass_play_df["frame_id"].max()) & (postpass_play_df["nfl_id"] == defender)]["y"].values[0]

            edb_dx, edb_dy = landed_at_x - defender_at_throw_x ,  landed_at_y - defender_at_throw_y
            expected_path_dist = np.sqrt(edb_dx**2 + edb_dy**2)
            edb_dir = ((90 - np.degrees(np.arctan2(edb_dy, edb_dx)) + 180) % 360) - 180

            rdb_dx, rdb_dy = defender_at_landing_x - defender_at_throw_x, defender_at_landing_y - defender_at_throw_y
            real_path_dist = np.sqrt(rdb_dx**2 + rdb_dy**2)
            rdb_dir = ((90 - np.degrees(np.arctan2(rdb_dy, rdb_dx)) + 180) % 360) - 180

            db_dx, db_dy = landed_at_x - defender_at_landing_x, landed_at_y - defender_at_landing_y
            catch_dist = np.sqrt(db_dx**2 + db_dy**2)
            db_dir = ((90 - np.degrees(np.arctan2(db_dy, db_dx)) + 180) % 360) - 180

            drt_dx, drt_dy = target_at_throw_x - defender_at_throw_x,  target_at_throw_y - defender_at_throw_y
            prepass_cover_dist = np.sqrt(drt_dx**2 + drt_dy**2)
            drt_dir = ((90 - np.degrees(np.arctan2(drt_dy, drt_dx)) + 180) % 360) - 180

            drc_dx, drc_dy =  target_at_landing_x - defender_at_landing_x, target_at_landing_y - defender_at_landing_y
            postpass_cover_dist = np.sqrt(drc_dx**2 + drc_dy**2)
            drc_dir = ((90 - np.degrees(np.arctan2(drc_dy, drc_dx)) + 180) % 360) - 180

            defenders_stats.append({

                "game_id": prepass_play_df["game_id"].iloc[0],
                "play_id": prepass_play_df["play_id"].iloc[0],
                "nfl_id": defender,

                "player_at_throw_x": defender_at_throw_x,
                "player_at_throw_y": defender_at_throw_y,

                "player_at_landing_x": defender_at_landing_x,
                "player_at_landing_y": defender_at_landing_y,

                "expected_path_distance": expected_path_dist,
                "expected_path_dist_x" : edb_dx,
                "expected_path_dist_y" : edb_dy,
                "expected_path_dir": edb_dir,

                "real_path_distance": real_path_dist,
                "real_path_dist_x" : rdb_dx,
                "real_path_dist_y" : rdb_dy,
                "real_path_dir": rdb_dir,

                "catch_distance": catch_dist,
                "catch_dist_x" : db_dx,
                "catch_dist_y" : db_dy,
                "catch_dir": db_dir,

                "defender_target_at_throw_distance": prepass_cover_dist,
                "defender_target_at_throw_dist_x": drt_dx,
                "defender_target_at_throw_dist_y": drt_dy,
                "defender_target_at_throw_dir": drt_dir,

                "defender_target_at_landing_distance": postpass_cover_dist,
                "defender_target_at_landing_dist_x": drc_dx,
                "defender_target_at_landing_dist_y": drc_dy,
                "defender_target_at_landing_dir": drc_dir

            })
    return defenders_stats


def get_play_stats(data, game_id, play_id, ball_stats_fn):
    """Ball, targeted-receiver and predicted-defenders statistics of one play (three dataframes).
    `ball_stats_fn` is the notebook's `get_ball_stats`."""
    prepass_play_df, postpass_play_df = get_play(data, game_id, play_id)

    ball_stats = ball_stats_fn(prepass_play_df)
    target_stats = get_target_stats(prepass_play_df, postpass_play_df)
    predicted_defenders_stats = get_predicted_defenders_stats(prepass_play_df, postpass_play_df)

    return  pd.DataFrame(ball_stats).set_index(['game_id', 'play_id']), \
            pd.DataFrame(target_stats).set_index(['game_id', 'play_id', 'nfl_id']), \
            pd.DataFrame(predicted_defenders_stats).set_index(['game_id', 'play_id', 'nfl_id']) if predicted_defenders_stats != [] else pd.DataFrame(predicted_defenders_stats)


# ---------------------------------------------------------------- 5.3 Predicted players dataframe
def get_predicted_players_df(data, game_id, play_id, ball_stats_fn):

    per_play_cols = ['player_role',
                     'player_at_throw_x', 'player_at_throw_y', 'player_at_landing_x', 'player_at_landing_y',
                     'expected_path_distance', 'expected_path_dist_x','expected_path_dist_y', 'expected_path_dir',
                     'real_path_distance', 'real_path_dist_x', 'real_path_dist_y', 'real_path_dir',
                     'catch_distance', 'catch_dist_x', 'catch_dist_y', 'catch_dir',
                     'defender_target_at_throw_distance', 'defender_target_at_throw_dist_x', 'defender_target_at_throw_dist_y', 'defender_target_at_throw_dir',
                     'defender_target_at_landing_distance', 'defender_target_at_landing_dist_x', 'defender_target_at_landing_dist_y', 'defender_target_at_landing_dir']

    target_cols = ['player_at_throw_x', 'player_at_throw_y', 'player_at_landing_x',
       'player_at_landing_y', 'expected_path_distance', 'expected_path_dist_x',
       'expected_path_dist_y', 'expected_path_dir', 'real_path_distance',
       'real_path_dist_x', 'real_path_dist_y', 'real_path_dir',
       'catch_distance', 'catch_dist_x', 'catch_dist_y', 'catch_dir']

    def_cols = ['player_at_throw_x', 'player_at_throw_y', 'player_at_landing_x',
       'player_at_landing_y', 'expected_path_distance', 'expected_path_dist_x',
       'expected_path_dist_y', 'expected_path_dir', 'real_path_distance',
       'real_path_dist_x', 'real_path_dist_y', 'real_path_dir',
       'catch_distance', 'catch_dist_x', 'catch_dist_y', 'catch_dir',
       'defender_target_at_throw_distance', 'defender_target_at_throw_dist_x',
       'defender_target_at_throw_dist_y', 'defender_target_at_throw_dir',
       'defender_target_at_landing_distance',
       'defender_target_at_landing_dist_x',
       'defender_target_at_landing_dist_y', 'defender_target_at_landing_dir']

    _, target_stats_df, defenders_stats_df = get_play_stats(data, game_id, play_id, ball_stats_fn)

    if target_stats_df.empty:
        print(f"No targeted receiver on play {play_id} \n")
        return pd.DataFrame(columns=per_play_cols)

    if defenders_stats_df.empty:
        print(f"No defenders tracked on play {play_id} \n")
        defenders_stats_df = pd.DataFrame(columns=def_cols)

    target_stats_df['player_role'] = 'Targeted Receiver'
    defenders_stats_df['player_role'] = 'Defensive Coverage'

    # --- Align columns ---
    target_stats_df = target_stats_df.reindex(columns= per_play_cols, fill_value= np.nan)
    defenders_stats_df = defenders_stats_df.reindex(columns= per_play_cols, fill_value= np.nan)

    # --- Merge ---
    combined_df = pd.concat([target_stats_df, defenders_stats_df], ignore_index= False)
    combined_df.index.names = ["game_id", "play_id", "nfl_id"]

    return combined_df

# ---------------------------------------------------------------- 6.1-6.2 Node and edge features
def preprocess_play(data, game_id, play_id):
    """Node features of every frame of a play: {frame_id: array [players, 1 + F_node]} (first column = nfl_id)."""
    play_df, _ = get_play(data, game_id, play_id)

    # Select only relevant columns
    gnn_cols = ['game_id', 'play_id', 'frame_id', 'nfl_id',
                'player_role', 'x', 'y', 's', 'a', 'dir', 'o']

    play_df = play_df[gnn_cols].copy()

    # One-hot encode roles once. The categories are fixed: a play without one of the roles (e.g. no "Other Route
    # Runner") still gets the same 4 role columns, so every play has NODE_IN_DIM features.
    play_df['player_role'] = pd.Categorical(play_df['player_role'], categories=PLAYER_ROLES)
    play_df = pd.get_dummies(play_df, columns=['player_role'])
    play_df.columns = [col.replace(" ", "_") for col in play_df.columns]

    # Merge physical attributes once
    physical_df = data.preprocessed_players_df.reset_index()
    play_df = play_df.merge(physical_df, on="nfl_id", how="left")

    # Drop useless columns once
    play_df = play_df.drop(['game_id', 'play_id'], axis=1)

    # Split by frame: dict {frame_id: numpy array}
    play_df = play_df.astype(float)
    frames = {
    fid: df.drop("frame_id", axis=1)
           .to_numpy()
    for fid, df in play_df.groupby("frame_id")
    }

    return frames

def get_node_features(preprocessed_play_dict, frame_id):
    frames =  preprocessed_play_dict[frame_id]

    node_features = frames[:, 1:]
    node_ids = frames[:, 0].astype(int)

    return node_features, node_ids

def pack_play_features(preprocessed_play_dict):
    """
    Converts preprocessed play dict into flattened node features with unique node IDs.

    Args:
        preprocessed_play_dict: dict of {frame_id: frame_data}
        num_players: number of players per frame

    Returns:
        all_node_features: [num_nodes, F_node]
        node_ids: [num_nodes] (t*num_players + n)
        nfl_ids: [num_nodes] original player IDs repeated per frame
    """
    node_features_list = []
    nfl_ids_list = []

    for t, frame_id in enumerate(sorted(preprocessed_play_dict.keys())):
        features, nfl_ids_frame = get_node_features(preprocessed_play_dict, frame_id)       # [num_players_in_frame, F_node], [num_players_in_frame]
        num_players = len(nfl_ids_frame)

        # Pad if fewer than num_players
        if features.shape[0] < num_players:
            pad_feats = np.zeros((num_players - features.shape[0], features.shape[1]))
            features = np.vstack([features, pad_feats])

            pad_ids = np.zeros(num_players - len(nfl_ids_frame), dtype=nfl_ids_frame.dtype)
            nfl_ids_frame = np.hstack([nfl_ids_frame, pad_ids])

        node_features_list.append(features)
        nfl_ids_list.append(nfl_ids_frame)

    # Stack along frames and flatten
    all_node_features = np.vstack(node_features_list)                                        # [num_frames*num_players, F_node]

    # Node IDs: t*num_players + n
    num_frames = len(node_features_list)
    node_ids = np.arange(num_frames * num_players)

    return all_node_features, node_ids, nfl_ids_frame

def get_edge_features(node_features, num_frames, num_players, threshold=15, k=3, return_splits=False):
    """
    Compute spatial (KNN per frame) + temporal (same-player across frames) edges
    for a play.

    Args:
        node_features: np.array shape [T, P, F] where:
            positions = [:, :, :2], speed = [:, :, 2], accel = [:, :, 3],
            direction = [:, :, 4], orientations = [:, :, 5]
        threshold: unused here (kept for API compatibility)
        k: number of spatial neighbors per player (does NOT include self)
        return_splits: if True returns (edge_attr, edge_index, n_spatial_edges)

    Returns:
        edge_attr: [E, 7] float32, columns = [dist, dx, dy, speed_diff, accel_diff, dir_diff, ori_diff]
        edge_index: [E, 2] int64, flattened indices in range [0, T*P-1] (dst, src)
        (optional) n_spatial_edges: int number of spatial edges (useful to split arrays)
    """
    positions = node_features[:, :2]   # [T* P, 2]
    speed = node_features[:, 2]        # [T* P]
    accel = node_features[:, 3]        # [T* P]
    direction = node_features[:, 4]    # [T* P]
    orientations = node_features[:, 5] # [T* P]

    spatial_attr_list = []
    spatial_index_list = []

    # --- Spatial edges: compute KNN per frame and offset indices by frame ---
    for t in range(num_frames):
        pos_t = positions[t * num_players : (t + 1) * num_players]    # [P, 2]

        # pairwise distances (P x P)
        diff = pos_t[:, None, :] - pos_t[None, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist_matrix, np.inf)

        # k nearest neighbors per node
        knn_idx = np.argpartition(dist_matrix, kth=k, axis=1)[:, :k]  # [P, k]

        # source/target indices in frame-local coords
        i_local = np.repeat(np.arange(num_players), k)                # [P*k]
        j_local = knn_idx.reshape(-1)                                 # [P*k]

        # Offset to flattened node indices across frames
        frame_offset = t * num_players
        i_idx = (i_local + frame_offset).astype(np.int64)
        j_idx = (j_local + frame_offset).astype(np.int64)

        xi = pos_t[i_local]
        xj = pos_t[j_local]

        dx = (xj[:, 0] - xi[:, 0]).astype(np.float32)
        dy = (xj[:, 1] - xi[:, 1]).astype(np.float32)
        dist = np.sqrt(dx**2 + dy**2).astype(np.float32)

        # speed/accel/dir/ori diffs computed in frame-local indices
        spd_diff = (speed[j_idx] - speed[i_idx]).astype(np.float32)
        acc_diff = (accel[j_idx] - accel[i_idx]).astype(np.float32)

        dir_i = direction[i_idx]
        dir_j = direction[j_idx]
        dir_diff = (((dir_j - dir_i + 180) % 360) - 180).astype(np.float32)

        ori_i = orientations[i_idx]
        ori_j = orientations[j_idx]
        ori_diff = (((ori_j - ori_i + 180) % 360) - 180).astype(np.float32)

        spatial_attr_list.append(np.stack([dist, dx, dy, spd_diff, acc_diff, dir_diff, ori_diff], axis=1))
        spatial_index_list.append(np.stack([i_idx, j_idx], axis=1))

    # concatenate spatial across frames
    if spatial_attr_list:
        edge_spatial_attr = np.concatenate(spatial_attr_list, axis=0).astype(np.float32)    # [T*P*k, 7]
        edge_spatial_index = np.concatenate(spatial_index_list, axis=0).astype(np.int64)    # [T*P*k, 2]
    else:
        edge_spatial_attr = np.zeros((0,7), dtype=np.float32)
        edge_spatial_index = np.zeros((0,2), dtype=np.int64)

    n_spatial = edge_spatial_attr.shape[0]

    # --- Temporal edges: connect same player across consecutive frames with proper offsets ---
    temporal_attr_list = []
    temporal_index_list = []

    for t in range(num_frames - 1):
        start_idx_t = t * num_players
        start_idx_t1 = (t + 1) * num_players

        pos_t = positions[start_idx_t : start_idx_t + num_players]    # [P, 2]
        pos_t1 = positions[start_idx_t1 : start_idx_t1 + num_players] # [P, 2]

        # flattened indices
        i_idx = np.arange(start_idx_t, start_idx_t + num_players)
        j_idx = np.arange(start_idx_t1, start_idx_t1 + num_players)

        # flattened indices for frame t and t+1
        i = (np.arange(num_players) + t * num_players).astype(np.int64)        # [P]
        j = (np.arange(num_players) + (t + 1) * num_players).astype(np.int64)  # [P]

        dx = (pos_t1[:, 0] - pos_t[:, 0]).astype(np.float32)
        dy = (pos_t1[:, 1] - pos_t[:, 1]).astype(np.float32)
        dist = np.sqrt(dx**2 + dy**2).astype(np.float32)

        spd_diff = (speed[j_idx] - speed[i_idx]).astype(np.float32)
        acc_diff = (accel[j_idx] - accel[i_idx]).astype(np.float32)
        dir_diff = (((direction[j_idx] - direction[i_idx] + 180) % 360) - 180).astype(np.float32)
        ori_diff = (((orientations[j_idx] - orientations[i_idx] + 180) % 360) - 180).astype(np.float32)

        temporal_attr_list.append(np.stack([dist, dx, dy, spd_diff, acc_diff, dir_diff, ori_diff], axis=1))
        temporal_index_list.append(np.stack([i_idx, j_idx], axis=1))

    if temporal_attr_list:
        edge_temporal_attr = np.concatenate(temporal_attr_list, axis=0).astype(np.float32)    # [(T-1)*P, 7]
        edge_temporal_index = np.concatenate(temporal_index_list, axis=0).astype(np.int64)    # [(T-1)*P, 2]
    else:
        edge_temporal_attr = np.zeros((0,7), dtype=np.float32)
        edge_temporal_index = np.zeros((0,2), dtype=np.int64)

    # --- Combine spatial + temporal ---
    edge_attr = np.concatenate([edge_spatial_attr, edge_temporal_attr], axis=0)
    edge_index = np.concatenate([edge_spatial_index, edge_temporal_index], axis=0)

    if return_splits:
        return edge_attr, edge_index, n_spatial
    return edge_attr, edge_index

# ---------------------------------------------------------------- 6.3 Play context (global features)
def preprocess_aux(df):
    df = df.copy()
    keys = df[["game_id", "play_id"]]

    # ---- Normalize fixed-range columns ----
    df["game_clock_sec"] = (
        df["game_clock"]
        .str.split(":")
        .apply(lambda x: int(x[0]) * 60 + int(x[1]))
    )

    df["game_clock_norm"] = df["game_clock_sec"] / 900.0
    df["yards_to_go_norm"] = df["yards_to_go"] / 100.0

    # ---- Offense-relative pre-snap score ----
    is_home_offense = df["possession_team"] == df["home_team_abbr"]
    df["offense_score"] = np.where(is_home_offense, df["pre_snap_home_score"], df["pre_snap_visitor_score"])
    df["defense_score"] = np.where(is_home_offense, df["pre_snap_visitor_score"], df["pre_snap_home_score"])

    # ---- One-hot encode categorical columns ----
    ohe_cols = [
        "quarter", "down", "pass_result", "offense_formation",
        "route_of_targeted_receiver", "play_action", "dropback_type"
    ]
    df_ohe = pd.get_dummies(df[ohe_cols], columns=ohe_cols, prefix=ohe_cols)

    # ---- Final dataframe ----
    final_df = pd.concat(
        [keys,
         df[["offense_score", "defense_score", "game_clock_norm", "yards_to_go_norm", "expected_points", "expected_points_added", "defenders_in_the_box"]],
         df_ohe],
        axis=1
    )

    return final_df
