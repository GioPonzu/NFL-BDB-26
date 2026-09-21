"""Dataset construction (notebook Chapter 7): stratified split by play length, padded per-play tensors,
PyG batches.

Every function receives its inputs explicitly (`NFLData`, `Config`, feature callables): nothing is read from
notebook globals.
"""
import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm.auto import tqdm

from .features import pack_play_features, get_edge_features


# ---------------------------------------------------------------- Stratified sampling and split by play length
def compute_play_lengths(data, cfg):
    """One row per play: number of input (pre-pass) and output (post-pass) frames, plus the equal-frequency bins of
    both lengths and their joint label `length_bin`."""
    input_len_per_play = (
        data.input_df.groupby(["game_id", "play_id"])["frame_id"]
        .max()
        .rename("num_input_frames")
    )
    output_len_per_play = (
        data.output_df.groupby(["game_id", "play_id"])["frame_id"]
        .max()
        .rename("num_output_frames")
    )

    play_lengths = pd.concat([input_len_per_play, output_len_per_play], axis=1).dropna()
    play_lengths["num_output_frames"] = play_lengths["num_output_frames"].astype(int)

    play_lengths["input_len_bin"] = equal_frequency_bins(play_lengths["num_input_frames"], cfg.N_LENGTH_BINS)
    play_lengths["output_len_bin"] = equal_frequency_bins(play_lengths["num_output_frames"], cfg.N_LENGTH_BINS)
    play_lengths["length_bin"] = (
        play_lengths["input_len_bin"].astype(str) + " | " + play_lengths["output_len_bin"].astype(str)
    )
    return play_lengths


def equal_frequency_bins(series, q):
    """Quantile-cut by rank so every bin gets an (as close as possible) equal COUNT of rows, instead of pd.qcut's
    value-based cut which can produce badly unequal bins when the raw values are heavily tied."""
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=q, duplicates="drop")


def uniform_sample_by_bin(df, bin_col, n_total, random_state):
    """Sample n_total rows spread as evenly as possible across bin_col
    categories. Bins that run out of rows (fewer members than their
    equal share) hand their leftover quota to the remaining bins each
    round, so every available bin is drained before any bin is
    over-sampled, and the total always comes out to n_total."""
    rng = np.random.RandomState(random_state)
    pools = {b: g for b, g in df.groupby(bin_col) if len(g) > 0}

    selected_idx = []
    n_left = n_total
    bins_left = list(pools.keys())
    while n_left > 0 and bins_left:
        order = rng.permutation(len(bins_left))
        bins_left = [bins_left[i] for i in order]
        share = max(1, n_left // len(bins_left))
        next_bins_left = []
        for b in bins_left:
            avail = pools[b]
            take = min(share, len(avail))
            if take > 0:
                chosen = avail.sample(n=take, random_state=random_state)
                selected_idx.extend(chosen.index.tolist())
                pools[b] = avail.drop(chosen.index)
                n_left -= take
            if len(pools[b]) > 0:
                next_bins_left.append(b)
            if n_left <= 0:
                break
        bins_left = next_bins_left

    return df.loc[selected_idx]


@dataclass
class Splits:
    """Play-level split. Each table is indexed by (game_id, play_id) and carries the length columns."""
    play_lengths: pd.DataFrame     # all the plays
    sampled_plays: pd.DataFrame    # the stratified pool (N_TRAIN + N_VAL + N_TEST plays)
    train_plays: pd.DataFrame
    val_plays: pd.DataFrame
    test_plays: pd.DataFrame


def make_splits(play_lengths, cfg):
    """Stratified sample of `cfg.N_TOTAL` plays (uniform across joint length bins) and stratified train/val/test split.
    train_test_split works on plays, so no (game_id, play_id) is shared between the three sets."""
    sampled_plays = uniform_sample_by_bin(play_lengths, "length_bin", cfg.N_TOTAL, cfg.RANDOM_SEED)

    # Bins with a single play can't be stratified further (train_test_split
    # needs >= 2 members per class); merge those into a shared "rare" bucket.
    strat_counts = sampled_plays["length_bin"].value_counts()
    safe_strat_col = sampled_plays["length_bin"].where(
        sampled_plays["length_bin"].map(strat_counts) >= 2, other="__rare__"
    )

    train_plays, temp_plays = train_test_split(
        sampled_plays,
        test_size=cfg.N_VAL + cfg.N_TEST,
        random_state=cfg.RANDOM_SEED,
        shuffle=True,
        stratify=safe_strat_col,
    )

    temp_strat_col = safe_strat_col.loc[temp_plays.index]
    temp_strat_counts = temp_strat_col.value_counts()
    temp_strat_col = temp_strat_col.where(temp_strat_col.map(temp_strat_counts) >= 2, other="__rare__")

    val_plays, test_plays = train_test_split(
        temp_plays,
        test_size=cfg.N_TEST,
        random_state=cfg.RANDOM_SEED,
        shuffle=True,
        stratify=temp_strat_col,
    )

    print(f"Sampled pool: {len(sampled_plays)} plays "
          f"(covering {sampled_plays['length_bin'].nunique()}/{play_lengths['length_bin'].nunique()} length bins)")
    print(f"train={len(train_plays)}  val={len(val_plays)}  test={len(test_plays)}")
    return Splits(play_lengths, sampled_plays, train_plays, val_plays, test_plays)


# ---------------------------------------------------------------- 7.3 Padded per-play dataset
class PaddedPlayDataset(Dataset):
    """One item per play, padded to the limits of `cfg`.

    preprocess_play_fn(game_id, play_id) -> {frame_id: node features}          (features.preprocess_play)
    global_features_fn(game_id, play_id, global_mode) -> (pred_mask, vector)   (notebook: get_global_features)
    """

    def __init__(self, input_df, output_df, preprocess_play_fn, global_features_fn, cfg,
                 split='train', global_mode="all"):

        self.output_df = output_df
        self.global_features_fn = global_features_fn

        self.max_frames = cfg.MAX_INPUT_FRAMES
        self.max_nodes = cfg.MAX_INPUT_NODES
        self.max_output_frames = cfg.MAX_OUTPUT_FRAMES
        self.max_output_nodes = cfg.MAX_OUTPUT_NODES
        self.node_in_dim = cfg.NODE_IN_DIM
        self.edge_in_dim = cfg.EDGE_IN_DIM
        self.k = cfg.K

        # Which global features to include (see get_global_features): a parameter of the dataset, not a post-hoc
        # slicing of tensors already built -- same code path for the baseline and for every ablation.
        self.global_mode = global_mode

        # Post-pass rows of each play, indexed once (a filter of the whole output_df per item would be very slow)
        self._output_by_play = {key: g for key, g in output_df.groupby(["game_id", "play_id"])}

        # Build dataset list
        self.dataset = []
        plays = input_df[["game_id", "play_id"]].drop_duplicates()

        for _, row in tqdm(plays.iterrows(), total=len(plays), desc=f"Processing {split}"):
            game = row["game_id"]
            play = row["play_id"]
            prep = preprocess_play_fn(game, play)
            self.dataset.append((game, play, prep))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        game, play, prep_dict = self.dataset[idx]

        # Allocate padded tensors
        max_edges = self.max_nodes * (self.max_frames * self.k + self.max_frames - 1)
        x_nodes = torch.zeros((self.max_frames * self.max_nodes, self.node_in_dim), dtype=torch.float32)
        x_edges = torch.full((max_edges, self.edge_in_dim), fill_value=-1.0, dtype=torch.float32)
        edge_index = torch.full((max_edges, 2), fill_value=-1, dtype=torch.int64)

        edge_mask = torch.zeros(max_edges, dtype=torch.bool)
        input_mask = torch.zeros((self.max_frames, self.max_nodes), dtype=torch.bool)

        # Global features
        pred_mask, global_feats = self.global_features_fn(game, play, self.global_mode)

        all_node_features, all_node_ids, node_ids = pack_play_features(prep_dict)

        num_nodes = len(node_ids)
        num_frames = int(len(all_node_ids) / len(node_ids))

        features_tensor = torch.tensor(all_node_features, dtype=torch.float32)
        if features_tensor.shape[1] != self.node_in_dim:
            raise ValueError(f"Play {(game, play)}: {features_tensor.shape[1]} node features, "
                             f"expected {self.node_in_dim} (cfg.NODE_IN_DIM)")
        x_nodes[:num_frames * num_nodes] = features_tensor
        input_mask[:num_frames, :num_nodes] = True

        edge_attr, edge_idx, spatial_features = get_edge_features(all_node_features,
                                                                  num_frames= num_frames,
                                                                  num_players= num_nodes,
                                                                  k = self.k,
                                                                  return_splits= True)
        total_edges = edge_attr.shape[0]

        x_edges[:total_edges] = torch.tensor(edge_attr, dtype=torch.float32)
        edge_index[:total_edges] = torch.tensor(edge_idx, dtype=torch.int64)

        edge_mask[:total_edges] = True

        # ---- Output (Y) ----
        y = torch.zeros((self.max_output_frames, self.max_output_nodes, 2), dtype=torch.float32)
        output_mask = torch.zeros((self.max_output_frames, self.max_output_nodes), dtype=torch.bool)

        play_out = self._output_by_play[(game, play)]
        play_frames = sorted(play_out["frame_id"].unique())

        # Must use the last frame's node order
        node_order = node_ids

        for t, frame_id in enumerate(play_frames):
            if t >= self.max_output_frames:
                break

            frame_df = play_out[play_out["frame_id"] == frame_id].set_index("nfl_id")

            for n_idx, nfl_id in enumerate(node_order):
                if nfl_id in frame_df.index:
                    y[t, n_idx] = torch.tensor(frame_df.loc[nfl_id, ["x", "y"]].values, dtype=torch.float32)
                    output_mask[t, n_idx] = 1

        return {
            "x_nodes": x_nodes,           # [F * N, node_in_dim]
            "num_players": num_nodes,
            "x_edges": x_edges,           # [E_max, edge_in_dim]
            "edge_index": edge_index,     # [E_max, 2]
            "spatial_features": spatial_features,  # number of spatial edges (the rest are temporal)
            "edge_mask": edge_mask,       # [E_max]
            "x_global": global_feats,     # [F_global]
            "y": y,                       # [F_out, N, 2]
            "input_mask": input_mask,     # [F, N]
            "output_mask": output_mask    # [F_out, N]
        }


# ---------------------------------------------------------------- 7.3 PyG graphs and batches
def get_graph(batch, b):
    """PyG `Data` of the b-th play of a batch of padded tensors (as produced by the DataLoader): the padding is cut
    away with the masks, spatial and temporal edges are separated, target and mask travel with the graph."""
    x = torch.tensor(batch['x_nodes'][b], dtype=torch.float)
    edge_index = torch.tensor(batch['edge_index'][b], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(batch['x_edges'][b], dtype=torch.float)

    num_nodes = batch["input_mask"][b].sum()
    num_frames = batch["input_mask"][b, :, 0].sum()
    num_edges = batch["edge_mask"][b].sum()

    num_players = batch["num_players"][b]
    x_global = batch["x_global"][b].view(1, -1)   # [1, G] -> [B, G] after batching

    # Count output frames as "frames where ANY node is valid": indexing a
    # single node's column assumed the last input node is a predicted
    # player present in every output frame, which the node order (raw row
    # order of the last input frame) does not guarantee.
    num_output_frames = batch["output_mask"][b].any(dim=-1).sum()
    num_spatial = int(batch["spatial_features"][b])
    edge_index= edge_index[:, :num_edges]
    edge_attr= edge_attr[:num_edges]

    last_nodes = np.arange(num_nodes - num_players, num_nodes)
    predicted_nodes = batch["output_mask"][b, 0, :num_players].bool()

    return Data(x= x[:num_nodes],
                edge_index_spatial= edge_index[:, :num_spatial],     # only spatial edges
                edge_attr_spatial= edge_attr[:num_spatial],
                edge_index_temporal= edge_index[:, num_spatial:],    # only temporal edges
                edge_attr_temporal= edge_attr[num_spatial:],
                x_global= x_global,
                num_frames = num_frames,
                num_output_frames = num_output_frames,
                predicted_nodes = np.array(predicted_nodes),
                last_nodes= last_nodes,
                # y and its mask travel with the graph: the list of batches is self-contained and a parallel
                # DataLoader (with the risk of index misalignment) is not needed.
                # unsqueeze(0): Batch.from_data_list concatenates along dim 0 -> B tensors [1, T_out, N, 2]
                # become a real [B, T_out, N, 2] instead of gluing the frame axes of different plays together.
                y_target= batch["y"][b].unsqueeze(0),           # [1,T_out,N,2] -> [B,T_out,N,2]
                y_mask= batch["output_mask"][b].unsqueeze(0))   # [1,T_out,N] -> [B,T_out,N]


def get_gnn_batch(batch):
    """PyG `Batch` of all the plays of a DataLoader batch."""
    return Batch.from_data_list([get_graph(batch, b) for b in range(batch["x_nodes"].shape[0])])


# ---------------------------------------------------------------- Batch lists (one per batch size and global mode)
class BatchProvider:
    """Lists of PyG batches (train, val, test) for a given (batch_size, global_mode).

    The expensive step (padded tensors -> one PyG graph per play) depends only on `global_mode`, so it is done once per
    mode; changing the batch size only re-collates those graphs. Only the graphs of the last mode and the batch lists
    of the last combination are kept in memory: several full sets do not fit in a Colab session.

    Returns `global_in_dim` too, which depends on `global_mode` (0 for "none", different for "no_outcome" vs "all").
    """

    def __init__(self, data, splits, cfg, preprocess_play_fn, global_features_fn):
        self.data, self.splits, self.cfg = data, splits, cfg
        self.preprocess_play_fn = preprocess_play_fn
        self.global_features_fn = global_features_fn
        self._graphs = None      # (global_mode, {"train": [...], "val": [...], "test": [...]})
        self._batches = None     # ((batch_size, global_mode), result)

    def _play_graphs(self, global_mode):
        if self._graphs is not None and self._graphs[0] == global_mode:
            return self._graphs[1]
        self._graphs = self._batches = None
        gc.collect()

        graphs = {}
        for split, plays, name in (("train", self.splits.train_plays, "Train"),
                                   ("val", self.splits.val_plays, "Validation"),
                                   ("test", self.splits.test_plays, "Test")):
            in_split = self.data.input_df.set_index(["game_id", "play_id"]).index.isin(plays.index)
            ds = PaddedPlayDataset(self.data.input_df[in_split], self.data.output_df, self.preprocess_play_fn,
                                   self.global_features_fn, self.cfg, split=name, global_mode=global_mode)
            loader = DataLoader(ds, batch_size=1, shuffle=False)
            graphs[split] = [get_graph(b, 0) for b in tqdm(loader, desc=f"Graphs {name} ({global_mode})")]
            del ds, loader
            gc.collect()

        self._graphs = (global_mode, graphs)
        return graphs

    def get(self, batch_size, global_mode="all"):
        key = (batch_size, global_mode)
        if self._batches is not None and self._batches[0] == key:
            return self._batches[1]
        graphs = self._play_graphs(global_mode)
        self._batches = None
        gc.collect()

        lists = [[Batch.from_data_list(g[i:i + batch_size]) for i in range(0, len(g), batch_size)]
                 for g in (graphs["train"], graphs["val"], graphs["test"])]
        global_in_dim = int(lists[0][0].x_global.shape[-1]) if len(lists[0]) else 0

        self._batches = (key, (*lists, global_in_dim))
        return self._batches[1]
