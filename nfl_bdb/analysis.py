"""Exploratory analysis and diagnostics: data inspection, physique summaries, length bins and split checks.

Every function draws/prints what the corresponding notebook cell shows; the notebook only calls them.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------- 2.3 Data inspection
def plot_players_to_predict(input_df):
    """Share of plays for each number of players whose trajectory has to be predicted."""
    # Compute number of predicted players per play
    ret = (
        input_df[input_df["player_to_predict"]]
        .groupby(["game_id", "play_id"])["nfl_id"]
        .nunique()
        .reset_index(name="num_players_to_predict")
    )

    # Plot occurrences of each number of predicted players
    counts = ret["num_players_to_predict"].value_counts().sort_index()

    plt.figure(figsize=(8,5))
    (counts / counts.sum() * 100).plot(kind='bar')
    plt.ylabel("% of Plays")
    plt.xlabel("Number of Players to Predict per Play")
    plt.title("Distribution of player_to_predict per Play")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def frames_per_play(df, name):
    return (
        df.groupby(["game_id", "play_id"])["frame_id"]
        .max()
        .reset_index(name=name)
    )


def plot_frames_per_play(input_df, output_df):
    """Distribution of the number of pre-pass (input) and post-pass (output) frames per play, with the median."""
    frames_in = frames_per_play(input_df, "num_input_frames")
    frames_out = frames_per_play(output_df, "num_output_frames")
    panels = [
        (frames_in, "num_input_frames", "Input frames per play (pre-pass)", "#2a6fbb"),
        (frames_out, "num_output_frames", "Output frames per play (to predict)", "#d9822b"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    for ax, (df, col, title, color) in zip(axes, panels):
        counts = df[col].value_counts().sort_index()

        ax.bar(counts.index, counts.values, width=0.85, color=color)
        ax.axvline(df[col].median(), color="0.35", linestyle="--", linewidth=1.5)

        ax.set_xlim(0, counts.index.max() + 5)
        ax.set_xlabel(f"{col}")
        ax.set_ylabel("Number of plays")

        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 3.2 Position aggregation
def physique_summary(players_df):
    """Mean / std / min / max of the physical attributes for each position."""
    positions_df = players_df.drop(["player_name", "player_side"], axis= 1)

    summary_df = positions_df.groupby(by= "player_position").agg(["mean", "std", "min", "max"])
    return summary_df.fillna(0)


def plot_physique_kde(players_df):
    """Density of each physical attribute, one curve per position."""
    positions_df = players_df.drop(["player_name", "player_side"], axis= 1)
    numeric_cols = positions_df.drop("player_position", axis=1).columns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        sns.kdeplot(
            data=positions_df,
            x=col,
            hue="player_position",
            common_norm=True,
            fill=False,
            linewidth=2,
            ax=ax
        )
        ax.set_title(f"Distribution of {col} by Player Position")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 7.1-7.2 Length bins and split
def report_length_bins(play_lengths, cfg):
    """Sizes of the equal-frequency input/output length bins and of the joint bins.

    The marginal bins are near-identical by construction. The joint (input x output) bins can still be small and
    uneven since the two lengths are correlated: `uniform_sample_by_bin` corrects for that imbalance, not the bins.
    """
    input_bin_counts = play_lengths["input_len_bin"].value_counts().sort_index()
    output_bin_counts = play_lengths["output_len_bin"].value_counts().sort_index()
    joint_counts = play_lengths["length_bin"].value_counts()
    target_share = cfg.N_TOTAL / len(joint_counts)

    print("Input length bin sizes (rows per bin):")
    print(input_bin_counts)
    print("\nOutput length bin sizes (rows per bin):")
    print(output_bin_counts)
    print(f"\nJoint length_bin cells: {len(joint_counts)} "
          f"(size range: {joint_counts.min()}-{joint_counts.max()}, "
          f"{(joint_counts < cfg.N_TOTAL // len(joint_counts)).sum()} below the "
          f"{cfg.N_TOTAL // len(joint_counts)}/bin target share)")

    # Marginal bins: length bin (category) on the x-axis, play count on the y-axis.
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, counts, name, color in ((axes[0], input_bin_counts, "input", "tab:blue"),
                                    (axes[1], output_bin_counts, "output", "tab:orange")):
        ax.bar(range(len(counts)), counts.values, color=color)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(b) for b in counts.index], rotation=45, ha="right")
        ax.set_title(f"{name.capitalize()} length_bin sizes")
        ax.set_xlabel(f"{name}_len_bin")
        ax.set_ylabel("Number of plays")
    plt.tight_layout()
    plt.show()

    # Joint bins: too many categories for readable tick labels, so sort by count and show the target share.
    joint_counts_sorted = joint_counts.sort_values(ascending=False)
    plt.figure(figsize=(18, 5))
    plt.bar(range(len(joint_counts_sorted)), joint_counts_sorted.values, color="tab:green")
    plt.axhline(target_share, color="black", linestyle="--", linewidth=1,
                label=f"target share ({target_share:.0f}/bin)")
    plt.title("Joint length_bin sizes (input x output), sorted by count")
    plt.xlabel("length_bin (sorted by size; individual labels omitted, too many to show)")
    plt.ylabel("Number of plays")
    plt.legend()
    plt.tight_layout()
    plt.show()


def check_splits(splits):
    """No play shared between train/val/test, and comparable input/output length distributions."""
    train_idx = set(splits.train_plays.index)
    val_idx = set(splits.val_plays.index)
    test_idx = set(splits.test_plays.index)

    assert train_idx.isdisjoint(val_idx), "Train/Val overlap detected!"
    assert train_idx.isdisjoint(test_idx), "Train/Test overlap detected!"
    assert val_idx.isdisjoint(test_idx), "Val/Test overlap detected!"
    print(f"No (game_id, play_id) repeated across splits "
          f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

    split_labels = pd.concat([
        splits.train_plays.assign(split="train"),
        splits.val_plays.assign(split="val"),
        splits.test_plays.assign(split="test"),
    ])

    print("\nMean / std of frame lengths per split:")
    print(split_labels.groupby("split")[["num_input_frames", "num_output_frames"]].agg(["mean", "std"]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    for split_name, color in colors.items():
        subset = split_labels[split_labels["split"] == split_name]
        axes[0].hist(subset["num_input_frames"], bins=30, alpha=0.5, density=True, label=split_name, color=color)
        axes[1].hist(subset["num_output_frames"], bins=30, alpha=0.5, density=True, label=split_name, color=color)

    axes[0].set_title("Input frame-length distribution by split")
    axes[0].set_xlabel("num_input_frames")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].set_title("Output frame-length distribution by split")
    axes[1].set_xlabel("num_output_frames")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
