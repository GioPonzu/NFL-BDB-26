"""Results of the experiments (notebook Chapter 11): comparison table, one plot per ablation, learning curves and
diagnostics of the global-feature branch.

`results` is the list of dicts kept by `ExperimentRunner` (or read back from its json). The BASELINE is the run every
ablation is compared with: `dict(batch_size=..., hidden_dim=..., global_mode=...)`. Each ablation varies ONE of the
three factors and keeps the other two at their baseline value.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# axis -> (column, title of the study, x-axis label)
STUDIES = {
    "batch_size": ("batch_size", "Batch size", "batch size"),
    "hidden_dim": ("hidden_dim", "Hidden dimension", "hidden dimension"),
    "global_mode": ("global_mode", "Global features", "global features"),
}
GLOBAL_MODE_ORDER = ["none", "no_outcome", "all"]
GLOBAL_MODE_NAMES = {"none": "none", "no_outcome": "no outcome", "all": "all"}
VAL_COLOR, TEST_COLOR, BASELINE_COLOR = "tab:blue", "tab:orange", "black"


def load_results(path):
    """Results saved by `ExperimentRunner` (list of dicts)."""
    with open(path) as f:
        return json.load(f)


def results_frame(results, baseline):
    """One row per run. Runs saved before `hidden_dim` existed are assumed to use the baseline hidden dimension."""
    df = pd.DataFrame(results)
    if "hidden_dim" not in df.columns:
        df["hidden_dim"] = baseline["hidden_dim"]
    df["hidden_dim"] = df["hidden_dim"].fillna(baseline["hidden_dim"]).astype(int)
    df["is_baseline"] = (
        (df["batch_size"] == baseline["batch_size"])
        & (df["hidden_dim"] == baseline["hidden_dim"])
        & (df["global_mode"] == baseline["global_mode"])
    )
    return df


def study_subset(df, axis, baseline):
    """Runs of one ablation: `axis` varies, the other two factors are at their baseline value (so the baseline run
    belongs to all three studies)."""
    keep = pd.Series(True, index=df.index)
    for other, (col, _, _) in STUDIES.items():
        if other != axis:
            keep &= df[col] == baseline[col]
    sub = df[keep].copy()
    col = STUDIES[axis][0]
    if axis == "global_mode":
        sub["_order"] = sub[col].map({m: i for i, m in enumerate(GLOBAL_MODE_ORDER)})
        return sub.sort_values("_order").drop(columns="_order")
    return sub.sort_values(col)


def config_label(value, axis):
    return GLOBAL_MODE_NAMES.get(value, value) if axis == "global_mode" else str(value)


def summary_table(results, baseline):
    """Table of all the configurations: study, value, validation and test RMSE (yards), difference from the baseline,
    epochs actually trained and training time."""
    df = results_frame(results, baseline)
    base = df[df["is_baseline"]]
    base_test = float(base["test_metric"].iloc[0]) if len(base) else np.nan

    rows = []
    for axis, (col, title, _) in STUDIES.items():
        for _, r in study_subset(df, axis, baseline).iterrows():
            rows.append({
                "study": title,
                "configuration": f"{config_label(r[col], axis)}" + (" (baseline)" if r["is_baseline"] else ""),
                "val RMSE": r["best_val_metric"],
                "test RMSE": r["test_metric"],
                "vs baseline": r["test_metric"] - base_test,
                "epochs": int(r["actual_epochs"]),
                "minutes": r["elapsed_min"],
            })
    table = pd.DataFrame(rows)
    return table


def show_summary_table(results, baseline):
    """`summary_table` formatted for display: best test RMSE of each study in bold."""
    table = summary_table(results, baseline)
    fmt = {"val RMSE": "{:.3f}", "test RMSE": "{:.3f}", "vs baseline": "{:+.3f}", "minutes": "{:.1f}"}
    try:
        def bold_best(col):
            best = col.groupby(table["study"]).transform("min") == col
            return ["font-weight: bold" if b else "" for b in best]
        return table.style.format(fmt).apply(bold_best, subset=["test RMSE"]).hide(axis="index")
    except Exception:            # jinja2 missing: plain table
        return table.round(3)


def print_takeaways(results, baseline):
    """One sentence per study: the best configuration on the TEST RMSE and its difference from the baseline."""
    df = results_frame(results, baseline)
    base = df[df["is_baseline"]]
    if base.empty:
        print("The baseline run is not among the results.")
        return
    base_test = float(base["test_metric"].iloc[0])
    print(f"Baseline (batch size {baseline['batch_size']}, hidden dim {baseline['hidden_dim']}, "
          f"global features '{baseline['global_mode']}'): test RMSE {base_test:.3f} yards\n")
    for axis, (col, title, _) in STUDIES.items():
        sub = study_subset(df, axis, baseline)
        if len(sub) < 2:
            print(f"{title}: fewer than two runs, no comparison.")
            continue
        best = sub.loc[sub["test_metric"].idxmin()]
        delta = best["test_metric"] - base_test
        tried = ", ".join(config_label(v, axis) for v in sub[col])
        if best["is_baseline"]:
            print(f"{title} (tried: {tried}): the baseline is the best configuration.")
        else:
            print(f"{title} (tried: {tried}): best is {config_label(best[col], axis)} with test RMSE "
                  f"{best['test_metric']:.3f} yards ({delta:+.3f} vs baseline, {100 * delta / base_test:+.1f}%).")


def plot_ablations(results, baseline, suptitle="Effect of each factor on the RMSE (yards)"):
    """Three panels (batch size, hidden dimension, global features): validation and test RMSE of every configuration.
    The baseline configuration is outlined; all the panels share the y axis, so the size of the effects is comparable."""
    df = results_frame(results, baseline)
    axes_present = [(a, study_subset(df, a, baseline)) for a in STUDIES]
    fig, axes = plt.subplots(1, len(axes_present), figsize=(5.2 * len(axes_present), 4.6), sharey=True)
    axes = np.atleast_1d(axes)

    lo = df[["best_val_metric", "test_metric"]].min().min()
    hi = df[["best_val_metric", "test_metric"]].max().max()
    pad = 0.15 * (hi - lo if hi > lo else hi)

    for ax, (axis, sub) in zip(axes, axes_present):
        title, xlabel = STUDIES[axis][1], STUDIES[axis][2]
        if sub.empty:
            ax.set_title(f"{title}: no runs")
            ax.axis("off")
            continue
        x = np.arange(len(sub))
        w = 0.38
        ax.bar(x - w / 2, sub["best_val_metric"], w, color=VAL_COLOR, label="Validation")
        ax.bar(x + w / 2, sub["test_metric"], w, color=TEST_COLOR, label="Test")
        for xi, v in zip(x, sub["best_val_metric"]):
            ax.text(xi - w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        for xi, v in zip(x, sub["test_metric"]):
            ax.text(xi + w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        for xi, is_base in zip(x, sub["is_baseline"]):
            if is_base:
                ax.add_patch(plt.Rectangle((xi - 0.5, 0), 1, 1, transform=ax.get_xaxis_transform(), fill=False,
                                           edgecolor=BASELINE_COLOR, linestyle="--", linewidth=1.2, zorder=0))
        ax.set_xticks(x)
        ax.set_xticklabels([config_label(v, axis) for v in sub[STUDIES[axis][0]]])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("RMSE (yards)")
    axes[0].set_ylim(max(0, lo - pad), hi + pad)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=BASELINE_COLOR, linestyle="--"))
    labels.append("Baseline")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    plt.show()


def plot_learning_curves(results, baseline, axis):
    """Validation RMSE per epoch of every run of one study (batch size, hidden dimension or global features);
    the baseline is drawn thicker. Shows not only WHO is best but how fast each configuration gets there."""
    df = results_frame(results, baseline)
    sub = study_subset(df, axis, baseline)
    col, title, _ = STUDIES[axis]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for _, r in sub.iterrows():
        hist = r["val_metric_history"]
        ax.plot(range(1, len(hist) + 1), hist, linewidth=2.8 if r["is_baseline"] else 1.5,
                color=BASELINE_COLOR if r["is_baseline"] else None,
                label=f"{config_label(r[col], axis)}" + (" (baseline)" if r["is_baseline"] else ""))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (yards)")
    ax.set_title(f"Validation RMSE per epoch - {title}")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(title=title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.show()


def plot_run(result):
    """Training / validation loss and learning rate of a single run."""
    n = result["actual_epochs"]
    epochs = range(1, n + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, result["train_loss_history"], label='Train loss', color='tab:blue')
    ax1.plot(epochs, result["val_loss_history"], label='Val loss', color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    ax1.set_title(f"Training / validation loss - {result['run_label']}")

    ax2 = ax1.twinx()
    ax2.plot(epochs, result["lr_history"], label='LR', color='tab:green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Learning rate')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.show()
    print(f"Best val loss {result['best_val_loss']:.4f} | val RMSE {result['best_val_metric']:.4f} | "
          f"test RMSE {result['test_metric']:.4f}")


# ---------------------------------------------------------------- Global-feature branch diagnostics
def global_feature_names(data, ball_stats_fn):
    """Names of the columns of the global vector for global_mode="all", in their real order (same merge as the
    notebook's `get_global_features`, stopped before `.to_numpy()`)."""
    from .data import get_play
    game_id, play_id = data.input_df[["game_id", "play_id"]].drop_duplicates().iloc[0]
    play_df, _ = get_play(data, game_id, play_id)
    ball_df = pd.DataFrame(ball_stats_fn(play_df, include_outcome=True))
    more_df = data.aux_df_preprocessed[
        (data.aux_df_preprocessed["game_id"] == game_id) & (data.aux_df_preprocessed["play_id"] == play_id)
    ]
    row = pd.merge(ball_df, more_df, on=["game_id", "play_id"], how="inner")
    row = row.groupby(["game_id", "play_id"]).first().reset_index()
    return row.drop(columns=["game_id", "play_id"]).columns.tolist()


@torch.no_grad()
def global_branch_diagnostics(model, x_global_train, sample_batch, feature_names, device, top=15):
    """How much does the global branch weigh? Per feature: scale (std over the train set), mean absolute weight of
    `global_proj` and their product (effective contribution); then the size of the global embedding relative to the
    spatio-temporal one. A feature with a huge scale can dominate the input even with a small weight, and if
    ||global_embed|| << ||node_embed|| the global features are diluted whatever they contain.

    x_global_train: [n_plays, G] global vectors of the train set; sample_batch: one PyG batch of the train set.
    """
    assert len(feature_names) == model.global_proj.in_features, \
        "model.global_proj does not match global_mode='all': pass the model of the run with all the global features"

    x_global = x_global_train.detach().cpu()
    stats = pd.DataFrame({
        "feature": feature_names,
        "mean": x_global.mean(dim=0).numpy(),
        "std": x_global.std(dim=0).numpy(),
        "min": x_global.min(dim=0).values.numpy(),
        "max": x_global.max(dim=0).values.numpy(),
    })
    w = model.global_proj.weight.detach().cpu()  # [hidden_dim - 2, global_in_dim]
    stats["mean_abs_weight"] = w.abs().mean(dim=0).numpy()
    stats["effective_contribution"] = stats["mean_abs_weight"] * stats["std"]
    stats = stats.sort_values("effective_contribution", ascending=False)

    sample = sample_batch.to(device)
    model.eval()
    node_embed = model.encoder(sample.x, sample.edge_index_spatial, sample.edge_attr_spatial,
                               sample.edge_index_temporal, sample.edge_attr_temporal)
    global_embed = model.global_proj(sample.x_global)
    node_norm = node_embed.norm(dim=-1).mean().item()
    global_norm = global_embed.norm(dim=-1).mean().item()

    print(stats.head(top).to_string(index=False))
    print(f"\nmean ||node_embed|| per row:   {node_norm:.4f}")
    print(f"mean ||global_embed|| per row: {global_norm:.4f}")
    print(f"ratio ||global_embed|| / ||node_embed||: {global_norm / node_norm:.4f}")
    return stats
