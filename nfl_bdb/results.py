"""Results of the experiments (notebook Chapter 11): comparison table, one plot per ablation, learning curves and
diagnostics of the global-feature branch.

`results` is the list of dicts kept by `ExperimentRunner` (or read back from its json, or several of its jsons --
see `load_all_results`). Every function below now takes `baseline=None` as a valid input, not only an explicit
`dict(batch_size=..., hidden_dim=..., global_mode=..., gat_num_layers=..., regressor_hidden_dim=...)`:

- `baseline` given (a fully-resolved `BASELINE`, e.g. after 10.6): every study is filtered against THAT one
  configuration, exactly as before -- this is the authoritative final report.
- `baseline=None`: each study is instead read against its OWN local reference point, auto-detected from the data
  itself (`local_baseline`) as whichever configuration was actually held fixed while that study's axis was varied.
  This is what lets the table/bar-charts/learning-curves be built at ANY point during the sequential search --
  complete or not, and from `experiment_results.json` file(s) alone, no model, no GPU, no re-running Chapters 1-9 --
  which matters because the search is sequential (10.1-10.5): the global-features runs were made at `batch_size=8`,
  the hidden-dimension runs at whatever `batch_size` 10.2 had settled on, and so on, so a single FINAL baseline
  can't correctly filter the EARLIER stages (see 11.1's markdown in the notebook).

Runs collected across more than one Colab session commonly repeat a label (same config, run again after a
disconnect, or just re-run to sanity-check) -- `load_all_results` keeps every one of them as its own row (tagged by
`source`), rather than merging, averaging, or silently keeping only the last one. `results_frame`'s `dup` column
flags which labels this happened to; the plotting functions append a `(source)` tag to a configuration's label only
when it is actually duplicated, so single runs stay uncluttered.
"""
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# axis -> (column, title of the study, x-axis label), in the order the sequential search actually ran them:
# global features, then batch size, then hidden dimension, then regressor width, then GAT layers -- each stage's
# runs were made with every earlier stage's winner already fixed, regressor width included, so the GAT-layers
# study is read with the regressor width already resolved, not the other way around. This order only controls
# display (panel order in plot_ablations, row order in summary_table); local_baseline's auto-detection groups by
# the OTHER columns (all of STUDIES except the one axis in question) regardless of dict order, so it isn't
# affected by this ordering, and it isn't affected by adding a new axis either.
#
# "n_total" is NOT a stage of Chapter 10's sequential search -- it's Chapter 12's sampled-pool-size checks
# (12.1/12.2), which retrain one fixed configuration on bigger data instead of varying a model hyperparameter.
# It is tracked here as a sixth axis anyway, for one specific reason: without it, those checks are otherwise
# indistinguishable, on every OTHER axis, from the pre-tuning starting-point configuration already in the table
# (batch_size=1, hidden_dim=64, global_mode="all", gat_num_layers=1, regressor_hidden_dim=256) -- so
# local_baseline's auto-grouping would silently fold them into the batch_size/hidden_dim/regressor_width/
# gat_num_layers panels as fake extra points of an unrelated study. Giving "sampled pool size" its own axis (a)
# gives it its own clean panel and (b) makes it part of "the other columns held fixed" for every other study,
# which is what correctly excludes these rows from those studies' groupings.
STUDIES = {
    "global_mode": ("global_mode", "Global features", "global features"),
    "batch_size": ("batch_size", "Batch size", "batch size"),
    "hidden_dim": ("hidden_dim", "Hidden dimension", "hidden dimension"),
    "regressor_hidden_dim": ("regressor_hidden_dim", "Regressor width", "regressor hidden dim"),
    "gat_num_layers": ("gat_num_layers", "GAT layers", "GAT layers"),
    "n_total": ("n_total", "Sampled pool size", "sampled pool size"),
}
GLOBAL_MODE_ORDER = ["none", "all"]
GLOBAL_MODE_NAMES = {"none": "none", "all": "all"}
# N_TRAIN+N_VAL+N_TEST (3000/1000/1000) used throughout Chapters 9-10, before "n_total" existed as a stored
# field -- results_frame() fills it in with this for any run saved before that field was added, so old rows in
# experiment_results.json don't need to be touched by hand.
DEFAULT_N_TOTAL = 5000
VAL_COLOR, TEST_COLOR, BASELINE_COLOR = "tab:blue", "tab:orange", "black"


def _darken(color, factor=0.6):
    """A darker shade of `color` (each RGB channel scaled towards black by `factor`) -- used to mark the best
    configuration of a study directly on its bar, in `plot_ablations`, instead of a separate outline/legend
    mechanism."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


VAL_COLOR_BEST, TEST_COLOR_BEST = _darken(VAL_COLOR), _darken(TEST_COLOR)


# ---------------------------------------------------------------- Loading (one or several result files)
def load_results(path):
    """Results saved by `ExperimentRunner` (list of dicts) -- a single json file."""
    with open(path) as f:
        return json.load(f)


def load_all_results(sources):
    """Loads and concatenates one or more `experiment_results.json` files -- e.g. one per Colab session -- WITHOUT
    merging, averaging or dropping duplicate run_labels: every record is kept as its own row, tagged with which
    file it came from (`source`). This is the normal case once a study has been run more than once (a disconnect
    mid-search, or an intentional re-run): see the module docstring.

    `sources`: a single path, a list of paths (each tagged by its filename), or a {tag: path} dict for custom
    source names, e.g. `{"session 1": "outputs/experiment_results.json", "session 2": "outputs/experiment_results_session2.json"}`.
    Missing files are skipped with a printed warning rather than raising, so a partially-collected set of sessions
    still loads.
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]
    if not isinstance(sources, dict):
        sources = {Path(p).stem: p for p in sources}

    combined = []
    for tag, path in sources.items():
        path = Path(path)
        if not path.exists():
            print(f"[load_all_results] {path} not found, skipped")
            continue
        for r in load_results(path):
            r = dict(r)
            r.setdefault("source", tag)
            combined.append(r)
    return combined


# ---------------------------------------------------------------- Frame, local baselines, subsets
def results_frame(results, baseline=None):
    """One row per run. Runs saved before a column existed (e.g. `hidden_dim` in older results, or `gat_num_layers`
    / `regressor_hidden_dim` before they became ablation axes) fall back to `baseline`'s value of that column when
    a `baseline` is given; with `baseline=None` they are left as-is (NaN if truly missing -- harmless for current
    results, which always carry all five fields).
    """
    df = pd.DataFrame(results)
    if "source" not in df.columns:
        df["source"] = ""
    df["source"] = df["source"].fillna("")

    fill_from = baseline or {}
    for col in ("hidden_dim", "gat_num_layers", "regressor_hidden_dim"):
        if col not in df.columns:
            df[col] = fill_from.get(col, np.nan)
        if col in fill_from:
            df[col] = df[col].fillna(fill_from[col])
        if not df[col].isna().any():
            df[col] = df[col].astype(int)

    # "n_total" (sampled pool size) only exists on runs saved after Chapter 12's split-size checks were added --
    # every run before that (the whole Chapter 10 sequential search) trained on N_TRAIN+N_VAL+N_TEST=5000, so
    # that's the correct backfill, not NaN or the baseline dict (which doesn't carry this field).
    if "n_total" not in df.columns:
        df["n_total"] = DEFAULT_N_TOTAL
    else:
        df["n_total"] = df["n_total"].fillna(DEFAULT_N_TOTAL).astype(int)

    df["dup"] = df["run_label"].duplicated(keep=False) if "run_label" in df.columns else False

    if baseline is not None:
        df["is_baseline"] = (
            (df["batch_size"] == baseline["batch_size"])
            & (df["hidden_dim"] == baseline["hidden_dim"])
            & (df["global_mode"] == baseline["global_mode"])
            & (df["gat_num_layers"] == baseline["gat_num_layers"])
            & (df["regressor_hidden_dim"] == baseline["regressor_hidden_dim"])
            # BASELINE (the hyperparameter dict) doesn't carry a pool-size field, since it predates Chapter 12 --
            # default it to DEFAULT_N_TOTAL so "the baseline run" still means the one true final model (trained
            # at the standard 5k pool), not also any future rerun of it at a bigger sampled pool.
            & (df["n_total"] == baseline.get("n_total", DEFAULT_N_TOTAL))
        )
    else:
        df["is_baseline"] = False
    return df


def local_baseline(df, axis):
    """Auto-detects the configuration that was actually held fixed while `axis` was varied, directly from the
    data: groups the runs by every OTHER column in STUDIES (five of them, currently), and returns the values of
    the group where `axis` takes more than one distinct value, preferring the group with the most runs (ties
    broken by total epochs trained, then by the group encountered first). Returns None if `axis` was never varied
    in `results` (that study hasn't run yet). This is what `study_subset`/the plotting functions fall back to
    when `baseline=None`.
    """
    other_cols = [STUDIES[a][0] for a in STUDIES if a != axis]
    col = STUDIES[axis][0]
    best_group, best_score = None, None
    for key, g in df.groupby(other_cols, dropna=False):
        if g[col].nunique() > 1:
            score = (len(g), g["actual_epochs"].sum() if "actual_epochs" in g else 0)
            if best_score is None or score > best_score:
                best_score, best_group = score, key
    if best_group is None:
        return None
    if len(other_cols) == 1:
        best_group = (best_group,)
    return dict(zip(other_cols, best_group))


def study_subset(df, axis, baseline=None):
    """Runs of one ablation: `axis` varies, the other five factors are held at `baseline`'s value if given, or at
    the auto-detected `local_baseline` otherwise (so the baseline run, when there is one, belongs to every study it
    is part of)."""
    fixed = baseline if baseline is not None else local_baseline(df, axis)
    if fixed is None:
        return df.iloc[0:0]
    if baseline is not None and "n_total" not in fixed:
        # Same reasoning as is_baseline above: an explicit baseline dict predates pool-size tracking, so treat it
        # as implicitly pinned to the standard 5k pool unless the caller says otherwise -- otherwise a future
        # bigger-pool rerun of the exact tuned baseline would leak into every OTHER study's panel too.
        fixed = {**fixed, "n_total": DEFAULT_N_TOTAL}
    keep = pd.Series(True, index=df.index)
    for other, (col, _, _) in STUDIES.items():
        if other != axis and col in fixed:
            keep &= df[col] == fixed[col]
    sub = df[keep].copy()
    col = STUDIES[axis][0]
    if axis == "global_mode":
        sub["_order"] = sub[col].map({m: i for i, m in enumerate(GLOBAL_MODE_ORDER)})
        return sub.sort_values(["_order", "source"]).drop(columns="_order")
    return sub.sort_values([col, "source"])


def config_label(value, axis):
    if axis == "global_mode":
        return GLOBAL_MODE_NAMES.get(value, value)
    if axis == "n_total":
        # 5000/10000/14000 -> "5k"/"10k"/"14k"; g-format drops a trailing ".0" for exact thousands but still
        # shows one decimal for anything that isn't (e.g. a future 12500 -> "12.5k"), so this doesn't silently
        # mislabel an odd value if one is ever added.
        return f"{value / 1000:g}k"
    return str(value)


def disambiguate_labels(sub, axis):
    """One display label per row of `sub`: the configuration value, plus a `(source)` tag ONLY for values that
    appear more than once in this subset -- e.g. the same config trained in two different sessions -- so a single
    run's label stays clean and only genuine duplicates get tagged."""
    col = STUDIES[axis][0]
    counts = sub[col].value_counts()
    labels = []
    for _, r in sub.iterrows():
        label = config_label(r[col], axis)
        if counts[r[col]] > 1 and r["source"]:
            label += f" ({r['source']})"
        labels.append(label)
    return labels


# ---------------------------------------------------------------- Summary table
def summary_table(results, baseline=None):
    """Table of all the configurations: study, value, validation and test RMSE (yards), difference from either the
    given `baseline` or (if none given) the best run of that same study, epochs actually trained, training time and
    source. A study with fewer than one point (never run) is skipped.

    The row index is "study: value" (e.g. "Batch size: 64", "Global features: all (baseline)") rather than a bare
    0, 1, 2, ... -- so the value being compared is legible even if this DataFrame is displayed directly (not just
    through `show_summary_table`, whose styled path already hides the index). The study name is part of the index,
    not just the value, because the same value (e.g. "64") recurs across different studies -- a bare value would
    make the index non-unique, which `Styler.apply`/`.map` (used by `show_summary_table`) refuses outright."""
    df = results_frame(results, baseline)
    delta_col = "vs baseline" if baseline is not None else "vs best (this study)"
    base_test = None
    if baseline is not None:
        base = df[df["is_baseline"]]
        base_test = float(base["test_metric"].iloc[0]) if len(base) else np.nan

    rows = []
    for axis, (col, title, _) in STUDIES.items():
        sub = study_subset(df, axis, baseline)
        if sub.empty:
            continue
        ref_test = base_test if baseline is not None else sub["test_metric"].min()
        for label, (_, r) in zip(disambiguate_labels(sub, axis), sub.iterrows()):
            rows.append({
                "study": title,
                "configuration": label + (" (baseline)" if r["is_baseline"] else ""),
                #"source": r["source"],
                "val RMSE": r["best_val_metric"],
                "test RMSE": r["test_metric"],
                delta_col: r["test_metric"] - ref_test,
                "epochs": int(r["actual_epochs"]),
                "minutes": r["elapsed_min"],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out.index = pd.Index(out["study"] + ": " + out["configuration"], name=None)
    return out


def show_summary_table(results, baseline=None):
    """`summary_table` formatted for display: best test RMSE of each study in bold."""
    table = summary_table(results, baseline)
    if table.empty:
        print("No runs to show yet.")
        return table
    delta_col = "vs baseline" if "vs baseline" in table.columns else "vs best (this study)"
    fmt = {"val RMSE": "{:.3f}", "test RMSE": "{:.3f}", delta_col: "{:+.3f}", "minutes": "{:.1f}"}
    try:
        def bold_best(col):
            best = col.groupby(table["study"]).transform("min") == col
            return ["font-weight: bold" if b else "" for b in best]
        return table.style.format(fmt).apply(bold_best, subset=["test RMSE"]).hide(axis="index")
    except Exception:            # jinja2 missing: plain table
        return table.round(3)


# ---------------------------------------------------------------- Bar charts and learning curves
def plot_ablations(results, baseline=None, suptitle=None):
    """One panel per study: validation and test RMSE of every configuration in it. The best configuration of each
    panel -- the same one `summary_table`/`show_summary_table` bolds, `sub["test_metric"].idxmin()` -- is drawn
    in a darker shade of the same color, instead of being singled out with a separate outline; there's no
    baseline marker any more. "Best" is a property of each panel on its own: with an explicit `baseline` that
    happens not to be the best of a given panel (e.g. Chapter 12's checks, which retrain the Chapter 9
    starting-point configuration rather than the tuned `BASELINE`), it's that panel's actual best bar that's
    darkened, not the baseline's. Panels share the y axis, so the size of the effects is comparable."""
    df = results_frame(results, baseline)
    axes_present = [(a, study_subset(df, a, baseline)) for a in STUDIES]

    # A grid (at most 3 per row) instead of one ever-wider row: with 6 studies, a single row of 6 panels squeezes
    # each one down to where labels and value annotations start to overlap. Unused slots in the last row are hidden.
    n = len(axes_present)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.6 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for extra_ax in axes[n:]:
        extra_ax.axis("off")

    non_empty = [s for _, s in axes_present if not s.empty]
    if non_empty:
        allvals = pd.concat([s[["best_val_metric", "test_metric"]] for s in non_empty])
        lo, hi = allvals.min().min(), allvals.max().max()
        pad = 0.15 * (hi - lo if hi > lo else max(hi, 1e-6))
    else:
        lo, hi, pad = 0, 1, 0.1

    for ax, (axis, sub) in zip(axes, axes_present):
        title, xlabel = STUDIES[axis][1], STUDIES[axis][2]
        if sub.empty:
            ax.set_title(f"{title}: no runs")
            ax.axis("off")
            continue
        labels = disambiguate_labels(sub, axis)
        x = np.arange(len(sub))
        w = 0.38
        # The same "best" the table/its bold test-RMSE column uses: lowest test_metric in THIS panel's subset --
        # not tied to `is_baseline`, so an explicit `baseline` that isn't actually the best of a given study (e.g.
        # 12.1/12.2's gl=1 baseline vs. a gl=2/3 point in the GAT-layers panel) is no longer misleadingly marked.
        best_pos = int(np.argmin(sub["test_metric"].values))
        val_colors = [VAL_COLOR_BEST if i == best_pos else VAL_COLOR for i in range(len(sub))]
        test_colors = [TEST_COLOR_BEST if i == best_pos else TEST_COLOR for i in range(len(sub))]
        ax.bar(x - w / 2, sub["best_val_metric"], w, color=val_colors, label="Validation")
        ax.bar(x + w / 2, sub["test_metric"], w, color=test_colors, label="Test")
        for xi, v in zip(x, sub["best_val_metric"]):
            ax.text(xi - w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        for xi, v in zip(x, sub["test_metric"]):
            ax.text(xi + w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12 if any(len(l) > 6 for l in labels) else 0, ha="right", fontsize=8.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # sharey means every panel already has the same scale; only the leftmost column needs the tick labels and
    # axis title repeated, which is what actually made the row feel condensed with 5+ panels.
    for i, ax in enumerate(axes[:n]):
        if i % ncols == 0:
            ax.set_ylabel("RMSE (yards)")
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].set_ylim(max(0, lo - pad), hi + pad)
    handles, labels_ = axes[0].get_legend_handles_labels()
    handles += [plt.Rectangle((0, 0), 1, 1, color=VAL_COLOR_BEST), plt.Rectangle((0, 0), 1, 1, color=TEST_COLOR_BEST)]
    labels_ += ["Validation (best)", "Test (best)"]
    fig.legend(handles, labels_, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(suptitle or "Effect of each factor on the RMSE (yards) -- darker bars mark each panel's best "
                             "(lowest test RMSE)"
                 + ("" if baseline is not None else "; read at each study's own point in the search"))
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    plt.show()


def plot_learning_curves(results, baseline, axis):
    """Validation RMSE per epoch of every run of one study; the reference run (the given `baseline`, or the
    auto-detected local one if `baseline=None`) is drawn thicker. Shows not only WHO is best but how fast each
    configuration gets there."""
    df = results_frame(results, baseline)
    sub = study_subset(df, axis, baseline)
    col, title, _ = STUDIES[axis]
    if sub.empty:
        print(f"{title}: no runs to plot.")
        return

    fixed = baseline if baseline is not None else local_baseline(df, axis)
    is_ref = sub["is_baseline"] if baseline is not None else (sub[col] == fixed.get(col, object()))
    labels = disambiguate_labels(sub, axis)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for label, (_, r), ref in zip(labels, sub.iterrows(), is_ref):
        hist = r["val_metric_history"]
        ax.plot(range(1, len(hist) + 1), hist, linewidth=2.8 if ref else 1.5,
                color=BASELINE_COLOR if ref else None, label=label + (" (baseline)" if ref else ""))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (yards)")
    ax.set_title(f"Validation RMSE per epoch - {title}")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(title=title, fontsize=8)
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
    notebook's `get_global_features`, stopped before `.to_numpy()`). NOTE: unlike everything above, this and
    `global_branch_diagnostics` need the real model and real data (Chapters 1-9 executed, and a reloaded
    checkpoint) -- they can't be computed from `experiment_results.json` alone."""
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
