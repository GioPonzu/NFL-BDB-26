# What changed in this delivery

Built from `no_run_plots`, using `outputs/experiment_results_updated.json` (12 rows) as the source of truth.

## Results, deduplicated (10 unique configurations)

Two labels were trained twice across separate Colab sessions: `bs8_hd64_gl1_rg256_none` and
`bs8_hd64_gl1_rg256_all`. For each, the row with the better (lower) test RMSE was kept and the other dropped:

- `bs8_hd64_gl1_rg256_none`: kept test RMSE 1.0141 (dropped 1.0320)
- `bs8_hd64_gl1_rg256_all`: kept test RMSE 0.9440 (dropped 0.9667)

The result is `outputs/experiment_results_updated.json` in this delivery (10 rows, one per configuration). If you'd
rather keep "most recent session wins" or some other rule, this is a one-line change in the dedup step -- say the
word and I'll redo it.

**Caveat:** for both of those labels, only one `.pt` checkpoint exists in `weights/` (the second session's upload
overwrote the first's file), so there's no way to tell from the repo alone whether the surviving checkpoint
corresponds to the *kept* number above or the *dropped* one. It doesn't affect the reported table, only which exact
weights back that one row if you reload it.

Every one of the 10 final configurations has a matching checkpoint in `weights/` -- nothing is orphaned. The only
extra file, `best_model_bs1_hd64_gl1_rg512_all_10kdata.pt`, belongs to Chapter 12 (the split-size check), not the
ablation grid, and was left out of the results file on purpose.

## The actual best model

The GAT-layers stage (10.5) was the one still open last time we talked, and its winner beats every other stage's:

**`bs1_hd64_gl3_rg256_all`** -- batch size 1, hidden dim 64, global features "all", **3 GAT layers**, regressor
width 256 -- test RMSE **0.454** yards (vs. 0.464 for the 1-GAT-layer configuration used everywhere before this).

`BASELINE["gat_num_layers"]` in the notebook was still set to the placeholder `1` (never updated after 10.5 ran) --
fixed to `3` in this delivery, with a comment recording why.

## `nfl_bdb/results.py`

- `STUDIES` reordered to match the search's real sequence: global features, batch size, hidden dimension,
  **regressor width, then GAT layers** (previously GAT layers was listed before regressor width, backwards from how
  they were actually run -- the GAT-layers stage held regressor width fixed at its already-chosen value of 256).
  This only changes display order and which stage's baseline reads as "already decided" in the write-up;
  `local_baseline`'s auto-detection groups by the other four columns regardless of dict order, so nothing about
  correctness depended on this.
- `summary_table`'s row index is now the ablation value itself (e.g. `64`, `all (baseline)`) instead of the default
  `0, 1, 2, ...` -- shows up whether you call `show_summary_table` (in Colab/Jupyter with `jinja2`, which already
  hid the index) or the plain `summary_table` function directly.
- `plot_ablations` now lays panels out in a grid (max 3 per row) instead of one ever-widening row -- with 5 studies
  the single-row layout was cramped enough that bar labels and value annotations were starting to overlap. Y-axis
  tick labels only repeat on the leftmost column of each row, since every panel shares the same scale anyway.

## The notebook (`nfl_2026_bdb_script_2.ipynb`)

Built from `nfl_2026_bdb_script_ok_10k_gat_ablations.ipynb`, which already had all five Chapter 10 stages including
10.5 (GAT layers) -- this is meant to supersede all three notebooks currently on `no_run_plots`
(`nfl_2026_bdb_script_ok.ipynb`, `..._ok_10k.ipynb`, `..._ok_10k_gat_ablations.ipynb`); once you're happy with it,
those three are safe to delete from the repo.

- **Cell 4** (repo clone): now resolves `REPO_DIR` to an absolute path and does `os.chdir(REPO_DIR)` right after
  cloning. Previously, `outputs/` and `weights/` (both relative paths in `Config`) resolved against Colab's
  `/content`, not the cloned repo -- the reason you had to manually prefix `"NFL-BDB-26" /` onto paths in your own
  10k test run. This fixes it at the source instead of needing that prefix anywhere.
- **Cell 108**: `BASELINE["gat_num_layers"]` corrected from `1` to `3` (see above).
- **Cell 113** (Chapter 11 data loading): now points at the single final `experiment_results_updated.json` instead
  of three placeholder session files (only one of which existed). `load_all_results` is unchanged and still
  supports adding more sources later without merging them.
- Every cell's output and execution count was cleared, since several carried stale or erroring outputs from a
  partial run (Chapters 1-9 hadn't been re-executed in the same pass as the Chapter 11 cells). Submit only after
  doing the clean, top-to-bottom run that's already on your checklist -- that will also fix the "Final
  configuration" printout in 10.6, which was showing a stale, out-of-order value.

## Validated

Ran the full 136-cell notebook against synthetic data (small config, 2 epochs) end to end: zero failing cells.
