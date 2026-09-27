# Chapter 12 as a real sixth ablation — what changed and what to do

This replaces the previous `NFL-BDB-26-chapter12-fix.zip` (isolated `outputs_split10k` /
`outputs_split14k` / `weights_split10k` / `weights_split14k` folders). That approach is dropped —
everything now lives in the same `outputs/experiment_results.json` and `weights/` folder as the rest
of the notebook, and the two split-size checks are tracked as a genuine sixth ablation axis instead.

## What's in this zip

- `nfl_bdb/results.py` — replace the one in your repo. Adds `"n_total"` ("Sampled pool size") as a
  sixth entry in `STUDIES`, alongside the five from Chapter 10. Every function that reads `STUDIES`
  (`local_baseline`, `study_subset`, `summary_table`, `show_summary_table`, `print_takeaways`,
  `plot_ablations`, `plot_learning_curves`) already worked generically off that dict, so this one
  addition is what makes 12.1/12.2 show up as their own clean panel everywhere, instead of leaking
  into the batch-size / hidden-dim / regressor-width / GAT-layers panels as fake duplicate points.
  Runs saved before this field existed (all of Chapter 10) are backfilled to the standard 5000
  automatically — no need to touch their JSON entries by hand.

  Note on the name: I called the field `n_total` rather than `n_plays`, because `n_plays` is already
  used elsewhere in your code (`show_predictions(..., n_plays=20)` in 11.4) to mean "how many example
  plays to plot" — a completely different thing from "how many plays were sampled for train/val/test".
  Using the same name for both would have been confusing every time you read the notebook.

- `nfl_bdb/training.py` — replace the one in your repo. `ExperimentRunner.run()` now stamps every
  result with `n_total = cfg.N_TRAIN + cfg.N_VAL + cfg.N_TEST` (`cfg.N_TOTAL`, which already existed).
  This is what makes future runs self-describing, so nothing needs to be patched by hand again.

- `nfl_2026_bdb_script_final.ipynb` — Chapter 12 rewritten: `cfg_10k`/`cfg_14k` no longer override
  `OUTPUT_DIR`/`WEIGHTS_DIR` — they inherit the shared ones. `.load_results()` before training and the
  explicit unique `run_label`s (`..._10k`, `..._full`) are what still make this safe to share a folder
  with everything else (both were already true before; only the folder isolation is gone). I also
  fixed a stale line in the Chapter 10 intro that still mentioned a dropped "aggressive stress test"
  from an earlier plan.

- `outputs/experiment_results.json` — your real, current file, unchanged except for one thing: the two
  already-trained split-check rows (`bs1_hd64_gl1_rg256_all_10k`, `..._full`) now carry
  `"n_total": 10000` / `"n_total": 14000`. I diffed this against your committed file: every other field
  of every row is byte-for-byte identical, nothing else moved.

No weights files are included — the ones already committed
(`best_model_bs1_hd64_gl1_rg256_all_10k.pt`, `..._full.pt`) don't need to change; only the JSON needed
the new field.

## To apply

1. In your repo (`no_run_plots` branch): replace `nfl_bdb/results.py`, `nfl_bdb/training.py`,
   `outputs/experiment_results.json`, and `nfl_2026_bdb_script_final.ipynb` with the versions here.
2. Delete `outputs_split10k/`, `outputs_split14k/`, `weights_split10k/`, `weights_split14k/` if you'd
   created them from the previous zip — they're no longer used by anything.
3. Nothing needs to be retrained. Re-running Chapter 12's cells will just reload the two finished
   results from the shared file, as before.

## Verification performed

- The full notebook (with the new Chapter 12 cells) was executed end-to-end against synthetic data:
  all cells ran clean, the shared results file correctly grew 10 → 11 → 12 rows across the two split
  checks, and `n_total` was correctly stamped on every fresh run.
- `nfl_bdb.results`'s real functions were run against your actual 12-row `experiment_results.json`:
  `show_summary_table` no longer raises the `Styler` non-unique-index error, and `plot_ablations`
  produces a clean 3×2 grid — five familiar panels with exactly their real values (no duplicate bars)
  plus a new "Sampled pool size" panel (5k / 10k / 14k, RMSE improving with more data, as expected).
