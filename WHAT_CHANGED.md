# New metrics only (this zip assumes the previous restructuring is already applied)

Only the two files that actually changed in this round -- `training.py` and `dataset.py` are unchanged since
the last zip, so they're not included here.

## `nfl_bdb/results.py`
Two new functions added (nothing removed or renamed):

- `plot_cost_accuracy(results, baseline=None)` -- scatter of every finished run's training time
  (`elapsed_min`) against its test RMSE, with `baseline` starred if given. Reads only
  `experiment_results.json`, no model needed.
- `error_by_horizon(model, gnn_batch_list, device)` / `plot_error_by_horizon(df, title=...)` -- breaks the
  aggregate test RMSE down by output frame (a forward pass over an existing batch list with an already-loaded
  model, no retraining). Frames with fewer valid (play, player) pairs are drawn lighter.

## `nfl_2026_bdb_script_final_complete.ipynb`
- Cell 5's `from nfl_bdb.results import (...)` line now also imports `plot_cost_accuracy`, `error_by_horizon`,
  `plot_error_by_horizon`.
- New code cell right after `plot_ablations(results, baseline=None)` in 11.1: `plot_cost_accuracy(results, baseline=None)`.
- New section at the end of the notebook, **11.4 Error vs. Prediction Horizon**: one markdown cell + one code
  cell (`horizon_df = error_by_horizon(model, test_batches, device)` then `plot_error_by_horizon(horizon_df)`).

## Verification
Full notebook re-run end-to-end against synthetic data: all cells pass, including the two new ones.
`plot_cost_accuracy` additionally checked against your real 12-row `experiment_results.json` -- the BASELINE
point sits right at the knee of the time/accuracy curve, matching the reasoning already in Chapter 10's
comments.
