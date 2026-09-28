# Chapter 10/11/12 restructure + no_outcome removal

## What's in this zip
- `nfl_2026_bdb_script_final_complete.ipynb` — restructured notebook (130 cells).
- `nfl_bdb/results.py`, `nfl_bdb/training.py`, `nfl_bdb/dataset.py` — three stray "Chapter 12" doc-comment
  mentions updated to match the new structure; your own `run_stage()` edit (suppressed print, indexed
  return) is preserved as-is.

## Restructuring
- Chapter 12 ("Two More Points at Larger Splits") is gone as a standalone chapter. Its two split-size
  checks are now **10.6 Sampled Pool Size**, the last stage of Chapter 10, right before assembly.
- The separate `SPLIT_CHECK_CONFIG` dict is gone too: since `gat_num_layers=1` is confirmed as `BASELINE`'s
  real final value, it was byte-for-byte identical to `BASELINE` already, so 10.6 now just uses `**BASELINE`
  directly. The "5k" reference point is fetched with
  `runner_10k.get_result(ExperimentRunner.make_label(**BASELINE))` — the exact row already trained during
  10.5's own grid — instead of the earlier `baseline`/`final` variables (which would've pointed at two
  different objects depending on where in the notebook you were).
- "10.6 Assembling the Final Model" is renumbered **10.7**; Chapter 11 is renumbered 11.0→**11.1**,
  11.4→**11.2**, 11.5→**11.3** (11.1 now covers the table, learning curves and ablation plot together,
  matching your own earlier simplification of that section). All cross-references (10.6→10.7, "11.1-11.3
  / 11.4-11.5" language, etc.) are updated to match.
- Fixed cell 110's comment, which contradicted itself about why `gat_num_layers=1` was picked: it now says
  plainly that it's despite gl=3's lower test RMSE (0.454 vs 0.464), because that's not worth ~2x the
  training time for ~0.01 yards.

## no_outcome removal
Every trace of the `global_mode="no_outcome"` option is gone: the branch and the now-dead
`AUX_OUTCOME_COLS_EXACT`/`PREFIX` constants in `get_global_features` (5.1/6.3), and the markdown mentions in
5.1, 6.3, 8.4. `global_mode` is only ever `"none"` or `"all"` now, everywhere.

## Verification
Ran the full notebook end-to-end against synthetic data (small config) cell by cell, twice (once right
after the restructuring, once again after the doc-comment cleanup): all cells execute clean, including the
new 10.6 cells — the "10k"/"14k" checks train correctly, `get_result` correctly finds the 5k reference row,
and all three ("5k"/"10k"/"14k") print consistently across both new cells.
