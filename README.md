# NFL Big Data Bowl 2026: trajectory prediction with a temporal graph attention network

Given the tracking data of all the players until the ball is thrown, predict the `(x, y)` trajectory of the
players to predict while the ball is in the air. Each play is a spatio-temporal graph (nodes are player-frames,
spatial edges link the `K` nearest players in each frame, temporal edges link a player to himself in the next frame),
encoded by a TGAT, enriched with global play features and decoded in one shot.

The notebook `nfl_2026_bdb_script_final_complete.ipynb` is the report: it keeps the contributions (ball and global
features, model, loss, experiments, discussion) and calls this package for everything else, showing its output. Run
it in Colab (GPU runtime): its first cell clones this repository.

## Repository layout

| File | Content | Notebook chapter |
| --- | --- | --- |
| `nfl_2026_bdb_script_final_complete.ipynb` | The report | |
| `nfl_bdb/config.py` | `Config`: every parameter and hyperparameter of the pipeline | 1 |
| `nfl_bdb/utils.py` | Device and reproducibility (`get_device`, `fix_random`) | 1 |
| `nfl_bdb/data.py` | Loading, role sanitization, valid-play filter, players table, `get_play`, direction standardization | 2-4 |
| `nfl_bdb/analysis.py` | Data inspection, physique summaries, length bins and split checks | 2, 3, 7 |
| `nfl_bdb/viz.py` | Field, frame-by-frame play view, ground truth vs prediction | 4, 11 |
| `nfl_bdb/features.py` | Receiver/defender statistics, node features, edges, play context | 5, 6 |
| `nfl_bdb/dataset.py` | Stratified split, padded per-play dataset, PyG batches (`BatchProvider`) | 7 |
| `nfl_bdb/training.py` | Epoch loops, early stopping, `run_training`, `ExperimentRunner` | 9, 10 |
| `nfl_bdb/smoothing.py` | Savitzky-Golay post-processing of the predictions | 11 |
| `nfl_bdb/results.py` | Comparison table, ablation plots, cost-vs-accuracy plot, learning curves, global-branch diagnostics | 10, 11, 12 |

## Report structure

1. **Configuration** — repository/dependencies, data source, runtime, parameters, reproducibility.
2. **Data Loading and Preprocessing** — import, sanitization, inspection.
3. **Players Physique Inspection** — units, position aggregation, preprocessing.
4. **Plays: Retrieval, Standardization and Visualization**.
5. **Play Features** — ball travel features, targeted receiver/predicted defenders.
6. **Model Inputs** — node, edge and global features.
7. **Data Pipeline** — stratified sampling by play length (7.1), split sanity checks (7.2), padded tensors and PyG batches (7.3).
8. **Model** — GAT layer, spatio-temporal encoder, full model, size/dimension flow.
9. **Training and Baseline** — loss/metric, training procedure, starting point, test evaluation.
10. **Sequential Hyperparameter Search** — global features (10.1), batch size (10.2), hidden dimension (10.3),
    regressor width (10.4), GAT layers (10.5), sampled pool size (10.6), final model assembly (10.7). Each stage is
    decided on the same four criteria together — tradeoffs, timing, parameters and result — toward the smallest,
    fastest model that does not sacrifice accuracy.
11. **Results** — progress check across every run (11.1, including the cost-vs-accuracy plot), predicted
    trajectories on test plays (11.2), diagnostics of the global branch (11.3).
12. **Discussion and Conclusions** — justification of every Chapter 10 design choice under the same rubric (12.1),
    final result, generalization evidence from 10.6, global-branch takeaways, known qualitative limitations and
    future work (12.2).

## Design notes

* No function reads notebook globals: the tables travel in an `NFLData` object, the parameters in a `Config`, and
  the pieces that belong to the notebook (`get_ball_stats`, `get_global_features`, model, loss) are passed in as
  callables (`ExperimentRunner(cfg, batches, model_factory, loss_fn, metric_fn, device)`).
* `ExperimentRunner.run(batch_size=..., hidden_dim=..., global_mode=...)` is the single entry point of the baseline
  and of every ablation. Results are appended to `outputs/experiment_results.json` after each run, and a run that is
  already in the file is skipped, so an interrupted Colab session can be resumed.
* `BatchProvider` builds the PyG graphs once per `global_mode` and re-collates them for each batch size, keeping in
  memory only the last combination.
* Chapter 11.1 (and the final discussion in Chapter 12) reads only `outputs/experiment_results.json` through
  `load_all_results` / `plot_ablations` / `plot_cost_accuracy` — no model or GPU needed to reproduce those plots and
  tables from a finished run.
* The competition does not publish the leaderboard test set, so every "test" number in Chapters 9-12 comes from a
  local, stratified split (7.1) rather than an independent official set; Chapter 10.6 and Chapter 12.1 explain why,
  and 10.6 is read as a generalization check (does accuracy keep improving with more local data) rather than as an
  official held-out evaluation.

## Usage outside the notebook

```python
import sys; sys.path.insert(0, "NFL-BDB-26")
from nfl_bdb import Config
from nfl_bdb.data import load_raw_data

cfg = Config(DATA_PATH=..., AUXILIARY_DATA_PATH=...)
data = load_raw_data(cfg)
```

See `requirements.txt` for the dependencies.
