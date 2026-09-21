# NFL Big Data Bowl 2026: trajectory prediction with a temporal graph attention network

Given the tracking data of all the players until the ball is thrown, predict the `(x, y)` trajectory of the
players to predict while the ball is in the air. Each play is a spatio-temporal graph (nodes are player-frames,
spatial edges link the `K` nearest players in each frame, temporal edges link a player to himself in the next frame),
encoded by a TGAT, enriched with global play features and decoded in one shot.

The notebook `NFL_2026_BDB_script.ipynb` is the report: it keeps the contributions (ball and global features, model,
loss, experiments) and calls this package for everything else, showing its output. Run it in Colab (GPU runtime):
its first cell clones this repository.

## Repository layout

| File | Content | Notebook chapter |
| --- | --- | --- |
| `NFL_2026_BDB_script.ipynb` | The report | |
| `nfl_bdb/config.py` | `Config`: every parameter and hyperparameter of the pipeline | 1 |
| `nfl_bdb/utils.py` | Device and reproducibility (`get_device`, `fix_random`) | 1 |
| `nfl_bdb/data.py` | Loading, role sanitization, valid-play filter, players table, `get_play`, direction standardization | 2-4 |
| `nfl_bdb/analysis.py` | Data inspection, physique summaries, length bins and split checks | 2, 3, 7 |
| `nfl_bdb/viz.py` | Field, frame-by-frame play view, ground truth vs prediction | 4, 11 |
| `nfl_bdb/features.py` | Receiver/defender statistics, node features, edges, play context | 5, 6 |
| `nfl_bdb/dataset.py` | Stratified split, padded per-play dataset, PyG batches (`BatchProvider`) | 7 |
| `nfl_bdb/training.py` | Epoch loops, early stopping, `run_training`, `ExperimentRunner` | 9, 10 |
| `nfl_bdb/smoothing.py` | Savitzky-Golay post-processing of the predictions | 11 |
| `nfl_bdb/results.py` | Comparison table, ablation plots, learning curves, global-branch diagnostics | 11 |

## Design notes

* No function reads notebook globals: the tables travel in an `NFLData` object, the parameters in a `Config`, and
  the pieces that belong to the notebook (`get_ball_stats`, `get_global_features`, model, loss) are passed in as
  callables (`ExperimentRunner(cfg, batches, model_factory, loss_fn, metric_fn, device)`).
* `ExperimentRunner.run(batch_size=..., hidden_dim=..., global_mode=...)` is the single entry point of the baseline
  and of every ablation. Results are appended to `outputs/experiment_results.json` after each run, and a run that is
  already in the file is skipped, so an interrupted Colab session can be resumed.
* `BatchProvider` builds the PyG graphs once per `global_mode` and re-collates them for each batch size, keeping in
  memory only the last combination.

## Usage outside the notebook

```python
import sys; sys.path.insert(0, "NFL-BDB-26")
from nfl_bdb import Config
from nfl_bdb.data import load_raw_data

cfg = Config(DATA_PATH=..., AUXILIARY_DATA_PATH=...)
data = load_raw_data(cfg)
```

See `requirements.txt` for the dependencies.
