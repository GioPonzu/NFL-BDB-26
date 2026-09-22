"""Training machinery (notebook Chapters 9-10): one-epoch loops, early stopping, the single training
procedure `run_training`, and `ExperimentRunner`, the single entry point for the baseline and for every ablation.

The model and the loss are contributions of the notebook and are injected:
    model_factory(global_in_dim, hidden_dim, gat_num_layers, regressor_hidden_dim) -> nn.Module
    loss_fn(y_pred, y_true, output_mask), metric_fn(y_pred, y_true, output_mask) -> scalar tensor
"""
import gc
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .utils import fix_random


# ---------------------------------------------------------------- One epoch
def train_one_epoch(model, gnn_batch_list, optimizer, device, loss_fn, grad_clip_norm=1.0, progress=True):
    model.train()
    total_epoch_loss = 0

    pbar = tqdm(gnn_batch_list, desc="Training", disable=not progress)

    for gnn_batch in pbar:
        # y_target / y_mask travel inside the graph: no parallel DataLoader to keep aligned by index.
        gnn_batch = gnn_batch.to(device)
        y_true = gnn_batch.y_target        # [B, T, N, 2]
        output_mask = gnn_batch.y_mask     # [B, T, N]

        # 1. Forward pass: the model decodes each player directly (one-shot regressor)
        y_pred = model(gnn_batch)          # [B, T, N, 2]

        # 2. Loss: loss_fn is the one we backpropagate through. It can be the pure RMSE or a composite one
        # (RMSE + FDE + smoothness), chosen by the caller instead of being fixed here.
        loss = loss_fn(y_pred, y_true, output_mask)

        # 3. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: vital for spatio-temporal GNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= grad_clip_norm)

        optimizer.step()

        total_epoch_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return total_epoch_loss / len(gnn_batch_list)


@torch.no_grad()
def validate_one_epoch(model, gnn_batch_list, device, loss_fn, metric_fn, progress=True):
    """Keeps the loss (which the scheduler and early stopping act on; it can be composite) separate from the metric
    (the number to look at and compare: by default the pure RMSE in yards, independent of the extra terms the loss is
    optimizing). If loss_fn and metric_fn are the same function the two values coincide: expected, not a bug.
    Returns (avg_loss, avg_metric); the callers decide which one to show or log.
    """
    model.eval()
    total_val_loss = 0
    total_val_metric = 0

    pbar = tqdm(gnn_batch_list, desc="Validation", disable=not progress)

    for i, gnn_batch in enumerate(pbar):
        gnn_batch = gnn_batch.to(device)
        y_true = gnn_batch.y_target        # [B, T, N, 2]
        output_mask = gnn_batch.y_mask     # [B, T, N]

        # Forward once: loss and metric read the same y_pred
        y_pred = model(gnn_batch)

        loss = loss_fn(y_pred, y_true, output_mask)
        # If the two functions are the same one, do not compute it twice
        metric = loss if metric_fn is loss_fn else metric_fn(y_pred, y_true, output_mask)

        total_val_loss += loss.item()
        total_val_metric += metric.item()

        avg_loss = total_val_loss / (i + 1)
        avg_metric = total_val_metric / (i + 1)
        pbar.set_postfix({"Val Loss": f"{avg_loss:.4f}", "Val Metric": f"{avg_metric:.4f}"})

    n = len(gnn_batch_list)
    return total_val_loss / n, total_val_metric / n


# ---------------------------------------------------------------- Early stopping
class EarlyStopping:
    """'Percent plateau' early stopping: same relative-threshold logic as ReduceLROnPlateau (mode / threshold /
    threshold_mode), but instead of reducing the LR it stops the training when the validation loss stops improving by
    at least `threshold` (a percentage, like the scheduler) for `patience` consecutive epochs. The two mechanisms react
    to the same idea of "plateau", instead of one accepting numerical noise and the other not.
    """
    def __init__(self, mode='min', threshold=0.01, threshold_mode='rel', patience=20):
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.patience = patience

        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_better(self, current, best):
        if self.threshold_mode == 'rel':
            rel_delta = self.threshold * abs(best)
            return current < best - rel_delta if self.mode == 'min' else current > best + rel_delta
        return current < best - self.threshold if self.mode == 'min' else current > best + self.threshold

    def step(self, metric):
        if self.best is None or self._is_better(metric, self.best):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        self.should_stop = self.num_bad_epochs >= self.patience
        return self.should_stop


# ---------------------------------------------------------------- Single training procedure
def run_training(gnn_train_list, gnn_val_list, gnn_test_list, global_in_dim, run_label, model_factory,
                 loss_fn, metric_fn, cfg, device, hidden_dim=None, gat_num_layers=None, regressor_hidden_dim=None,
                 num_epochs=None, verbose=True, keep_model=False, progress=False):
    """The ONE training procedure, called with different configurations instead of copying the loop: the baseline and
    every ablation use it. Training/scheduler/early-stopping hyperparameters come from `cfg`; only what is being tested
    changes (batch lists, `global_in_dim`, `hidden_dim`, `gat_num_layers`, `regressor_hidden_dim`), in line with the
    one-factor-at-a-time principle.

    The seed is fixed again at every call: without it, the differences between runs would be confounded with the
    variance of the random weight initialization, not only with the factor under test.

    keep_model: if True the model (reloaded on the best checkpoint of THIS run before the test evaluation) is returned
    as result["model"]. Default False: the ablations train several models in a row and keeping them all would
    accumulate RAM/VRAM.
    """
    hidden_dim = hidden_dim or cfg.HIDDEN_DIM
    gat_num_layers = gat_num_layers or cfg.GAT_NUM_LAYERS
    regressor_hidden_dim = regressor_hidden_dim or cfg.REGRESSOR_HIDDEN_DIM
    num_epochs = num_epochs or cfg.NUM_EPOCHS
    fix_random(seed=cfg.RANDOM_SEED)

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(cfg.OUTPUT_DIR) / f"best_model_{run_label}.pt"
    t0 = time.time()

    model = model_factory(global_in_dim, hidden_dim, gat_num_layers, regressor_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    # ReduceLROnPlateau with a cooldown: without it a reduction could trigger another one right away, before the
    # model had time to benefit from the new LR. min_lr keeps the LR from collapsing to useless values.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.SCHEDULER_FACTOR, patience=cfg.SCHEDULER_PATIENCE,
        threshold=cfg.PLATEAU_THRESHOLD, threshold_mode=cfg.PLATEAU_THRESHOLD_MODE,
        cooldown=cfg.SCHEDULER_COOLDOWN, min_lr=cfg.SCHEDULER_MIN_LR,
    )
    # Higher patience than the scheduler, same relative threshold: the scheduler acts before we give up.
    early_stopping = EarlyStopping(mode='min', threshold=cfg.PLATEAU_THRESHOLD,
                                   threshold_mode=cfg.PLATEAU_THRESHOLD_MODE, patience=cfg.EARLY_STOP_PATIENCE)

    best_val_loss = float("inf")
    best_val_metric = None
    train_loss_history, val_loss_history, val_metric_history, lr_history = [], [], [], []

    print()
    for epoch in range(num_epochs):
        if verbose:
            print(f"[{run_label}] Epoch {epoch + 1}/{num_epochs}")

        avg_train_loss = train_one_epoch(model, gnn_train_list, optimizer, device, loss_fn=loss_fn,
                                         grad_clip_norm=cfg.GRAD_CLIP_NORM, progress=progress)
        avg_val_loss, avg_val_metric = validate_one_epoch(model, gnn_val_list, device,
                                                          loss_fn=loss_fn, metric_fn=metric_fn, progress=progress)
        scheduler.step(avg_val_loss)  # ReduceLROnPlateau: VAL, not TRAIN

        # Saving is driven by the LOSS (it can be composite); the metric is recorded at the same checkpoint. ANY
        # improvement, even tiny, saves: only the patience (scheduler + early stopping) ignores noise below 1%.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_metric = avg_val_metric
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print("Best model saved")

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        val_metric_history.append(avg_val_metric)
        lr_history.append(optimizer.param_groups[0]['lr'])

        if verbose:
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Val Metric: {avg_val_metric:.4f} | LR: {lr_history[-1]:.2e}")

        if early_stopping.step(avg_val_loss):
            if verbose:
                print(f"Early stopping ({run_label}) at epoch {epoch + 1}/{num_epochs}")
            break

    # Test set: reload the best checkpoint of THIS run, so test_loss/test_metric are computed on the best model.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    test_loss, test_metric = validate_one_epoch(model, gnn_test_list, device, loss_fn=loss_fn, metric_fn=metric_fn,
                                                progress=progress)
    elapsed_min = (time.time() - t0) / 60

    # Persist a copy of the best checkpoint outside OUTPUT_DIR (gitignored, ephemeral Colab scratch): every run's
    # weights land here too, not just the ones you end up choosing, so nothing has to be retrained later just to
    # get its .pt file back.
    if cfg.WEIGHTS_DIR:
        weights_dir = Path(cfg.WEIGHTS_DIR)
        weights_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, weights_dir / checkpoint_path.name)

    result = {
        "run_label": run_label,
        "hidden_dim": hidden_dim,
        "gat_num_layers": gat_num_layers,
        "regressor_hidden_dim": regressor_hidden_dim,
        "global_in_dim": global_in_dim,
        "actual_epochs": len(train_loss_history),
        "best_val_loss": best_val_loss,
        "best_val_metric": best_val_metric,
        "test_loss": test_loss,
        "test_metric": test_metric,
        "elapsed_min": elapsed_min,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "val_metric_history": val_metric_history,
        "lr_history": lr_history,
    }
    print(f"[{run_label}] done in {elapsed_min:.1f} min - "
          f"best val loss {best_val_loss:.4f} (metric {best_val_metric:.4f}), "
          f"test loss {test_loss:.4f} (metric {test_metric:.4f})")

    if keep_model:
        result["model"] = model
    else:
        del model
    # Free optimizer/scheduler before the next run: GPU memory would otherwise accumulate run after run.
    del optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------- Experiments (baseline and ablations)
class ExperimentRunner:
    """Single entry point for a training run, baseline or ablation: batch size, hidden dimension and global features
    are parameters of one call, which builds (or reuses) the batches and launches `run_training`.

    Every result is appended to `self.results` (a list of dicts, one per run) and, if `results_path` is set, saved
    to a json after each run: a Colab disconnection does not lose the finished runs, and a run whose label is already
    in the results is not repeated (`load_results` + `run` resume an interrupted study).
    """

    def __init__(self, cfg, batch_provider, model_factory, loss_fn, metric_fn, device, results_path=None):
        self.cfg = cfg
        self.batches = batch_provider
        self.model_factory = model_factory
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.device = device
        self.results_path = Path(results_path) if results_path else None
        self.results = []

    # ---- labels and bookkeeping
    @staticmethod
    def make_label(batch_size, hidden_dim, global_mode, gat_num_layers, regressor_hidden_dim):
        return f"bs{batch_size}_hd{hidden_dim}_gl{gat_num_layers}_rg{regressor_hidden_dim}_{global_mode}"

    def get_result(self, label):
        return next((r for r in self.results if r["run_label"] == label), None)

    def save_results(self, path=None):
        path = Path(path or self.results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f)
        return path

    def load_results(self, path=None):
        path = Path(path or self.results_path)
        if not path.exists():
            print(f"No saved results at {path}")
            return self.results
        with open(path) as f:
            self.results = json.load(f)
        print(f"Loaded {len(self.results)} finished runs from {path}: {[r['run_label'] for r in self.results]}")
        return self.results

    # ---- one run
    def run(self, batch_size=None, global_mode="all", hidden_dim=None, gat_num_layers=None,
            regressor_hidden_dim=None, num_epochs=None, run_label=None,
            verbose=True, keep_model=False, progress=False):
        batch_size = batch_size or self.cfg.BATCH_SIZE
        hidden_dim = hidden_dim or self.cfg.HIDDEN_DIM
        gat_num_layers = gat_num_layers or self.cfg.GAT_NUM_LAYERS
        regressor_hidden_dim = regressor_hidden_dim or self.cfg.REGRESSOR_HIDDEN_DIM
        label = run_label or self.make_label(batch_size, hidden_dim, global_mode, gat_num_layers,
                                             regressor_hidden_dim)

        done = self.get_result(label)
        if done is not None and not keep_model:
            print(f"[{label}] already done (test RMSE {done['test_metric']:.4f}), skipped")
            return done

        tr_list, va_list, te_list, g_dim = self.batches.get(batch_size, global_mode)

        if done is not None and keep_model:
            # Finished in a previous session: rebuild the model from its checkpoint instead of retraining
            ckpt = Path(self.cfg.OUTPUT_DIR) / f"best_model_{label}.pt"
            if ckpt.exists():
                model = self.model_factory(g_dim, hidden_dim, gat_num_layers, regressor_hidden_dim).to(self.device)
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.eval()
                print(f"[{label}] already done, model reloaded from {ckpt}")
                return {**done, "model": model}

        res = run_training(tr_list, va_list, te_list, global_in_dim=g_dim, run_label=label,
                           model_factory=self.model_factory, loss_fn=self.loss_fn, metric_fn=self.metric_fn,
                           cfg=self.cfg, device=self.device, hidden_dim=hidden_dim, gat_num_layers=gat_num_layers,
                           regressor_hidden_dim=regressor_hidden_dim, num_epochs=num_epochs,
                           verbose=verbose, keep_model=keep_model, progress=progress)
        res["batch_size"] = batch_size
        res["global_mode"] = global_mode

        stored = {k: v for k, v in res.items() if k != "model"}
        self.results = [r for r in self.results if r["run_label"] != label] + [stored]
        if self.results_path:
            self.save_results()
        return res


# ---------------------------------------------------------------- Sequential (greedy) search
def run_stage(runner, baseline, axis, grid, verbose=False):
    """One stage of a SEQUENTIAL/greedy search, as opposed to a one-factor-at-a-time ablation against one fixed
    baseline: runs `axis` over `grid` with every other factor fixed at `baseline`'s CURRENT value (which is
    expected to already carry the winners of whatever stages ran before this one), and prints a comparison table
    (validation/test RMSE, epochs actually trained, training time) for you to read.

    It does NOT pick a winner itself -- test RMSE alone does not capture training time or resource cost, and that
    trade-off is a judgment call. After reading the table, set `BASELINE["<axis>"] = <the value you pick>` yourself
    (a plain dict assignment) before moving to the next stage; nothing here does that for you.
    """
    results = [runner.run(**{**baseline, axis: v}, verbose=verbose) for v in grid]
    table = pd.DataFrame([{
        axis: r[axis], "val RMSE": r["best_val_metric"], "test RMSE": r["test_metric"],
        "epochs": r["actual_epochs"], "minutes": r["elapsed_min"],
    } for r in results])
    print(table.to_string(index=False))
    return table
