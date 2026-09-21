"""Support code of the NFL Big Data Bowl 2026 notebook: data handling, feature helpers, dataset, training, plots.

Modules (in the order in which the notebook uses them):
    config      Config dataclass with every parameter of the pipeline
    utils       device and reproducibility helpers
    data        loading, sanitization, players table, play retrieval and direction standardization
    analysis    data inspection, physique summaries, length bins and split checks
    features    targeted receiver / defender statistics, node features, edges, play context
    dataset     stratified split, padded per-play dataset, PyG batches
    training    epoch loops, early stopping, run_training and ExperimentRunner
    smoothing   Savitzky-Golay post-processing of the trajectories
    viz         field, play strips, ground truth vs prediction
    results     comparison table, ablation plots, learning curves
"""
from .config import Config
from .utils import fix_random, get_device
