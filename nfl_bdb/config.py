"""Parameters and hyperparameters of the whole pipeline (Section 1.4 of the notebook).

The notebook instantiates `Config(...)` with every value spelled out, so what is used in a run is visible in the
notebook itself; the defaults below are the values of the reported experiments.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    # ---------------- Data ----------------
    DATA_PATH: Optional[Path] = None
    AUXILIARY_DATA_PATH: Optional[Path] = None
    WEEKS: List[int] = field(default_factory=lambda: list(range(1, 19)))
    FRAME_RATE: int = 10                 # frames per second

    # ---------------- Padding limits (upper bounds of the padded tensors) ----------------
    MAX_INPUT_NODES: int = 17
    MAX_INPUT_FRAMES: int = 123
    MAX_OUTPUT_NODES: int = 17
    MAX_OUTPUT_FRAMES: int = 94
    K: int = 5                           # spatial neighbours per node

    # ---------------- Model ----------------
    HIDDEN_DIM: int = 128
    NODE_IN_DIM: int = 36
    EDGE_IN_DIM: int = 7
    GAT_NUM_LAYERS: int = 1              # message-passing layers stacked in each GAT branch (spatial, temporal)
    REGRESSOR_HIDDEN_DIM: int = 512      # width of the hidden layer of the final per-player regressor

    # ---------------- Dataset split ----------------
    N_TRAIN: int = 300
    N_VAL: int = 50
    N_TEST: int = 50
    N_LENGTH_BINS: int = 15

    # ---------------- Training ----------------
    NUM_EPOCHS: int = 100
    EARLY_STOP_PATIENCE: int = 20
    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 2e-3
    GRAD_CLIP_NORM: float = 1.0

    # Scheduler + early stopping share the same "percent plateau" definition of improvement
    PLATEAU_THRESHOLD: float = 0.01
    PLATEAU_THRESHOLD_MODE: str = "rel"
    SCHEDULER_FACTOR: float = 0.5
    SCHEDULER_PATIENCE: int = 6
    SCHEDULER_COOLDOWN: int = 1
    SCHEDULER_MIN_LR: float = 1e-7

    # ---------------- Loss ----------------
    W_FDE: float = 2.0                   # weight of the error on the last valid frame (FDE)
    W_SMOOTH: float = 0.5                # weight of the smoothness (jerk) regularizer

    # ---------------- Bookkeeping ----------------
    RANDOM_SEED: int = 42
    OUTPUT_DIR: Path = Path("outputs")   # checkpoints and results of the runs

    @property
    def N_TOTAL(self) -> int:
        return self.N_TRAIN + self.N_VAL + self.N_TEST

    @property
    def MIN_INPUT_NODES(self) -> int:
        """KNN needs at least K + 1 players in every frame."""
        return self.K + 1
