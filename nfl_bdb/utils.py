"""Small helpers shared by the notebook sections."""
import numpy as np
import torch


def get_device():
    """GPU when available, CPU otherwise."""
    if torch.cuda.is_available():
        print("All good, a GPU is available.")
        return torch.device("cuda:0")
    print("Please set GPU via Runtime -> Change runtime type.")
    return "cpu"


def fix_random(seed: int) -> None:
    """Fix the sources of randomness of NumPy and PyTorch (CPU and CUDA)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
