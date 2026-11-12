import random

import numpy as np
import torch
import yaml


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
