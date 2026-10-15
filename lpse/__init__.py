"""LPSE package: model, encoders, ingestion, replay."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all global RNG sources (python, numpy, torch, cuda) for
    reproducible node dropout, replay sampling, QA template sampling, and
    model initialization across every trainer."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
