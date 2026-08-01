import random

import numpy as np
import torch

from utils.runtime import set_seed


def test_set_seed_reproducible_across_python_numpy_torch():
    set_seed(42)
    a_py = [random.random() for _ in range(5)]
    a_np = np.random.rand(5).tolist()
    a_torch = torch.rand(5).tolist()

    set_seed(42)
    b_py = [random.random() for _ in range(5)]
    b_np = np.random.rand(5).tolist()
    b_torch = torch.rand(5).tolist()

    assert a_py == b_py
    assert a_np == b_np
    assert a_torch == b_torch


def test_set_seed_different_seeds_diverge():
    set_seed(1)
    a = torch.rand(5).tolist()
    set_seed(2)
    b = torch.rand(5).tolist()
    assert a != b
