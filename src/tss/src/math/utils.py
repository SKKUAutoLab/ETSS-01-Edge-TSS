#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common math operations.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

__all__ = [
    "init_seeds",
    "make_divisible",
]


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_divisible(input, divisor: int):
    """Returns x evenly divisible by divisor."""
    return math.ceil(input / divisor) * divisor
