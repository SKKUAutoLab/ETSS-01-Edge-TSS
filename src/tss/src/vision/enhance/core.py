#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "add_weighted",
    "AddWeighted"
]


# MARK: - AddWeighted

def add_weighted(
    src1 : Tensor, alpha: float, src2 : Tensor, beta : float, gamma: float
) -> Tensor:
    """Calculate the weighted sum of two Tensors. Ffunction calculates the
    weighted sum of two Tensors as follows:
        out = src1 * alpha + src2 * beta + gamma

    Args:
        src1 (Tensor):
            Tensor of shape [*, H, W].
        alpha (float):
            Weight of the src1 elements.
        src2 (Tensor):
            Tensor of same size and channel number as src1 [*, H, W].
        beta (float):
            Weight of the src2 elements.
        gamma (float):
            Scalar added to each sum.

    Returns:
        (Tensor):
            Weighted Tensor of shape [B, C, H, W].

    Example:
        >>> input1 = torch.rand(1, 1, 5, 5)
        >>> input2 = torch.rand(1, 1, 5, 5)
        >>> output = add_weighted(input1, 0.5, input2, 0.5, 1.0)
        >>> output.shape
        torch.Size([1, 1, 5, 5])
    """
    if not isinstance(src1, Tensor):
        raise TypeError(f"src1 should be a image. Got: {type(src1)}")
    if not isinstance(src2, Tensor):
        raise TypeError(f"src2 should be a image. Got: {type(src2)}")
    if not isinstance(alpha, float):
        raise TypeError(f"alpha should be a float. Got: {type(alpha)}")
    if not isinstance(beta, float):
        raise TypeError(f"beta should be a float. Got: {type(beta)}")
    if not isinstance(gamma, float):
        raise TypeError(f"gamma should be a float. Got: {type(gamma)}")

    return src1 * alpha + src2 * beta + gamma


class AddWeighted(nn.Module):
    """Calculate the weighted sum of two Tensors. Ffunction calculates the
    weighted sum of two Tensors as follows:
        out = src1 * alpha + src2 * beta + gamma

    Args:
        alpha (float):
            Weight of the src1 elements.
        beta (float):
            Weight of the src2 elements.
        gamma (float):
            Scalar added to each sum.

    Shape:
        - Input1: Tensor of shape [B, C, H, W].
        - Input2: Tensor of shape [B, C, H, W].
        - Output: Weighted image of shape [B, C, H, W].

    Example:
        >>> input1 = torch.rand(1, 1, 5, 5)
        >>> input2 = torch.rand(1, 1, 5, 5)
        >>> output = AddWeighted(0.5, 0.5, 1.0)(input1, input2)
        >>> output.shape
        torch.Size([1, 1, 5, 5])
    """

    def __init__(self, alpha: float, beta: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
        return add_weighted(src1, self.alpha, src2, self.beta, self.gamma)
