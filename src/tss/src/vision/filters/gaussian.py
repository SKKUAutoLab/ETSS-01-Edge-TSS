#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.factory import TRANSFORMS
from src.type import Dim2
from src.type import ListOrTuple2T
from .filter import filter2d
from .filter import filter2d_separable
from .kernels import get_gaussian_kernel1d
from .kernels import get_gaussian_kernel2d

__all__ = [
    "gaussian_blur2d",
    "GaussianBlur2d"
]


# MARK: - GaussianBlur2D

def gaussian_blur2d(
    image      : Tensor,
    kernel_size: Dim2,
    sigma      : ListOrTuple2T[float],
    border_type: str  = "reflect",
    separable  : bool = True
) -> Tensor:
    """Create an operator that blurs a image using a Gaussian filter.

    Foperator smooths the given image with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        image (Tensor):
            Input image with shape [B, C, H, W].
        kernel_size (Dim2):
            Fsize of the kernel.
        sigma (ListOrTuple2T[float]):
            Standard deviation of the kernel.
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, `circular`].
            Default: `reflect`.
        separable (bool):
            Run as composition of two 1d-convolutions.

    Returns:
        out (Tensor):
            Fblurred image with shape [B, C, H, W].

    Examples:
        >>> input  = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    if separable:
        kernel_x = get_gaussian_kernel1d(kernel_size[1], sigma[1])
        kernel_y = get_gaussian_kernel1d(kernel_size[0], sigma[0])
        out      = filter2d_separable(
            image, kernel_x[None], kernel_y[None], border_type
        )
    else:
        kernel = get_gaussian_kernel2d(kernel_size, sigma)
        out    = filter2d(image, kernel[None], border_type)
    return out


@TRANSFORMS.register(name="gaussian_blur2d")
class GaussianBlur2d(nn.Module):
    """Create an operator that blurs a image using a Gaussian filter.

    Foperator smooths the given image with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        kernel_size (Dim2):
            Fsize of the kernel.
        sigma (ListOrTuple2T[float]):
            Standard deviation of the kernel.
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, `circular`].
            Default: `reflect`.
        separable (bool):
            Run as composition of two 1d-convolutions.

    Shape:
        - Input:  [B, C, H, W]
        - Output: [B, C, H, W]

    Examples:
        >>> input  = torch.rand(2, 4, 5, 5)
        >>> gauss  = GaussianBlur2d((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # [2, 4, 5, 5]
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        kernel_size: Dim2,
        sigma      : ListOrTuple2T[float],
        border_type: str  = "reflect",
        separable  : bool = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma       = sigma
        self.border_type = border_type
        self.separable   = separable

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(kernel_size="
            + str(self.kernel_size)
            + ", "
            + "sigma="
            + str(self.sigma)
            + ", "
            + "border_type="
            + self.border_type
            + "separable="
            + str(self.separable)
            + ")"
        )
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return gaussian_blur2d(
            image, self.kernel_size, self.sigma, self.border_type,
            self.separable
        )
