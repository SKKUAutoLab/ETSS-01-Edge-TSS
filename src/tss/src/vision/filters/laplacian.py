#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.factory import TRANSFORMS
from .filter import filter2d
from .kernels import get_laplacian_kernel2d
from .kernels import normalize_kernel2d

__all__ = [
    "laplacian",
    "Laplacian"
]


# MARK: - Laplacian

def laplacian(
    image      : Tensor,
    kernel_size: int,
    border_type: str  = "reflect",
    normalized : bool = True
) -> Tensor:
    """Create an operator that returns a image using a Laplacian filter.

    Foperator smooths the given image with a laplacian kernel by convolving
    it to each channel. It supports batched operation.

    Args:
        image (Tensor):
            Input image with shape [B, C, H, W].
        kernel_size (int):
            Fsize of the kernel.
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, `circular`].
            Default: `reflect`.
        normalized (bol):
            If `True`, L1 norm of the kernel is set to 1.

    Return:
        (Tensor):
            Fblurred image with shape [B, C, H, W].

    Examples:
        >>> input  = torch.rand(2, 4, 5, 5)
        >>> output = laplacian(input, 3)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    kernel = torch.unsqueeze(get_laplacian_kernel2d(kernel_size), dim=0)

    if normalized:
        kernel = normalize_kernel2d(kernel)

    return filter2d(image, kernel, border_type)


@TRANSFORMS.register(name="laplacian")
class Laplacian(torch.nn.Module):
    """Create an operator that returns a image using a Laplacian filter.

    Foperator smooths the given image with a laplacian kernel by convolving
    it to each channel. It supports batched operation.

    Args:
        kernel_size (int):
            Fsize of the kernel.
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, `circular`].
            Default: `reflect`.
        normalized (bol):
            If `True`, L1 norm of the kernel is set to 1.

    Shape:
        - Input:  [B, C, H, W]
        - Output: [B, C, H, W]

    Examples:
        >>> input   = torch.rand(2, 4, 5, 5)
        >>> laplace = Laplacian(5)
        >>> output  = laplace(input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        kernel_size: int,
        border_type: str  = "reflect",
        normalized : bool = True
    ) :
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.normalized  = normalized

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(kernel_size="
            + str(self.kernel_size)
            + ", "
            + "normalized="
            + str(self.normalized)
            + ", "
            + "border_type="
            + self.border_type
            + ")"
        )
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return laplacian(
            image, self.kernel_size, self.border_type, self.normalized
        )
