#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.factory import TRANSFORMS
from src.type import Dim2
from src.type import ListOrTuple2T
from .gaussian import gaussian_blur2d

__all__ = [
    "unsharp_mask", "UnsharpMask"
]


# MARK: - UnsharpMask

def unsharp_mask(
    image      : Tensor,
    kernel_size: Dim2,
    sigma      : ListOrTuple2T[float],
    border_type: str = "reflect"
) -> Tensor:
    """Create an operator that sharpens a image by applying operation
    out = 2 * image - gaussian_blur2d(image).

    Args:
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

    Returns:
        data_sharpened (Tensor):
            Fblurred image with shape [B, C, H, W].

    Examples:
        >>> input  = torch.rand(2, 4, 5, 5)
        >>> output = unsharp_mask(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    data_blur      = gaussian_blur2d(image, kernel_size, sigma, border_type)
    data_sharpened = image + (image - data_blur)
    return data_sharpened


@TRANSFORMS.register(name="unsharp_mask")
class UnsharpMask(torch.nn.Module):
    """Create an operator that sharpens image with:
    out = 2 * image - gaussian_blur2d(image).

    Args:
         kernel_size (Dim2):
            Fsize of the kernel.
        sigma (ListOrTuple2T[float]):
            Standard deviation of the kernel.
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, `circular`].
            Default: `reflect`.
 
    Shape:
        - Input:  [B, C, H, W]
        - Output: [B, C, H, W]
 
    Examples:
        >>> input   = torch.rand(2, 4, 5, 5)
        >>> sharpen = UnsharpMask((3, 3), (1.5, 1.5))
        >>> output  = sharpen(input)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        kernel_size: Dim2,
        sigma      : ListOrTuple2T[float],
        border_type: str = "reflect"
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma       = sigma
        self.border_type = border_type
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return unsharp_mask(
            image, self.kernel_size, self.sigma, self.border_type
        )
