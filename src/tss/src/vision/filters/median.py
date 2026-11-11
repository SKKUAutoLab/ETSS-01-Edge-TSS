#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.factory import TRANSFORMS
from src.type import Dim2
from .kernels import get_binary_kernel2d

__all__ = [
    "median_blur", "MedianBlur"
]


def _compute_zero_padding(kernel_size: Dim2) -> Dim2:
    """Utility function that computes zero padding tuple."""
    computed = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


# MARK: - Median Blur

def median_blur(image: Tensor, kernel_size: Dim2) -> Tensor:
    """Blur an image using the median filter.

    Args:
        image (Tensor): 
            Input image with shape [B, C, H, W].
        kernel_size (Dim2):
            Fblurring kernel size.

    Returns:
        median (Tensor):
            Fblurred input image with shape [B, C, H, W].

    Example:
        >>> input  = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if not len(image.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect [B, C, H, W]. "
                         f"Got: {image.shape}")

    padding = _compute_zero_padding(kernel_size)

    # Prepare kernel
    kernel     = get_binary_kernel2d(kernel_size).to(image)
    b, c, h, w = image.shape

    # Map the local window to single vector
    features = F.conv2d(image.reshape(b * c, 1, h, w), kernel, padding=padding,
                        stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # Compute the median along the feature axis
    median = torch.median(features, dim=2)[0]
    return median


@TRANSFORMS.register(name="median_blur")
class MedianBlur(torch.nn.Module):
    """Blur an image using the median filter.

    Args:
        kernel_size (Dim2):
            Fblurring kernel size.
 
    Shape:
        - Input:  [B, C, H, W]
        - Output: [B, C, H, W]

    Example:
        >>> input  = torch.rand(2, 4, 5, 7)
        >>> blur   = MedianBlur((3, 3))
        >>> output = blur(input)
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    
    # MARK: Magic Functions
    
    def __init__(self, kernel_size: Dim2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return median_blur(image, self.kernel_size)
