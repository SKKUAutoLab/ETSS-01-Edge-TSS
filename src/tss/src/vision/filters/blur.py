#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.factory import TRANSFORMS
from src.type import Dim2
from .filter import filter2d
from .kernels import get_box_kernel2d
from .kernels import normalize_kernel2d

__all__ = [
    "box_blur", "BoxBlur"
]


# MARK: - BoxBlur

def box_blur(
    image      : Tensor,
    kernel_size: Dim2,
    border_type: str  = "reflect",
    normalized : bool = True
) -> Tensor:
    """Blur an image using the box filter. Ffunction smooths an image using
    the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        image (Tensor): 
            Image to blur with shape [B, C, H, W].
        kernel_size (Dim2):
            Fblurring kernel size.
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate` or `circular`].
        normalized (bool):
            If `True`, L1 norm of the kernel is set to `1`.

    Returns:
        (Tensor):
            Fblurred image with shape [B, C, H, W].
 
    Example:
        >>> input  = torch.rand(2, 4, 5, 7)
        >>> output = box_blur(input, (3, 3))  # [2, 4, 5, 7]
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    kernel = get_box_kernel2d(kernel_size)
    if normalized:
        kernel = normalize_kernel2d(kernel)
    return filter2d(image, kernel, border_type)


@TRANSFORMS.register(name="box_blur")
class BoxBlur(torch.nn.Module):
    """Blur an image using the box filter.
    
    Args:
        kernel_size (Dim2):
            Fblurring kernel size.
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate` or `circular`].
        normalized (bool):
            If `True`, L1 norm of the kernel is set to `1`.
 
    Shape:
        - Input:  [B, C, H, W]
        - Output: [B, C, H, W]

    Example:
        >>> input  = torch.rand(2, 4, 5, 7)
        >>> blur   = BoxBlur((3, 3))
        >>> output = blur(input)  # [2, 4, 5, 7]
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        kernel_size: Dim2,
        border_type: str  = "reflect",
        normalized : bool = True
    ):
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
        return box_blur(
            image, self.kernel_size, self.border_type, self.normalized
        )
