#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.factory import TRANSFORMS
from .kernels import get_pascal_kernel_2d
from .median import _compute_zero_padding  # TODO: Move to proper place

__all__ = [
    "blur_pool2d",
    "BlurPool2D",
    "max_blur_pool2d",
    "MaxBlurPool2D",
]


# MARK: - BlurPool2D

def _blur_pool_by_kernel2d(
    image: Tensor, kernel: Tensor, stride: int
) -> Tensor:
    """Compute blur_pool by a given [C, C_out, N, N] kernel."""
    if not (len(kernel.shape) == 4 and kernel.size(-1) == kernel.size(-2)):
        raise AssertionError(f"Invalid kernel shape. Expect [C, C_out, N, N], "
                             f"Got: {kernel.shape}")
    padding = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return F.conv2d(image, kernel, padding=padding, stride=stride,
                    groups=image.size(1))


def blur_pool2d(
    image: Tensor, kernel_size: int, stride: int = 2
) -> Tensor:
    """Compute blurs and downsample a given feature map.

    Args:
        image (Tensor):
            Image.
        kernel_size (int):
            Kernel size for max pooling.
        stride (int):
            Should be true to match output size of conv2d with same kernel size.

    Shape:
        - Input:  [B, C, H, W]
        - Output: [N, C, H_{out}, W_{out}], where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{kernel\_size//2}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{kernel\_size//2}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Returns:
        (Tensor):
            Ftransformed image.

    Notes:
        This function is tested against https://github.com/adobe/antialiased-cnns.
 
    Examples:
        >>> input = torch.eye(5)[None, None]
        >>> blur_pool2d(input, 3)
        image([[[[0.3125, 0.0625, 0.0000],
                  [0.0625, 0.3750, 0.0625],
                  [0.0000, 0.0625, 0.3125]]]])
    """
    kernel = (
        get_pascal_kernel_2d(kernel_size, norm=True)
        .repeat((image.size(1), 1, 1, 1))
        .to(image)
    )
    return _blur_pool_by_kernel2d(image, kernel, stride)


@TRANSFORMS.register(name="blur_pool2d")
class BlurPool2D(torch.nn.Module):
    """Compute blur (anti-aliasing) and downsample a given feature map.

    Args:
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.

    Shape:
        - Input:  [B, C, H, W]
        - Output: [N, C, H_{out}, W_{out}], where:

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{kernel\_size//2}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{kernel\_size//2}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Examples:
        >>> input = torch.eye(5)[None, None]
        >>> bp    = BlurPool2D(kernel_size=3, stride=2)
        >>> bp(input)
        image([[[[0.3125, 0.0625, 0.0000],
                  [0.0625, 0.3750, 0.0625],
                  [0.0000, 0.0625, 0.3125]]]])
    """

    def __init__(self, kernel_size: int, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.register_buffer(
            "kernel", get_pascal_kernel_2d(kernel_size, norm=True)
        )

    def forward(self, image: Tensor) -> Tensor:
        # To align the logic with the whole lib
        kernel = torch.as_tensor(
            self.kernel, device=image.device, dtype=image.dtype
        )
        return _blur_pool_by_kernel2d(
            image, kernel.repeat((image.size(1), 1, 1, 1)), self.stride
        )


# MARK: - MaxBlurPool2D

def _max_blur_pool_by_kernel2d(
    image        : Tensor,
    kernel       : Tensor,
    stride       : int,
    max_pool_size: int,
    ceil_mode    : bool
) -> Tensor:
    """Compute max_blur_pool by a given [C, C_out, N, N] kernel."""
    if not (len(kernel.shape) == 4 and kernel.size(-1) == kernel.size(-2)):
        raise AssertionError(f"Invalid kernel shape. Expect [C, C_out, N, N], "
                             f"Got: {kernel.shape}")
    # Compute local maxima
    image = F.max_pool2d(image, kernel_size=max_pool_size, padding=0, stride=1,
                         ceil_mode=ceil_mode)
    # Blur and downsample
    padding = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return F.conv2d(
        image, kernel, padding=padding, stride=stride, groups=image.size(1)
    )


def max_blur_pool2d(
    image        : Tensor,
    kernel_size  : int,
    stride       : int  = 2,
    max_pool_size: int  = 2,
    ceil_mode    : bool = False
) -> Tensor:
    """Compute pools and blurs and downsample a given feature map.

    Args:
        image (Tensor):
            Image.
        kernel_size (int):
            Kernel size for max pooling.
        stride (int):
            Stride for pooling.
        max_pool_size (int):
            Kernel size for max pooling.
        ceil_mode (bool):
            Should be true to match output size of conv2d with same kernel size.

    Notes:
        This function is tested against:
        https://github.com/adobe/antialiased-cnns.

    Examples:
        >>> input = torch.eye(5)[None, None]
        >>> max_blur_pool2d(input, 3)
        image([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])
    """
    if not len(image.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect [B, C, H, W]. "
                         f"Got: {image.shape}")
    kernel = (
        get_pascal_kernel_2d(kernel_size, norm=True)
        .repeat((image.size(1), 1, 1, 1))
        .to(image)
    )
    return _max_blur_pool_by_kernel2d(
        image, kernel, stride, max_pool_size, ceil_mode
    )


@TRANSFORMS.register(name="max_blur_pool2d")
class MaxBlurPool2D(torch.nn.Module):
    """Compute pools and blurs and downsample a given feature map.

    Equivalent to `nn.Sequential(nn.MaxPool2d(...), BlurPool2D(...))`

    Args:
        kernel_size (int):
            Kernel size for max pooling.
        stride (int):
            Stride for pooling.
        max_pool_size (int):
            Kernel size for max pooling.
        ceil_mode (bool):
            Should be true to match output size of conv2d with same kernel size.

    Shape:
        - Input:  [B, C, H, W]
        - Output: [B, C, H / stride, W / stride]

    Examples:
        >>> input = torch.eye(5)[None, None]
        >>> mbp   = MaxBlurPool2D(kernel_size=3, stride=2, max_pool_size=2, ceil_mode=False)
        >>> mbp(input)
        image([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])
        >>> seq = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), BlurPool2D(kernel_size=3, stride=2))
        >>> seq(input)
        image([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        kernel_size  : int,
        stride       : int  = 2,
        max_pool_size: int  = 2,
        ceil_mode    : bool = False
    ):
        super().__init__()
        self.kernel_size   = kernel_size
        self.stride        = stride
        self.max_pool_size = max_pool_size
        self.ceil_mode     = ceil_mode
        self.register_buffer(
            "kernel", get_pascal_kernel_2d(kernel_size, norm=True)
        )
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        # To align the logic with the whole lib
        kernel = torch.as_tensor(
            self.kernel, device=image.device, dtype=image.dtype
        )
        return _max_blur_pool_by_kernel2d(
            image, kernel.repeat((image.size(1), 1, 1, 1)), self.stride,
            self.max_pool_size, self.ceil_mode
        )
