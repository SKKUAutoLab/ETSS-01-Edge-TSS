#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ycbcr color space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.factory import TRANSFORMS

__all__ = [
    "rgb_to_ycbcr",
    "RgbToYcbcr",
    "ycbcr_to_rgb",
    "YcbcrToRgb"
]


# MARK: - RgbToYcbcr

def rgb_to_ycbcr(image: Tensor) -> Tensor:
    """Convert an RGB image to YCbCr.

    Args:
        image (Tensor):
            RGB Image to be converted to YCbCr with shape [*, 3, H, W].

    Returns:
        ycbcr (Tensor):
            YCbCr version of the image with shape [*, 3, H, W].

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    delta = 0.5
    y     = 0.299 * r + 0.587 * g + 0.114 * b
    cb    = (b - y) * 0.564 + delta
    cr    = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rgb_to_ycbcr")
class RgbToYcbcr(nn.Module):
    """Convert an image from RGB to YCbCr. Image data is assumed to be in
    the range of [0.0, 1.0].
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> ycbcr  = RgbToYcbcr()
        >>> output = ycbcr(input)  # [2, 3, 4, 5]
    """

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_ycbcr(image)


# MARK: - YcbcrToRgb

def ycbcr_to_rgb(image: Tensor) -> Tensor:
    """Convert an YCbCr image to RGB. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (Tensor):
            YCbCr Image to be converted to RGB with shape [*, 3, H, W].

    Returns:
        rbg (Tensor):
            RGB version of the image with shape [*, 3, H, W].

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    y  = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    delta = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta

    r = y + 1.403 * cr_shifted
    g = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="ycbcr_to_rgb")
class YcbcrToRgb(nn.Module):
    """Convert an image from YCbCr to Rgb. Image data is assumed to be in
    the range of [0.0, 1.0].
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> rgb    = YcbcrToRgb()
        >>> output = rgb(input)  # [2, 3, 4, 5]
    """

    def forward(self, image: Tensor) -> Tensor:
        return ycbcr_to_rgb(image)
