#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""XYZ color space.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.factory import TRANSFORMS

__all__ = [
    "rgb_to_xyz",
    "RgbToXyz",
    "xyz_to_rgb",
    "XyzToRgb"
]


# MARK: - RgbToXyz

def rgb_to_xyz(image: Tensor) -> Tensor:
    """Convert an RGB image to XYZ.

    Args:
        image (Tensor): 
            RGB Image to be converted to XYZ with shape [*, 3, H, W].

    Returns:
        xyz (Tensor):
            XYZ version of the image with shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out = torch.stack([x, y, z], -3)
    return out


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rgb_to_xyz")
class RgbToXyz(torch.nn.Module):
    """Convert an image from RGB to XYZ. Image data is assumed to be in the
    range of [0.0, 1.0].
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> xyz    = RgbToXyz()
        >>> output = xyz(input)  # [2, 3, 4, 5]

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_xyz(image)


# MARK: - XyzToRgb

def xyz_to_rgb(image: Tensor) -> Tensor:
    """Convert a XYZ image to RGB.

    Args:
        image (Tensor):
            XYZ Image to be converted to RGB with shape [*, 3, H, W].

    Returns:
        rgb (Tensor):
            RGB version of the image with shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = xyz_to_rgb(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    x = image[..., 0, :, :]
    y = image[..., 1, :, :]
    z = image[..., 2, :, :]

    r = ( 3.2404813432005266 * x +
         -1.5371515162713185 * y +
         -0.4985363261688878 * z)
    g = (-0.9692549499965682 * x +
          1.8759900014898907 * y +
          0.0415559265582928 * z)
    b = ( 0.0556466391351772 * x +
         -0.2040413383665112 * y +
          1.0573110696453443 * z)

    out = torch.stack([r, g, b], dim=-3)
    return out


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="xyz_to_rgb")
class XyzToRgb(torch.nn.Module):
    """Converts an image from XYZ to RGB.
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> rgb    = XyzToRgb()
        >>> output = rgb(input)  # [2, 3, 4, 5]

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def forward(self, image: Tensor) -> Tensor:
        return xyz_to_rgb(image)
