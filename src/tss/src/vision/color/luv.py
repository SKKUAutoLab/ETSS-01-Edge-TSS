#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Luv color space.

FRGB to Luv color transformations were translated from scikit image's
rgb2luv and luv2rgb:
https://github.com/scikit-image/scikit-image/blob/a48bf6774718c64dade4548153ae16065b595ca9/skimage/color/colorconv.py
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.factory import TRANSFORMS
from .rgb import linear_rgb_to_rgb
from .rgb import rgb_to_linear_rgb
from .xyz import rgb_to_xyz
from .xyz import xyz_to_rgb

__all__ = [
    "luv_to_rgb",
    "LuvToRgb",
    "rgb_to_luv",
    "RgbToLuv"
]


# MARK: - LuvToRgb

def luv_to_rgb(image: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert a Luv image to RGB.

    Args:
        image (Tensor):
            Luv image to be converted to RGB with shape [*, 3, H, W].
        eps (float):
            For numerically stability when dividing.

    Returns:
        luv (Tensor):
            Luv version of the image with shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = luv_to_rgb(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    L = image[..., 0, :, :]
    u = image[..., 1, :, :]
    v = image[..., 2, :, :]

    # Convert from Luv to XYZ
    y = torch.where(L > 7.999625, torch.pow((L + 16) / 116, 3.0), L / 903.3)

    # Compute white point
    xyz_ref_white = (0.95047, 1.0, 1.08883)
    u_w = ((4 * xyz_ref_white[0]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))
    v_w = ((9 * xyz_ref_white[1]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))

    a = u_w + u / (13 * L + eps)
    d = v_w + v / (13 * L + eps)
    c = 3 * y * (5 * d - 3)
    z = ((a - 4) * c - 15 * a * d * y) / (12 * d + eps)
    x = -(c / (d + eps) + 3.0 * z)

    xyz_im  = torch.stack([x, y, z], -3)
    rgbs_im = xyz_to_rgb(xyz_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)
    return rgb_im


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="luv_to_rgb")
class LuvToRgb(torch.nn.Module):
    """Convert an image from Luv to RGB.

    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> rgb    = LuvToRgb()
        >>> output = rgb(input)  # [2, 3, 4, 5]

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] http://www.poynton.com/ColorFAQ.html
    """

    def forward(self, image: Tensor) -> Tensor:
        return luv_to_rgb(image)


# MARK: - RgbToLuv

def rgb_to_luv(image: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert an RGB image to Luv. Image data is assumed to be in the
    range of [0.0, 1.0]. Luv color is computed using the D65 illuminant and
    Observer 2.

    Args:
        image (Tensor):
            RGB Image to be converted to Luv with shape [*, 3, H, W].
        eps (float):
            For numerically stability when dividing.

    Returns:
        luv (Tensor):
            Luv version of the image with shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_luv(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)
    
    xyz_im  = rgb_to_xyz(lin_rgb)
    x       = xyz_im[..., 0, :, :]
    y       = xyz_im[..., 1, :, :]
    z       = xyz_im[..., 2, :, :]

    threshold = 0.008856
    L = torch.where(y > threshold,
                    116.0 * torch.pow(y.clamp(min=threshold), 1.0 / 3.0) - 16.0,
                    903.3 * y)

    # Compute reference white point
    xyz_ref_white = (0.95047, 1.0, 1.08883)
    u_w = ((4 * xyz_ref_white[0]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))
    v_w = ((9 * xyz_ref_white[1]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))

    u_p = (4 * x) / (x + 15 * y + 3 * z + eps)
    v_p = (9 * y) / (x + 15 * y + 3 * z + eps)

    u   = 13 * L * (u_p - u_w)
    v   = 13 * L * (v_p - v_w)

    out = torch.stack([L, u, v], dim=-3)
    return out


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rgb_to_luv")
class RgbToLuv(torch.nn.Module):
    """Convert an image from RGB to Luv. Image data is assumed to be in the
    range of [0.0, 1.0]. Luv color is computed using the D65 illuminant and
    Observer 2.

    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> luv    = RgbToLuv()
        >>> output = luv(input)  # [2, 3, 4, 5]

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] http://www.poynton.com/ColorFAQ.html
    """

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_luv(image)
