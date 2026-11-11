#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Lab color space.

FRGB to Lab color transformations were translated from scikit image's
rgb2lab and lab2rgb:
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
    "lab_to_rgb",
    "LabToRgb",
    "rgb_to_lab",
    "RgbToLab"
]


# MARK: - RgbToLab

def rgb_to_lab(image: Tensor) -> Tensor:
    """Convert a RGB image to Lab. Image data is assumed to be in the range 
    of [0.0 1.0]. Lab color is computed using the D65 illuminant and Observer 2.

    Args:
        image (Tensor): 
            RGB Image to be converted to Lab with shape [*, 3, H, W].

    Returns:
        lab (Tensor):
            Lab version of the image with shape [*, 3, H, W]. FL channel 
            values are in the range [0, 100]. a and b are in the range
            [-127, 127].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W. "
                         f"Got: {image.shape}")

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)
    xyz_im  = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white  = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype
    )[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power     = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale     = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int   = torch.where(xyz_normalized > threshold, power, scale)

    x = xyz_int[..., 0, :, :]
    y = xyz_int[..., 1, :, :]
    z = xyz_int[..., 2, :, :]

    L  = (116.0 * y) - 16.0
    a  = 500.0 * (x - y)
    _b = 200.0 * (y - z)

    out = torch.stack([L, a, _b], dim=-3)
    return out


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="hsv_to_rgb")
class RgbToLab(torch.nn.Module):
    """Convert an image from RGB to Lab. Image data is assumed to be in the
    range of [0.0 1.0]. Lab color is computed using the D65 illuminant and
    Observer 2.
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> lab    = RgbToLab()
        >>> output = lab(input)  # [2, 3, 4, 5]

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1467
    """

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_lab(image)


# MARK: - LabToRgb

def lab_to_rgb(image: Tensor, clip: bool = True) -> Tensor:
    """Convert a Lab image to RGB.

    Args:
        image (Tensor):
            Lab image to be converted to RGB with shape [*, 3, H, W].
        clip (bool:
            Whether to apply clipping to insure output RGB values in range
            [0.0 1.0].

    Returns:
        lab (Tensor):
            Lab version of the image with shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = lab_to_rgb(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    L  = image[..., 0, :, :]
    a  = image[..., 1, :, :]
    _b = image[..., 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (_b / 200.0)

    # If color data out of range: Z < 0
    fz   = fz.clamp(min=0.0)
    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz   = torch.where(fxyz > 0.2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype
    )[..., :, None, None]
    xyz_im  = xyz * xyz_ref_white
    rgbs_im = xyz_to_rgb(xyz_im)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    #     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    # Clip to [0.0, 1.0] https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0.0, max=1.0)

    return rgb_im


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="lab_to_rgb")
class LabToRgb(torch.nn.Module):
    """Convert an image from Lab to RGB.
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> rgb    = LabToRgb()
        >>> output = rgb(input)  # [2, 3, 4, 5]

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1518
    """

    def forward(self, image: Tensor, clip: bool = True) -> Tensor:
        return lab_to_rgb(image, clip)
