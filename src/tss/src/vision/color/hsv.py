#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HSV color space.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from src.factory import TRANSFORMS

__all__ = [
    "hsv_to_rgb",
    "HsvToRgb",
    "rgb_to_hsv",
    "RgbToHsv"
]


# MARK: - HsvToRgb

def hsv_to_rgb(image: Tensor) -> Tensor:
    """Convert an image from HSV to RGB. FH channel values are assumed to be
    in the range [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Args:
        image (Tensor):
            HSV Image to be converted to HSV with shape of [*, 3, H, W].

    Returns:
        rgb (Tensor):
            RGB version of the image with shape of [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    h   = image[..., 0, :, :] / (2 * math.pi)
    s   = image[..., 1, :, :]
    v   = image[..., 2, :, :]

    hi  = torch.floor(h * 6) % 6
    f   = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p   = v * (one - s)
    q   = v * (one - f * s)
    t   = v * (one - (one - f) * s)

    hi      = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    
    out = torch.stack((
        v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q
    ), dim=-3)
    out = torch.gather(out, -3, indices)
    return out


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="hsv_to_rgb")
class HsvToRgb(torch.nn.Module):
    """Convert an image from HSV to RGB. H channel values are assumed to be in
    the range [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> rgb    = HsvToRgb()
        >>> output = rgb(input)  # [2, 3, 4, 5]
    """

    def forward(self, image: Tensor) -> Tensor:
        return hsv_to_rgb(image)


# MARK: - RgbToHsv

def rgb_to_hsv(image: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert an image from RGB to HSV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (Tensor):
            RGB Image to be converted to HSV with shape of [*, 3, H, W].
        eps (float):
            Scalar to enforce numarical stability.

    Returns:
        hsv (Tensor):
            HSV version of the image with shape of [*, 3, H, W]. FH channel
            values are in the range [0.0 2pi]. S and V are in the range
            [0.0, 1.0].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac              = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac     = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = (bc - gc)
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h  = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h  = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h  = (h / 6.0) % 1.0
    h *= 2.0 * math.pi  # We return 0/2pi output

    return torch.stack((h, s, v), dim=-3)


@TRANSFORMS.register(name="rgb_to_hsv")
class RgbToHsv(torch.nn.Module):
    """Convert an image from RGB to HSV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        eps (float):
            Scalar to enforce numarical stability.

    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> hsv    = RgbToHsv()
        >>> output = hsv(input)  # [2, 3, 4, 5]
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_hsv(image, self.eps)
