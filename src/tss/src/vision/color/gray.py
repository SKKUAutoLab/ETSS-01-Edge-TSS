#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Grayscale color space.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.factory import TRANSFORMS
from .rgb import bgr_to_rgb

__all__ = [
    "bgr_to_grayscale",
    "BgrToGrayscale",
    "grayscale_to_rgb",
    "GrayscaleToRgb",
    "rgb_to_grayscale",
    "RgbToGrayscale"
]


# MARK: - BgrToGrayscale

def bgr_to_grayscale(image: Tensor) -> Tensor:
    """Convert a BGR image to grayscale. Image data is assumed to be in the
    range of [0.0, 1.0]. First flips to RGB, then converts.

    Args:
        image (Tensor):
            BGR image to be converted to grayscale with shape [*, 3, H, W].

    Returns:
        grayscale (Tensor):
            Grayscale version of the image with shape [*, 1, H, W].

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray  = bgr_to_grayscale(input) # [2, 1, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    image_rgb = bgr_to_rgb(image)
    return rgb_to_grayscale(image_rgb)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="bgr_to_grayscale")
class BgrToGrayscale(torch.nn.Module):
    """Module to convert a BGR image to grayscale version of image. Image
    data is assumed to be in the range of [0.0, 1.0]. First flips to RGB, then
    converts.

    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 1, H, W]

    Reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> gray   = BgrToGrayscale()
        >>> output = gray(input)  # [2, 1, 4, 5]
    """

    def forward(self, image: Tensor) -> Tensor:
        return bgr_to_grayscale(image)


# MARK: - GrayscaleToRgb

def grayscale_to_rgb(image: Tensor) -> Tensor:
    """Convert a grayscale image to RGB version of image. Image data is
    assumed to be in the range of [0.0, 1.0].

    Args:
        image (Tensor):
            Grayscale image to be converted to RGB with shape [*, 1, H, W].

    Returns:
        rgb (Tensor):
            RGB version of the image with shape [*, 3, H, W].

    Example:
        >>> input = torch.randn(2, 1, 4, 5)
        >>> gray  = grayscale_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if image.dim() < 3 or image.size(-3) != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). "
                         f"Got: {image.shape}.")
    
    rgb = torch.cat([image, image, image], dim=-3)

    # TODO: we should find a better way to raise this kind of warnings
    # if not torch.is_floating_point(image):
    #     warnings.warn(f"Input image is not of float dtype. Got: {image.dtype}")

    return rgb


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="grayscale_to_rgb")
class GrayscaleToRgb(torch.nn.Module):
    """Module to convert a grayscale image to RGB version of image. Image
    data is assumed to be in the range of [0.0, 1.0].

    Shape:
        - image:  [*, 1, H, W]
        - output: [*, 3, H, W]

    Reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> input  = torch.rand(2, 1, 4, 5)
        >>> rgb    = GrayscaleToRgb()
        >>> output = rgb(input)  # [2, 3, 4, 5]
    """
    
    def forward(self, image: Tensor) -> Tensor:
        return grayscale_to_rgb(image)


# MARK: - RgbToGrayscale

def rgb_to_grayscale(
    image      : Tensor,
    rgb_weights: Tensor = torch.tensor([0.299, 0.587, 0.114])
) -> Tensor:
    """Convert an RGB image to grayscale version of image. Image data is
    assumed to be in the range of [0.0, 1.0].

    Args:
        image (Tensor):
            RGB image to be converted to grayscale with shape [*, 3, H, W].
        rgb_weights (Tensor):
            Weights that will be applied on each channel (RGB). Fsum of the
            weights should add up to one.
    
    Returns:
        grayscale (Tensor):
            Grayscale version of the image with shape [*, 1, H, W].

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")
    if not isinstance(rgb_weights, Tensor):
        raise TypeError(f"rgb_weights is not a Tensor. "
                        f"Got: {type(rgb_weights)}")
    if rgb_weights.shape[-1] != 3:
        raise ValueError(f"rgb_weights must have a shape of [*, 3]. "
                         f"Got: {rgb_weights.shape}")

    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    if not torch.is_floating_point(image) and (image.dtype != rgb_weights.dtype):
        raise TypeError(
            f"Input image and rgb_weights should be of same dtype. "
            f"Got: {image.dtype} and {rgb_weights.dtype}"
        )

    w_r, w_g, w_b = rgb_weights.to(image).unbind()
    return w_r * r + w_g * g + w_b * b


@TRANSFORMS.register(name="rgb_to_grayscale")
class RgbToGrayscale(torch.nn.Module):
    """Module to convert a RGB image to grayscale version of image. Image
    data is assumed to be in the range of [0.0, 1.0].

    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 1, H, W]

    Reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> gray   = RgbToGrayscale()
        >>> output = gray(input)  # [2, 1, 4, 5]
    """
    
    def __init__(
        self, rgb_weights: Tensor = torch.tensor([0.299, 0.587, 0.114])
    ):
        super().__init__()
        self.rgb_weights = rgb_weights

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_grayscale(image, rgb_weights=self.rgb_weights)
