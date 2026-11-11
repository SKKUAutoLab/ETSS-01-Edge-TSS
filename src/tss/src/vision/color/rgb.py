#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RGB color space.
"""

from __future__ import annotations

from typing import cast
from typing import Union

import torch
from torch import Tensor

from src.factory import TRANSFORMS

__all__ = [
    "bgr_to_rgb",
    "bgr_to_rgba",
    "BgrToRgb",
    "BgrToRgba",
    "linear_rgb_to_rgb",
    "LinearRgbToRgb",
    "rgb_to_bgr",
    "rgb_to_linear_rgb",
    "rgb_to_rgba",
    "rgba_to_bgr",
    "rgba_to_rgb",
    "RgbaToBgr",
    "RgbaToRgb",
    "RgbToBgr",
    "RgbToLinearRgb",
    "RgbToRgba"
]


# MARK: - BgrToRgb

def bgr_to_rgb(image: Tensor) -> Tensor:
    """Convert a BGR image to RGB.

    Args:
        image (Tensor):
            BGR Image to be converted to BGR of shape [*, 3, H, W].

    Returns:
        rgb (Tensor):
            RGB version of the image with shape of shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = bgr_to_rgb(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    # Flip image channels
    out = image.flip(-3)
    return out


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="bgr_to_rgb")
class BgrToRgb(torch.nn.Module):
    """Convert image from BGR to RGB. Image data is assumed to be in the
    range of [0.0, 1.0].
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> rgb    = BgrToRgb()
        >>> output = rgb(input)  # [2, 3, 4, 5]
    """

    def forward(self, image: Tensor) -> Tensor:
        return bgr_to_rgb(image)


# MARK: - BgrToRgba

def bgr_to_rgba(image: Tensor, alpha_val: Union[float, Tensor]) -> Tensor:
    """Convert an image from BGR to RGBA.

    Args:
        image (Tensor):
            BGR Image to be converted to RGBA of shape [*, 3, H, W].
        alpha_val (float, Tensor):
            A float number for the alpha value or a image of shape
            [*, 1, H, W].

    Returns:
        rgba (Tensor):
            RGBA version of the image with shape [*, 4, H, W].

    Notes:
        Current functionality is NOT supported by Torchscript.

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = bgr_to_rgba(input, 1.) # [2, 4, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")
    if not isinstance(alpha_val, (float, Tensor)):
        raise TypeError(f"alpha_val type is not a float or Tensor. "
                        f"Got: {type(alpha_val)}")

    # Convert first to RGB, then add alpha channel
    x_rgb = bgr_to_rgb(image)
    return rgb_to_rgba(x_rgb, alpha_val)


@TRANSFORMS.register(name="bgr_to_rgba")
class BgrToRgba(torch.nn.Module):
    """Convert an image from BGR to RGBA. Add an alpha channel to existing RGB
    image.

    Args:
        alpha_val (float, Tensor):
            A float number for the alpha value or a image of shape
            [*, 1, H, W].
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 4, H, W]

    Notes:
        Current functionality is NOT supported by Torchscript.

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> rgba   = BgrToRgba(1.)
        >>> output = rgba(input)  # [2, 3, 4, 5]
    """

    def __init__(self, alpha_val: Union[float, Tensor]) -> None:
        super().__init__()
        self.alpha_val = alpha_val

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_rgba(image, self.alpha_val)


# MARK: - LinearRgbToRgb

def linear_rgb_to_rgb(image: Tensor) -> Tensor:
    """Convert a linear RGB image to sRGB. Used in colorspace conversions.

    Args:
        image (Tensor):
            Linear RGB Image to be converted to sRGB of shape [*, 3, H, W].

    Returns:
        rgb (Tensor):
            sRGB version of the image with shape of shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = linear_rgb_to_rgb(input) # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    threshold = 0.0031308
    rgb = torch.where(
        image > threshold,
        1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055,
        12.92 * image
    )
    return rgb


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="linear_rgb_to_rgb")
class LinearRgbToRgb(torch.nn.Module):
    """Convert a linear RGB image to sRGB. Applies gamma correction to linear
    RGB values, at the end of colorspace conversions, to get sRGB.
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> srgb   = LinearRgbToRgb()
        >>> output = srgb(input)  # [2, 3, 4, 5]

    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb
        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
        [3] https://en.wikipedia.org/wiki/SRGB
    """

    def forward(self, image: Tensor) -> Tensor:
        return linear_rgb_to_rgb(image)


# MARK: - RgbToBgr

def rgb_to_bgr(image: Tensor) -> Tensor:
    """Convert an RGB image to BGR.

    Args:
        image (orch.Tensor):
            RGB Image to be converted to BGRof of shape [*, 3, H, W].

    Returns:
        bgr (Tensor):
            BGR version of the image with shape of shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_bgr(input) # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    return bgr_to_rgb(image)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rgb_to_bgr")
class RgbToBgr(torch.nn.Module):
    """Convert an image from RGB to BGR. Image data is assumed to be in the
    range of [0.0, 1.0].
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> bgr = RgbToBgr()
        >>> output = bgr(input)  # [2, 3, 4, 5]
    """

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_bgr(image)


# MARK: - RgbToLinearRgb

def rgb_to_linear_rgb(image: Tensor) -> Tensor:
    """Convert an sRGB image to linear RGB. Used in colorspace conversions.

    Args:
        image (Tensor):
            sRGB Image to be converted to linear RGB of shape [*, 3, H, W].

    Returns:
        linear_rgb (Tensor):
            linear RGB version of the image with shape of [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_linear_rgb(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    lin_rgb = torch.where(image > 0.04045,
                          torch.pow(((image + 0.055) / 1.055), 2.4),
                          image / 12.92)
    return lin_rgb


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rgb_to_linear_rgb")
class RgbToLinearRgb(torch.nn.Module):
    """Convert an image from sRGB to linear RGB. Reverses the gamma correction
    of sRGB to get linear RGB values for colorspace conversions. Image data
    is assumed to be in the range of [0.0, 1.0].
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Example:
        >>> input   = torch.rand(2, 3, 4, 5)
        >>> rgb_lin = RgbToLinearRgb()
        >>> output  = rgb_lin(input)  # [2, 3, 4, 5]

    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb
        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
        [3] https://en.wikipedia.org/wiki/SRGB
    """

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_linear_rgb(image)


# MARK: - RgbToRgba

def rgb_to_rgba(image: Tensor, alpha_val: Union[float, Tensor]) -> Tensor:
    """Convert an image from RGB to RGBA.

    Args:
        image (Tensor):
            RGB Image to be converted to RGBA of shape [*, 3, H, W].
        alpha_val (float, Tensor):
            A float number for the alpha value or a image of shape
            [*, 1, H, W].

    Returns:
        rgba (Tensor):
            RGBA version of the image with shape [*, 4, H, W].

    Notes:
        Current functionality is NOT supported by Torchscript.

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_rgba(input, 1.) # [2, 4, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")
    if not isinstance(alpha_val, (float, Tensor)):
        raise TypeError(f"alpha_val type is not a float or Tensor. "
                        f"Got: {type(alpha_val)}")

    # Add one channel
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)
    a       = cast(Tensor, alpha_val)

    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))

    return torch.cat([r, g, b, a], dim=-3)


@TRANSFORMS.register(name="rgb_to_rgba")
class RgbToRgba(torch.nn.Module):
    """Convert an image from RGB to RGBA. Add an alpha channel to existing RGB
    image.

    Args:
        alpha_val (float, Tensor):
            A float number for the alpha value or a image of shape
            [*, 1, H, W].
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 4, H, W]

    Notes:
        Current functionality is NOT supported by Torchscript.

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> rgba   = RgbToRgba(1.)
        >>> output = rgba(input)  # [2, 4, 4, 5]
    """

    def __init__(self, alpha_val: Union[float, Tensor]) -> None:
        super().__init__()
        self.alpha_val = alpha_val

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_rgba(image, self.alpha_val)


# MARK: - RgbaToBgr

def rgba_to_bgr(image: Tensor) -> Tensor:
    """Convert an image from RGBA to BGR.

    Args:
        image (Tensor):
            RGBA Image to be converted to BGR of shape [*, 4, H, W].

    Returns:
        rgb (Tensor):
            RGB version of the image with shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 4, 4, 5)
        >>> output = rgba_to_bgr(input) # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f"Input size must have a shape of [*, 4, H, W]. "
                         f"Got: {image.shape}")

    # Convert to RGB first, then to BGR
    x_rgb = rgba_to_rgb(image)
    return rgb_to_bgr(x_rgb)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rgba_to_bgr")
class RgbaToBgr(torch.nn.Module):
    """Convert an image from RGBA to BGR. Remove an alpha channel from BGR
    image.
 
    Shape:
        - image:  [*, 4, H, W]
        - output: [*, 3, H, W]

    Example:
        >>> input  = torch.rand(2, 4, 4, 5)
        >>> rgba   = RgbaToBgr()
        >>> output = rgba(input)  # [2, 4, 4, 5]
    """

    def forward(self, image: Tensor) -> Tensor:
        return rgba_to_bgr(image)
    

# MARK: - RgbaToRgb

def rgba_to_rgb(image: Tensor) -> Tensor:
    """Convert an image from RGBA to RGB.

    Args:
        image (Tensor):
            RGBA Image to be converted to RGB of shape [*, 4, H, W].

    Returns:
        rgb (Tensor):
            RGB version of the image with shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 4, 4, 5)
        >>> output = rgba_to_rgb(input) # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f"Input size must have a shape of [*, 4, H, W]. "
                         f"Got: {image.shape}")

    # Unpack channels
    r, g, b, a = torch.chunk(image, image.shape[-3], dim=-3)

    # Compute new channels
    a_one = torch.tensor(1.0) - a
    r_new = a_one * r + a * r
    g_new = a_one * g + a * g
    b_new = a_one * b + a * b

    return torch.cat([r_new, g_new, b_new], dim=-3)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rgba_to_rgb")
class RgbaToRgb(torch.nn.Module):
    """Convert an image from RGBA to RGB. Remove an alpha channel from RGB
    image.
 
    Shape:
        - image:  [*, 4, H, W]
        - output: [*, 3, H, W]

    Example:
        >>> input  = torch.rand(2, 4, 4, 5)
        >>> rgba   = RgbaToRgb()
        >>> output = rgba(input)  # [2, 3, 4, 5]
    """

    def forward(self, image: Tensor) -> Tensor:
        return rgba_to_rgb(image)
