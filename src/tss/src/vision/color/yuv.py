#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YUV color space.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.factory import TRANSFORMS
from src.type import ListOrTuple2T

__all__ = [
    "rgb_to_yuv",
    "rgb_to_yuv420",
    "rgb_to_yuv422",
    "RgbToYuv",
    "RgbToYuv420",
    "RgbToYuv422",
    "yuv420_to_rgb",
    "Yuv420ToRgb",
    "yuv422_to_rgb",
    "Yuv422ToRgb",
    "yuv_to_rgb",
    "YuvToRgb"
]


# MARK: - RgbToYuv

def rgb_to_yuv(image: Tensor) -> Tensor:
    """Convert an RGB image to YUV. Image data is assumed to be in the 
    range of [0.0, 1.0].

    Args:
        image (Tensor):
            RGB Image to be converted to YUV with shape [*, 3, H, W].

    Returns:
        yuv (Tensor):
            YUV version of the image with shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    y =  0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v =  0.615 * r - 0.515 * g - 0.100 * b

    out = torch.stack([y, u, v], -3)
    return out


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rgb_to_yuv")
class RgbToYuv(torch.nn.Module):
    """Convert an image from RGB to YUV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> yuv    = RgbToYuv()
        >>> output = yuv(input)  # [2, 3, 4, 5]

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_yuv(image)


# MARK: - RgbToYuv420

def rgb_to_yuv420(image: Tensor) -> ListOrTuple2T[Tensor]:
    """Convert an RGB image to YUV 420 (subsampled). Image data is assumed 
    to be in the range of [0.0, 1.0]. Input need to be padded to be evenly 
    divisible by 2 horizontal and vertical. This function will output chroma 
    siting [0.5, 0.5]

    Args:
        image (Tensor):
            RGB Image to be converted to YUV with shape [*, 3, H, W].

    Returns:
        A Tensor containing the Y plane with shape [*, 1, H, W]
        A Tensor containing the UV planes with shape [*, 2, H/2, W/2]

    Example:
        >>> input  = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_yuv420(input)  # ([2, 1, 4, 6], [2, 2, 2, 3])
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")
    if (len(image.shape) < 2 or
        image.shape[-2] % 2 == 1 or
        image.shape[-1] % 2 == 1):
        raise ValueError(f"Input H&W must be evenly divisible by 2. "
                         f"Got: {image.shape}")

    yuvimage = rgb_to_yuv(image)
    return (
        yuvimage[..., :1, :, :],
        torch.nn.functional.avg_pool2d(yuvimage[..., 1:3, :, :], (2, 2))
    )


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rgb_to_yuv420")
class RgbToYuv420(torch.nn.Module):
    """Convert an image from RGB to YUV420. Image data is assumed to be in
    the range of [0.0, 1.0]. Width and Height evenly divisible by 2.

    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 1, H, W] and [*, 2, H/2, W/2]

    Examples:
        >>> yuvinput = torch.rand(2, 3, 4, 6)
        >>> yuv      = RgbToYuv420()
        >>> output   = yuv(yuvinput)  # ([2, 1, 4, 6], [2, 1, 2, 3])

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def forward(self, yuv_input: Tensor) -> ListOrTuple2T[Tensor]:
        return rgb_to_yuv420(yuv_input)


# MARK: - RgbToYuv422

def rgb_to_yuv422(image: Tensor) -> ListOrTuple2T[Tensor]:
    """Convert an RGB image to YUV 422 (subsampled). Image data is assumed
    to be in the range of [0.0, 1.0]. Input need to be padded to be evenly
    divisible by 2 vertical. This function will output chroma siting (0.5)

    Args:
        image (Tensor):
            RGB Image to be converted to YUV with shape [*, 3, H, W].

    Returns:
       A Tensor containing the Y plane with shape [*, 1, H, W].
       A Tensor containing the UV planes with shape [*, 2, H, W/2].

    Example:
        >>> input  = torch.rand(2, 3, 4, 6)
        >>> output = rgb_to_yuv420(input)  # ([2, 1, 4, 6], [2, 1, 4, 3])
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")
    if (len(image.shape) < 2 or
        image.shape[-2] % 2 == 1 or
        image.shape[-1] % 2 == 1):
        raise ValueError(f"Input H&W must be evenly divisible by 2. "
                         f"Got: {image.shape}")

    yuvimage = rgb_to_yuv(image)
    return (
        yuvimage[..., :1, :, :],
        torch.nn.functional.avg_pool2d(yuvimage[..., 1:3, :, :], (1, 2))
    )


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rgb_to_yuv422")
class RgbToYuv422(torch.nn.Module):
    """Convert an image from RGB to YUV422. Image data is assumed to be in
    the range of [0.0, 1.0]. Width evenly disvisible by 2.
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 1, H, W] and [*, 2, H, W/2]

    Examples:
        >>> yuvinput = torch.rand(2, 3, 4, 6)
        >>> yuv      = RgbToYuv422()
        >>> output   = yuv(yuvinput)  # ([2, 1, 4, 6], [2, 2, 4, 3])

    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def forward(self, yuv_input: Tensor) -> ListOrTuple2T[Tensor]:
        return rgb_to_yuv422(yuv_input)


# MARK: - Yuv420ToRgb

def yuv420_to_rgb(image_y: Tensor, image_uv: Tensor) -> Tensor:
    """Convert an YUV420 image to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Input need to be
    padded to be evenly divisible by 2 horizontal and vertical. This function 
    assumed chroma siting is [0.5, 0.5]

    Args:
        image_y (Tensor):
            Y (luma) Image plane to be converted to RGB with shape [*, 1, H, W].
        image_uv (Tensor):
            UV (chroma) Image planes to be converted to RGB with shape
            [*, 2, H/2, W/2].

    Returns:
        rgb (Tensor):
            RGB version of the image with shape [*, 3, H, W].

    Example:
        >>> inputy  = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> output  = yuv420_to_rgb(inputy, inputuv)  # [2, 3, 4, 6]
    """
    if not isinstance(image_y, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image_y)}")
    if not isinstance(image_uv, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image_uv)}")
    if len(image_y.shape) < 3 or image_y.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of "
                         f"[*, 1, H, W]. Got: {image_y.shape}")
    if len(image_uv.shape) < 3 or image_uv.shape[-3] != 2:
        raise ValueError(f"Input imageuv size must have a shape of "
                         f"[*, 2, H/2, W/2]. Got: {image_uv.shape}")
    if (len(image_y.shape) < 2 or
        image_y.shape[-2] % 2 == 1 or
        image_y.shape[-1] % 2 == 1):
        raise ValueError(f"Input H&W must be evenly divisible by 2. "
                         f"Got: {image_y.shape}")
    if (len(image_uv.shape) < 2 or
        len(image_y.shape) < 2 or
        image_y.shape[-2] / image_uv.shape[-2] != 2 or
        image_y.shape[-1] / image_uv.shape[-1] != 2):
        raise ValueError(f"Input imageuv H&W must be half the size of the luma "
                         f"plane. Got: {image_y.shape} and {image_uv.shape}")

    # First upsample
    yuv444image = torch.cat([
        image_y,
        image_uv.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    ], dim=-3)
    # Then convert the yuv444 image
    return yuv_to_rgb(yuv444image)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="yuv420_to_rgb")
class Yuv420ToRgb(torch.nn.Module):
    """Convert an image from YUV to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Width and Height
    evenly divisible by 2.

    Shape:
        - imagey:  [*, 1, H, W]
        - imageuv: [*, 2, H/2, W/2]
        - output:  [*, 3, H, W]

    Examples:
        >>> inputy  = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> rgb     = Yuv420ToRgb()
        >>> output  = rgb(inputy, inputuv)  # [2, 3, 4, 6]
    """
    
    def forward(self, input_y: Tensor, input_uv: Tensor) -> Tensor:  # skipcq: PYL-R0201
        return yuv420_to_rgb(input_y, input_uv)


# MARK: - Yuv422ToRgb

def yuv422_to_rgb(image_y: Tensor, image_uv: Tensor) -> Tensor:
    """Convert an YUV422 image to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Input need to be
    padded to be evenly divisible by 2 vertical. This function assumed chroma
    siting is (0.5)

    Args:
        image_y (Tensor):
            Y (luma) Image plane to be converted to RGB with shape [*, 1, H, W].
        image_uv (Tensor):
            UV (luma) Image planes to be converted to RGB with shape
            [*, 2, H, W/2].

    Returns:
        rgb (Tensor):
            RGB version of the image with shape [*, 3, H, W].

    Example:
        >>> inputy  = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 2, 3)
        >>> output  = yuv420_to_rgb(inputy, inputuv)  # [2, 3, 4, 5]
    """
    if not isinstance(image_y, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image_y)}")
    if not isinstance(image_uv, Tensor):
        raise TypeError(f"Input type is not a Tensor. "
                        f"Got: {type(image_uv)}")
    if len(image_y.shape) < 3 or image_y.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of "
                         f"[*, 1, H, W]. Got: {image_y.shape}")
    if len(image_uv.shape) < 3 or image_uv.shape[-3] != 2:
        raise ValueError(f"Input imageuv size must have a shape of "
                         f"[*, 2, H, W/2]. Got: {image_uv.shape}")
    if (len(image_y.shape) < 2 or
        image_y.shape[-2] % 2 == 1 or
        image_y.shape[-1] % 2 == 1):
        raise ValueError(f"Input H&W must be evenly divisible by 2. "
                         f"Got: {image_y.shape}")
    if (len(image_uv.shape) < 2 or
        len(image_y.shape) < 2 or
        image_y.shape[-1] / image_uv.shape[-1] != 2):
        raise ValueError(f"Input imageuv W must be half the size of the luma "
                         f"plane. Got: {image_y.shape} and {image_uv.shape}")

    # First upsample
    yuv444image = torch.cat([
        image_y, image_uv.repeat_interleave(2, dim=-1)
    ], dim=-3)
    # Then convert the yuv444 image
    return yuv_to_rgb(yuv444image)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="yuv422_to_rgb")
class Yuv422ToRgb(torch.nn.Module):
    """Convert an image from YUV to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Width evenly
    divisible by 2.
 
    Shape:
        - imagey:  [*, 1, H, W]
        - imageuv: [*, 2, H, W/2]
        - output:  [*, 3, H, W]

    Examples:
        >>> inputy  = torch.rand(2, 1, 4, 6)
        >>> inputuv = torch.rand(2, 2, 4, 3)
        >>> rgb     = Yuv422ToRgb()
        >>> output  = rgb(inputy, inputuv)  # [2, 3, 4, 6]
    """

    def forward(self, input_y: Tensor, input_uv: Tensor) -> Tensor:
        return yuv422_to_rgb(input_y, input_uv)


# MARK: - YuvToRgb

def yuv_to_rgb(image: Tensor) -> Tensor:
    """Convert an YUV image to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.

    Args:
        image (Tensor):
            YUV Image to be converted to RGB with shape [*, 3, H, W].

    Returns:
        rgb (Tensor):
            RGB version of the image with shape [*, 3, H, W].

    Example:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # [2, 3, 4, 5]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of [*, 3, H, W]. "
                         f"Got: {image.shape}")

    y = image[..., 0, :, :]
    u = image[..., 1, :, :]
    v = image[..., 2, :, :]

    r = y + 1.14 * v  # coefficient for g is 0
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u  # coefficient for b is 0

    out = torch.stack([r, g, b], -3)
    return out


# noinspection PyShadowingBuiltins,PyMethodMayBeStatic
@TRANSFORMS.register(name="yuv_to_rgb")
class YuvToRgb(torch.nn.Module):
    """Convert an image from YUV to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.
 
    Shape:
        - image:  [*, 3, H, W]
        - output: [*, 3, H, W]

    Examples:
        >>> input  = torch.rand(2, 3, 4, 5)
        >>> rgb    = YuvToRgb()
        >>> output = rgb(input)  # [2, 3, 4, 5]
    """

    def forward(self, image: Tensor) -> Tensor:
        return yuv_to_rgb(image)
