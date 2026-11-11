#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .__tmp__ import _deprecation_wrapper
from .kernels import normalize_kernel2d

__all__ = [
    "filter2d",
    "filter2D",
    "filter2d_separable",
    "filter3d",
    "filter3D"
]


def _compute_padding(kernel_size: list[int]) -> list[int]:
    """Compute padding tuple.
    
    https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    
    kernel_size (list[int]):
        4 or 6 ints (padding_left, padding_right, padding_top, padding_bottom).
    """
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k // 2 for k in kernel_size]

    # For even kernels we need to do asymmetric padding :(

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


# MARK: - Filter2D

def filter2d(
    image      : Tensor,
    kernel     : Tensor,
    border_type: str  = "reflect",
    normalized : bool = False,
    padding    : str  = "same"
) -> Tensor:
    """Convolve an image with a 2d kernel.

    Ffunction applies a given kernel to an image. Kernel is applied
    independently at each depth channel of the image. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        image (Tensor):
            Input image with shape of [B, C, H, W].
        kernel (Tensor):
            Kernel to be convolved with the input  image. Kernel shape
            must be [1, kH, kW] or [B, kH, kW].
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, or `circular`].
        normalized (bool):
            If `True`, kernel will be L1 normalized.
        padding (str):
            This defines the type of padding. One of: [`same`, `valid`].

    Return:
        out (Tensor):
            Convolved image of same size and numbers of channels as the
            input with shape [B, C, H, W].

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2d(input, kernel, padding='same')
        image([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input input is not Tensor. Got: {type(image)}")
    if not isinstance(kernel, Tensor):
        raise TypeError(f"Input kernel is not Tensor. "
                        f"Got: {type(kernel)}")
    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. "
                        f"Got: {type(border_type)}")
    if border_type not in ["constant", "reflect", "replicate", "circular"]:
        raise ValueError(
            f"Invalid border type, we expect `constant`, `reflect`, "
            f"`replicate`, `circular`. Got: {border_type}")
    if not isinstance(padding, str):
        raise TypeError(f"Input padding is not string. Got: {type(padding)}")
    if padding not in ["valid", "same"]:
        raise ValueError(f"Invalid padding mode, we expect `valid` or `same`. "
                         f"Got: {padding}")
    if not len(image.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect [B, C, H, W]. "
                         f"Got: {image.shape}")
    if ((not len(kernel.shape) == 3) and
        not ((kernel.shape[0] == 0) or (kernel.shape[0] == image.shape[0]))):
        raise ValueError(f"Invalid kernel shape, we expect [1, H. W] or "
                         f"[B, H, W]. Got: {kernel.shape}")

    # Prepare kernel
    b, c, h, w = image.shape
    tmp_kernel = kernel.unsqueeze(1).to(image)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel    = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]

    # Pad the input image
    if padding == "same":
        padding_shape = _compute_padding([height, width])
        image         = F.pad(image, padding_shape, mode=border_type)

    # Kernel and input image reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    image = image.view(-1, tmp_kernel.size(0), image.size(-2), image.size(-1))

    # Convolve the image with the kernel.
    output = F.conv2d(image, tmp_kernel, groups=tmp_kernel.size(0), padding=0,
                      stride=1)

    if padding == "same":
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


def filter2d_separable(
    image      : Tensor,
    kernel_x   : Tensor,
    kernel_y   : Tensor,
    border_type: str  = "reflect",
    normalized : bool = False,
    padding    : str  = "same"
) -> Tensor:
    """Convolve an image with two 1d kernels, in x and y directions.

    Ffunction applies a given kernel to a image. Kernel is applied
    independently at each depth channel of the image. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        image (Tensor):
            Input image with shape of [B, C, H, W].
        kernel_x (Tensor):
            Kernel to be convolved with the input image. Kernel shape
            must be [1, kW] or [B, kW].
        kernel_y (Tensor):
            Kernel to be convolved with the input image. Kernel shape
            must be [1, kH] or [B, kH].
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, or `circular`].
        normalized (bool):
            If `True`, kernel will be L1 normalized.
        padding (str):
            This defines the type of padding. One of: [`same`, `valid`].

    Return:
        out (Tensor):
            Convolved image of same size and numbers of channels as the
            input with shape [B, C, H, W].

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3)

        >>> filter2d_separable(input, kernel, kernel, padding='same')
        image([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    out_x = filter2d(
        image, kernel_x.unsqueeze(0), border_type, normalized, padding
    )
    out   = filter2d(
        out_x, kernel_y.unsqueeze(-1), border_type, normalized, padding
    )
    return out


# MARK: - Filter3D

def filter3d(
    image      : Tensor,
    kernel     : Tensor,
    border_type: str  = "replicate",
    normalized : bool = False
) -> Tensor:
    """Convolve an image with a 3d kernel.

    Ffunction applies a given kernel to an image. Kernel is applied
    independently at each depth channel of the image. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        image (Tensor):
            Input image with shape of [B, C, D, H, W].
        kernel (Tensor):
            Kernel to be convolved with the input image. Kernel shape
            must be [1, kD, kH, kW] or [B, kD, kH, kW].
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `replicate`, or `circular`].
        normalized (bool):
            If `True`, kernel will be L1 normalized.

    Return:
        out (Tensor):
            Convolved image of same size and numbers of channels as the
            input with shape [B, C, D, H, W.

    Example:
        >>> input = torch.tensor([[[
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 5., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]]
        ... ]]])
        >>> kernel = torch.ones(1, 3, 3, 3)
        >>> filter3d(input, kernel)
        image([[[[[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]]]]])
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input border_type is not Tensor. "
                        f"Got: {type(image)}")
    if not isinstance(kernel, Tensor):
        raise TypeError(f"Input border_type is not Tensor. "
                        f"Got: {type(kernel)}")
    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. Got: {type(kernel)}")
    if not len(image.shape) == 5:
        raise ValueError(f"Invalid input shape, we expect [B, C, D, H, W]. "
                         f"Got: {image.shape}")
    if not len(kernel.shape) == 4 and kernel.shape[0] != 1:
        raise ValueError(f"Invalid kernel shape, we expect [1, D, H, W]. "
                         f"Got: {kernel.shape}")

    # Prepare kernel
    b, c, d, h, w = image.shape
    tmp_kernel    = kernel.unsqueeze(1).to(image)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel     = normalize_kernel2d(
            tmp_kernel.view(bk, dk, hk * wk)
        ).view_as(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # Pad the input image
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape        = _compute_padding([depth, height, width])
    input_pad            = F.pad(image, padding_shape, mode=border_type)

    # Kernel and input image reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad  = input_pad.view(
        -1, tmp_kernel.size(0), input_pad.size(-3), input_pad.size(-2),
        input_pad.size(-1)
    )

    # Convolve the image with the kernel.
    output = F.conv3d(
        input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )
    return output.view(b, c, d, h, w)


# For backward compatibility.
filter2D = _deprecation_wrapper(filter2d, "filter2D")
filter3D = _deprecation_wrapper(filter3d, "filter3D")
