#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor

from src.factory import TRANSFORMS
from src.type import ScalarOrTuple3T
from src.type import to_3tuple
from .filter import filter2d
from .filter import filter3d
from .kernels_geometry import get_motion_kernel2d
from .kernels_geometry import get_motion_kernel3d

__all__ = [
    "motion_blur",
    "motion_blur3d",
    "MotionBlur",
    "MotionBlur3D"
]


# MARK: - MotionBlur

def motion_blur(
    image      : Tensor,
    kernel_size: int,
    angle      : Union[float, Tensor],
    direction  : Union[float, Tensor],
    border_type: str = "constant",
    mode       : str = "nearest",
) -> Tensor:
    """Perform motion blur on images.

    Args:
        image (Tensor):
            Input image with shape [B, C, H, W].
        kernel_size (int):
            Motion kernel width and height. It should be odd and positive.
        angle (Tensor, float):
            Angle of the motion blur in degrees (anti-clockwise rotation).
            If image, it must be [B, ].
        direction (Tensor, float):
            Forward/backward direction of the motion blur. Lower values towards
            -1.0 will point the motion blur towards the back (with angle
            provided via angle), while higher values towards 1.0 will point the
            motion blur forward. A value of 0.0 leads to a uniformly (but still
            angled) motion blur. If image, it must be [B, ].
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, `circular`].
            Default: `constant`.
        mode (str):
            Interpolation mode for rotating the kernel. One of:
            [`bilinear`, `nearest`].

    Return:
        (Tensor):
            Fblurred image with shape [B, C, H, W].

    Example:
        >>> input = torch.randn(1, 3, 80, 90).repeat(2, 1, 1, 1)
        >>> # perform exact motion blur across the batch
        >>> out_1 = motion_blur(input, 5, 90., 1)
        >>> torch.allclose(out_1[0], out_1[1])
        True
        >>> # perform element-wise motion blur across the batch
        >>> out_1 = motion_blur(input, 5, Tensor([90., 180,]), Tensor([1., -1.]))
        >>> torch.allclose(out_1[0], out_1[1])
        False
    """
    if border_type not in ["constant", "reflect", "replicate", "circular"]:
        raise AssertionError
    kernel = get_motion_kernel2d(kernel_size, angle, direction, mode)
    return filter2d(image, kernel, border_type)


@TRANSFORMS.register(name="motion_blur")
class MotionBlur(torch.nn.Module):
    """Blur 2D images (4D image) using the motion filter.

    Args:
        kernel_size (int):
            Motion kernel width and height. It should be odd and positive.
        angle (Tensor, float):
            Angle of the motion blur in degrees (anti-clockwise rotation).
            If image, it must be [B, ].
        direction (Tensor, float):
            Forward/backward direction of the motion blur. Lower values towards
            -1.0 will point the motion blur towards the back (with angle
            provided via angle), while higher values towards 1.0 will point the
            motion blur forward. A value of 0.0 leads to a uniformly (but still
            angled) motion blur. If image, it must be [B, ].
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, `circular`].
            Default: `constant`.
     
    Returns:
        the blurred input image.

    Shape:
        - Input:  [B, C, H, W]
        - Output: [B, C, H, W]

    Examples:
        >>> input       = torch.rand(2, 4, 5, 7)
        >>> motion_blur = MotionBlur(3, 35., 0.5)
        >>> output      = motion_blur(input)  # 2x4x5x7
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        kernel_size: int,
        angle      : float,
        direction  : float,
        border_type: str = "constant"
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.angle       = angle
        self.direction   = direction
        self.border_type = border_type

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__} (kernel_size={self.kernel_size}, '
            f'angle={self.angle}, direction={self.direction}, border_type={self.border_type})'
        )
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return motion_blur(
            image, self.kernel_size, self.angle, self.direction, self.border_type
        )


# MARK: - MotionBlur3D

def motion_blur3d(
    image      : Tensor,
    kernel_size: int,
    angle      : Union[tuple[float, float, float], Tensor],
    direction  : Union[float, Tensor],
    border_type: str = "constant",
    mode       : str = "nearest",
) -> Tensor:
    """Perform motion blur on 3D volumes (5D image).

    Args:
        image (Tensor):
            Input image with shape [B, C, D, H, W].
        kernel_size (int):
            Motion kernel width, height and depth. It should be odd and
            positive.
        angle (tuple[float, float, float], Tensor):
            Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If image, it must be [B, 3].
        direction (float, Tensor):
            Forward/backward direction of the motion blur. Lower values towards
            -1.0 will point the motion blur towards the back (with angle
            provided via angle), while higher values towards 1.0 will point the
            motion blur forward. A value of 0.0 leads to a uniformly (but still
            angled) motion blur. If image, it must be [B, ].
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, `circular`].
            Default: `constant`.
        mode (str):
            Interpolation mode for rotating the kernel. One of:
            [`bilinear`, `nearest`].

    Return:
        (Tensor):
            Fblurred image with shape [B, C, D, H, W].

    Example:
        >>> input = torch.randn(1, 3, 120, 80, 90).repeat(2, 1, 1, 1, 1)
        >>> # perform exact motion blur across the batch
        >>> out_1 = motion_blur3d(input, 5, (0., 90., 90.), 1)
        >>> torch.allclose(out_1[0], out_1[1])
        True
        >>> # perform element-wise motion blur across the batch
        >>> out_1 = motion_blur3d(input, 5, Tensor([[0., 90., 90.], [90., 180., 0.]]), Tensor([1., -1.]))
        >>> torch.allclose(out_1[0], out_1[1])
        False
    """
    if border_type not in ["constant", "reflect", "replicate", "circular"]:
        raise AssertionError
    kernel = get_motion_kernel3d(kernel_size, angle, direction, mode)
    return filter3d(image, kernel, border_type)


@TRANSFORMS.register(name="motion_blur3d")
class MotionBlur3D(torch.nn.Module):
    """Blur 3D volumes (5D image) using the motion filter.

    Args:
        kernel_size (int):
            Motion kernel width, height and depth. It should be odd and
            positive.
        angle (ScalarOrTuple3T[float]):
            Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If image, it must be [B, 3].
        direction (float, Tensor):
            Forward/backward direction of the motion blur. Lower values towards
            -1.0 will point the motion blur towards the back (with angle
            provided via angle), while higher values towards 1.0 will point the
            motion blur forward. A value of 0.0 leads to a uniformly (but still
            angled) motion blur. If image, it must be [B, ].
        border_type (str):
            Padding mode to be applied before convolving. One of:
            [`constant`, `reflect`, `replicate`, `circular`].
            Default: `constant`.
 
    Shape:
        - Input:  [B, C, D, H, W]
        - Output: [B, C, D, H, W

    Examples:
        >>> input       = torch.rand(2, 4, 5, 7, 9)
        >>> motion_blur = MotionBlur3D(3, 35., 0.5)
        >>> output      = motion_blur(input)  # 2x4x5x7x9
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        kernel_size: int,
        angle      : ScalarOrTuple3T[float],
        direction  : float,
        border_type: str = "constant",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.angle       = to_3tuple(angle)
        self.direction   = direction
        self.border_type = border_type

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__} (kernel_size={self.kernel_size}, '
            f'angle={self.angle}, direction={self.direction}, border_type={self.border_type})'
        )
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return motion_blur3d(
            image, self.kernel_size, self.angle, self.direction, self.border_type
        )
