#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.factory import TRANSFORMS
from .kernels import get_spatial_gradient_kernel2d
from .kernels import get_spatial_gradient_kernel3d
from .kernels import normalize_kernel2d

__all__ = [
    "sobel",
    "Sobel",
    "spatial_gradient",
    "spatial_gradient3d",
    "SpatialGradient",
    "SpatialGradient3d"
]


# MARK: - Sobel

def sobel(image: Tensor, normalized: bool = True, eps: float = 1e-6) -> Tensor:
    """Compute the Sobel operator and returns the magnitude per channel.

    Args:
        image (Tensor): 
            Input image with shape [B, C, H, W].
        normalized (bool):
            If `True`, L1 norm of the kernel is set to `1`.
        eps (float):
            Regularization number to avoid NaN during backprop.

    Return:
        (Tensor):
            Fsobel edge gradient magnitudes map with shape [B, C, H, W].

    Example:
        >>> input  = torch.rand(1, 3, 4, 4)
        >>> output = sobel(input)  # [1, 3, 4, 4]
        >>> output.shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if not len(image.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect [B, C, H, W]. "
                         f"Got: {image.shape}")
    
    # Compute the x/y gradients
    edges = spatial_gradient(image, normalized=normalized)
    
    # Unpack the edges
    gx = edges[:, :, 0]
    gy = edges[:, :, 1]
    
    # Compute gradient magnitude
    magnitude = torch.sqrt(gx * gx + gy * gy + eps)
    return magnitude


@TRANSFORMS.register(name="sobel")
class Sobel(nn.Module):
    """Compute the Sobel operator and returns the magnitude per channel.

    Args:
        normalized (bool):
            If `True`, L1 norm of the kernel is set to `1`.
        eps (float):
            Regularization number to avoid NaN during backprop.

    Shape:
        - Input:  [B, C, H, W]
        - Output: [B, C, H, W]

    Examples:
        >>> input  = torch.rand(1, 3, 4, 4)
        >>> output = Sobel()(input)  # [1, 3, 4, 4]
    """
    
    # MARK: Magic Functions
    
    def __init__(self, normalized: bool = True, eps: float = 1e-6):
        super().__init__()
        self.normalized = normalized
        self.eps = eps
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} + " \
               f"(normalized={str(self.normalized)})"
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return sobel(image, self.normalized, self.eps)


# MARK: - SpatialGradient

def spatial_gradient(
    image     : Tensor,
    mode      : str  = "sobel",
    order     : int  = 1,
    normalized: bool = True
) -> Tensor:
    """Compute the first order image derivative in both x and y using a Sobel
    operator.

    Args:
        image (Tensor):
            Input image with shape [B, C, H, W].
        mode (str):
            Derivatives modality. One of: [`sobel`, `diff`]. Default: `sobel`.
        order (int):
            Order of the derivatives. Default: `1`.
        normalized (bool):
            Whether the output is normalized. Default: `True`.

    Return:
        (Tensor):
            Derivatives of the input feature map with shape [B, C, 2, H, W].
 
    Examples:
        >>> input  = torch.rand(1, 3, 4, 4)
        >>> output = spatial_gradient(input)  # 1x3x2x4x4
        >>> output.shape
        torch.Size([1, 3, 2, 4, 4])
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if not len(image.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect [B, C, H, W]. "
                         f"Got: {image.shape}")
    
    # Allocate kernel
    kernel = get_spatial_gradient_kernel2d(mode, order)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # Prepare kernel
    b, c, h, w = image.shape
    tmp_kernel = kernel.to(image).detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)

    # Convolve input image with sobel kernel
    kernel_flip = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad  = [kernel.size(1) // 2, kernel.size(1) // 2,
                    kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels = 3 if order == 2 else 2
    padded_inp   = F.pad(
        image.reshape(b * c, 1, h, w), spatial_pad, "replicate"
    )[:, :, None]

    return F.conv3d(
        padded_inp, kernel_flip, padding=0
    ).view(b, c, out_channels, h, w)


@TRANSFORMS.register(name="spatial_gradient")
class SpatialGradient(nn.Module):
    """Compute the first order image derivative in both x and y using a Sobel
    operator.

    Args:
        mode (str):
            Derivatives modality. One of: [`sobel`, `diff`]. Default: `sobel`.
        order (int):
            Order of the derivatives. Default: `1`.
        normalized (bool):
            Whether the output is normalized. Default: `True`.
 
    Shape:
        - Input:  [B, C, H, W]
        - Output: [B, C, 2, H, W]

    Examples:
        >>> input  = torch.rand(1, 3, 4, 4)
        >>> output = SpatialGradient()(input)  # 1x3x2x4x4
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        mode      : str  = "sobel",
        order     : int  = 1,
        normalized: bool = True
    ):
        super().__init__()
        self.normalized = normalized
        self.order      = order
        self.mode       = mode

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + "("
            "order=" + str(self.order) + ", " + "normalized=" +
            str(self.normalized) + ", " + "mode=" + self.mode + ")"
        )
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return spatial_gradient(image, self.mode, self.order, self.normalized)


# MARK: - SpatialGradient3d

def spatial_gradient3d(
    image: Tensor, mode: str = "diff", order: int = 1
) -> Tensor:
    """Compute the first and second order volume derivative in x, y and d
    using a diff operator.

    Args:
        image (Tensor):
            Input features image with shape [B, C, D, H, W].
        mode (str):
            Derivatives modality. One of: [`sobel`, `diff`]. Default: `sobel`.
        order (int):
            Order of the derivatives. Default: `1`.

    Return:
        out (Tensor):
            Fspatial gradients of the input feature map with shape
            [B, C, 3, D, H, W] or [B, C, 6, D, H, W].

    Examples:
        >>> input  = torch.rand(1, 4, 2, 4, 4)
        >>> output = spatial_gradient3d(input)
        >>> output.shape
        torch.Size([1, 4, 3, 2, 4, 4])
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")

    if not len(image.shape) == 5:
        raise ValueError(f"Invalid input shape, we expect [B, C, D, H, W]. "
                         f"Got: {image.shape}")
    
    b, c, d, h, w = image.shape
    dev           = image.device
    dtype         = image.dtype
    if (mode == "diff") and (order == 1):
        # We go for the special case implementation due to conv3d bad speed
        x      = F.pad(image, 6 * [1], "replicate")
        center = slice(1, -1)
        left   = slice(0, -2)
        right  = slice(2, None)
        out    = torch.empty(b, c, 3, d, h, w, device=dev, dtype=dtype)
        out[..., 0, :, :, :] = (
            x[..., center, center, right] - x[..., center, center, left]
        )
        out[..., 1, :, :, :] = (
            x[..., center, right, center] - x[..., center, left, center]
        )
        out[..., 2, :, :, :] = (
            x[..., right, center, center] - x[..., left, center, center]
        )
        out = 0.5 * out
    else:
        # Prepare kernel
        # Allocate kernel
        kernel     = get_spatial_gradient_kernel3d(mode, order)

        tmp_kernel = kernel.to(image).detach()
        tmp_kernel = tmp_kernel.repeat(c, 1, 1, 1, 1)

        # Convolve input image with grad kernel
        kernel_flip = tmp_kernel.flip(-3)

        # Pad with "replicate for spatial dims, but with zeros for channel
        spatial_pad = [kernel.size(2) // 2,
                       kernel.size(2) // 2,
                       kernel.size(3) // 2,
                       kernel.size(3) // 2,
                       kernel.size(4) // 2,
                       kernel.size(4) // 2]
        out_ch      = 6 if order == 2 else 3
        out         = F.conv3d(
            F.pad(image, spatial_pad, "replicate"), kernel_flip, padding=0,
            groups=c
        ).view(b, c, out_ch, d, h, w)
    return out


@TRANSFORMS.register(name="spatial_gradient3d")
class SpatialGradient3d(nn.Module):
    """Compute the first and second order volume derivative in x, y and d using
    a diff operator.

    Args:
         mode (str):
            Derivatives modality. One of: [`sobel`, `diff`]. Default: `sobel`.
        order (int):
            Order of the derivatives. Default: `1`.

    Shape:
        - Input:  [B, C, D, H, W]. D, H, W are spatial dimensions, gradient is
          calculated w.r.t to them.
        - Output: [B, C, 3, D, H, W] or [B, C, 6, D, H, W]

    Examples:
        >>> input  = torch.rand(1, 4, 2, 4, 4)
        >>> output = SpatialGradient3d()(input)
        >>> output.shape
        torch.Size([1, 4, 3, 2, 4, 4])
    """
    
    # MARK: Magic Functions
    
    def __init__(self, mode: str = "diff", order: int = 1):
        super().__init__()
        self.order  = order
        self.mode   = mode
        self.kernel = get_spatial_gradient_kernel3d(mode, order)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} " \
               f"(order={self.order}, mode={self.mode})"
         
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return spatial_gradient3d(image, self.mode, self.order)
