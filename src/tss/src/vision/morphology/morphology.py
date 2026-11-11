#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.factory import TRANSFORMS

__all__ = [
    "_neight2channels_like_kernel",
    "bottom_hat",
    "closing",
    "dilation",
    "erosion",
    "morphology_gradient",
    "opening",
    "top_hat"
]


def _neight2channels_like_kernel(kernel: Tensor) -> Tensor:
    h, w   = kernel.size()
    kernel = torch.eye(h * w, dtype=kernel.dtype, device=kernel.device)
    return kernel.view(h * w, 1, h, w)


# MARK: - Dilation

def dilation(
    image              : Tensor,
    kernel             : Tensor,
    structuring_element: Optional[Tensor]    = None,
    origin             : Optional[list[int]] = None,
    border_type        : str                 = "geodesic",
    border_value       : float               = 0.0,
    max_val            : float               = 1e4,
    engine             : str                 = "unfold",
) -> Tensor:
    """Return the dilated image applying the same kernel in each channel.
    Kernel must have 2 dimensions.

    Args:
        image (Tensor):
            Image with shape [B, C, H, W].
        kernel (Tensor):
            Positions of non-infinite elements of a flat structuring element.
            Non-zero values give the set of neighbors of the center over which 
            the operation is applied. Its shape is [k_x, k_y]. For full
            structural elements use torch.ones_like(structural_element).
        structuring_element (Tensor, optional):
            Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin (list[int], optional):
            Origin of the structuring element. Default: `None` and uses the
            center of the structuring element as origin (rounding towards zero).
        border_type (str):
            It determines how the image borders are handled, where
            `border_value` is the value when `border_type` is equal to
            `constant`. Default: `geodesic` which ignores the values that are
            outside the image when applying the operation.
        border_value (float):
            Value to fill past edges of input if `border_type` is `constant`.
        max_val (float):
            Fvalue of the infinite elements in the kernel.
        engine (str):
            Convolution is faster and less memory hungry, and unfold is more
            stable numerically.

    Returns:
        output (Tensor):
            Dilated image with shape [B, C, H, W].

    Example:
        >>> image      = torch.rand(1, 3, 5, 5)
        >>> kernel      = torch.ones(3, 3)
        >>> dilated_img = dilation(image, kernel)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. "
                        f"Got: {type(image)}")
    if len(image.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. "
                         f"Got: {image.dim()}")
    if not isinstance(kernel, Tensor):
        raise TypeError(f"Kernel type is not a Tensor. "
                        f"Got: {type(kernel)}")
    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. "
                         f"Got: {kernel.dim()}")

    # Origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # Pad
    pad_e = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == "geodesic":
        border_value = -max_val
        border_type  = "constant"
    output = F.pad(image, pad_e, mode=border_type, value=border_value)

    # Computation
    if structuring_element is None:
        neighborhood              = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood              = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    if engine == "unfold":
        output    = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output, _ = torch.max(output + neighborhood.flip((0, 1)), 4)
        output, _ = torch.max(output, 4)
    elif engine == "convolution":
        B, C, H, W     = image.size()
        h_pad, w_pad   = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel)
        output, _      = F.conv2d(
            output.view(B * C, 1, h_pad, w_pad), reshape_kernel, padding=0,
            bias=neighborhood.view(-1).flip(0)
        ).max(dim=1)
        output = output.view(B, C, H, W)
    else:
        raise NotImplementedError(f"engine {engine} is unknown, use "
                                  f"`convolution` or `unfold`")
    return output.view_as(image)


@TRANSFORMS.register(name="dilation")
class Dilation(torch.nn.Module):

    def __init__(
        self,
        kernel             : Tensor,
        structuring_element: Optional[Tensor]    = None,
        origin             : Optional[list[int]] = None,
        border_type        : str                 = "geodesic",
        border_value       : float               = 0.0,
        max_val            : float               = 1e4,
        engine             : str                 = "unfold",
    ):
        super().__init__()
        self.kernel              = kernel
        self.structuring_element = structuring_element
        self.origin              = origin
        self.border_type         = border_type
        self.border_value        = border_value
        self.max_val             = max_val
        self.engine              = engine
    
    def forward(self, image: Tensor) -> Tensor:
        return dilation(
            image, self.kernel, self.structuring_element, self.origin,
            self.border_type, self.border_value, self.max_val, self.engine
        )


# MARK: - Erosion

def erosion(
    image              : Tensor,
    kernel             : Tensor,
    structuring_element: Optional[Tensor]    = None,
    origin             : Optional[list[int]] = None,
    border_type        : str                 = "geodesic",
    border_value       : float               = 0.0,
    max_val            : float               = 1e4,
    engine             : str                 = "unfold",
) -> Tensor:
    """Return the eroded image applying the same kernel in each channel.
    Kernel must have 2 dimensions.

    Args:
        image (Tensor):
            Image with shape [B, C, H, W].
        kernel (Tensor):
            Positions of non-infinite elements of a flat structuring element.
            Non-zero values give the set of neighbors of the center over which
            the operation is applied. Its shape is [k_x, k_y]. For full
            structural elements use torch.ones_like(structural_element).
        structuring_element (Tensor, optional):
            Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin (list[int], optional):
            Origin of the structuring element. Default: `None` and uses the
            center of the structuring element as origin (rounding towards zero).
        border_type (str):
            It determines how the image borders are handled, where
            `border_value` is the value when `border_type` is equal to
            `constant`. Default: `geodesic` which ignores the values that are
            outside the image when applying the operation.
        border_value (float):
            Value to fill past edges of input if `border_type` is `constant`.
        max_val (float):
            Fvalue of the infinite elements in the kernel.
        engine (str):
            Convolution is faster and less memory hungry, and unfold is more
            stable numerically.
        
    Returns:
        output (Tensor):
            Eroded image with shape [B, C, H, W].

    Example:
        >>> image = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(5, 5)
        >>> output = erosion(image, kernel)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. "
                        f"Got: {type(image)}")
    if len(image.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. "
                         f"Got: {image.dim()}")
    if not isinstance(kernel, Tensor):
        raise TypeError(f"Kernel type is not a Tensor. "
                        f"Got: {type(kernel)}")
    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. "
                         f"Got: {kernel.dim()}")

    # Origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # Pad
    pad_e = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == "geodesic":
        border_value = max_val
        border_type  = "constant"
    output = F.pad(image, pad_e, mode=border_type, value=border_value)

    # Computation
    if structuring_element is None:
        neighborhood              = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood              = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    if engine == "unfold":
        output    = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output, _ = torch.min(output - neighborhood, 4)
        output, _ = torch.min(output, 4)
    elif engine == "convolution":
        B, C, H, W     = image.size()
        Hpad, Wpad     = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel)
        output, _      = F.conv2d(
            output.view(B * C, 1, Hpad, Wpad), reshape_kernel, padding=0,
            bias=-neighborhood.view(-1)
        ).min(dim=1)
        output = output.view(B, C, H, W)
    else:
        raise NotImplementedError(f"engine {engine} is unknown, use "
                                  f"`convolution` or `unfold`")

    return output


@TRANSFORMS.register(name="erosion")
class Erosion(torch.nn.Module):

    def __init__(
        self,
        kernel             : Tensor,
        structuring_element: Optional[Tensor]    = None,
        origin             : Optional[list[int]] = None,
        border_type        : str                 = "geodesic",
        border_value       : float               = 0.0,
        max_val            : float               = 1e4,
        engine             : str                 = "unfold",
    ):
        super().__init__()
        self.kernel              = kernel
        self.structuring_element = structuring_element
        self.origin              = origin
        self.border_type         = border_type
        self.border_value        = border_value
        self.max_val             = max_val
        self.engine              = engine
    
    def forward(self, image: Tensor) -> Tensor:
        return erosion(
            image, self.kernel, self.structuring_element, self.origin,
            self.border_type, self.border_value, self.max_val, self.engine
        )


# MARK: - Opening

def opening(
    image              : Tensor,
    kernel             : Tensor,
    structuring_element: Optional[Tensor]    = None,
    origin             : Optional[list[int]] = None,
    border_type        : str                 = "geodesic",
    border_value       : float               = 0.0,
    max_val            : float               = 1e4,
    engine             : str                 = "unfold",
) -> Tensor:
    """Return the opened image, (that means, dilation after an erosion)
    applying the same kernel in each channel. Kernel must have 2 dimensions.

    Args:
        image (Tensor):
            Image with shape [B, C, H, W].
        kernel (Tensor):
            Positions of non-infinite elements of a flat structuring element.
            Non-zero values give the set of neighbors of the center over which
            the operation is applied. Its shape is [k_x, k_y]. For full
            structural elements use torch.ones_like(structural_element).
        structuring_element (Tensor, optional):
            Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin (list[int], optional):
            Origin of the structuring element. Default: `None` and uses the
            center of the structuring element as origin (rounding towards zero).
        border_type (str):
            It determines how the image borders are handled, where
            `border_value` is the value when `border_type` is equal to
            `constant`. Default: `geodesic` which ignores the values that are
            outside the image when applying the operation.
        border_value (float):
            Value to fill past edges of input if `border_type` is `constant`.
        max_val (float):
            Fvalue of the infinite elements in the kernel.
        engine (str):
            Convolution is faster and less memory hungry, and unfold is more
            stable numerically.
        
    Returns:
        output (Tensor):
            Opened image with shape [B, C, H, W].

    Example:
        >>> image     = torch.rand(1, 3, 5, 5)
        >>> kernel     = torch.ones(3, 3)
        >>> opened_img = opening(image, kernel)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. "
                        f"Got: {type(image)}")
    if len(image.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. "
                         f"Got: {image.dim()}")
    if not isinstance(kernel, Tensor):
        raise TypeError(f"Kernel type is not a Tensor. "
                        f"Got: {type(kernel)}")
    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. "
                         f"Got: {kernel.dim()}")

    return dilation(
        erosion(
            image,
            kernel              = kernel,
            structuring_element = structuring_element,
            origin              = origin,
            border_type         = border_type,
            border_value        = border_value,
            max_val             = max_val,
        ),
        kernel              = kernel,
        structuring_element = structuring_element,
        origin              = origin,
        border_type         = border_type,
        border_value        = border_value,
        max_val             = max_val,
        engine              = engine,
    )


@TRANSFORMS.register(name="opening")
class Opening(torch.nn.Module):

    def __init__(
        self,
        kernel             : Tensor,
        structuring_element: Optional[Tensor]    = None,
        origin             : Optional[list[int]] = None,
        border_type        : str                 = "geodesic",
        border_value       : float               = 0.0,
        max_val            : float               = 1e4,
        engine             : str                 = "unfold",
    ):
        super().__init__()
        self.kernel              = kernel
        self.structuring_element = structuring_element
        self.origin              = origin
        self.border_type         = border_type
        self.border_value        = border_value
        self.max_val             = max_val
        self.engine              = engine
    
    def forward(self, image: Tensor) -> Tensor:
        return opening(
            image, self.kernel, self.structuring_element, self.origin,
            self.border_type, self.border_value, self.max_val, self.engine
        )


# MARK: - Closing

def closing(
    image              : Tensor,
    kernel             : Tensor,
    structuring_element: Optional[Tensor]    = None,
    origin             : Optional[list[int]] = None,
    border_type        : str                 = "geodesic",
    border_value       : float               = 0.0,
    max_val            : float               = 1e4,
    engine             : str                 = "unfold",
) -> Tensor:
    r"""Return the closed image, (that means, erosion after a dilation)
    applying the same kernel in each channel. Kernel must have 2 dimensions.

    Args:
        image (Tensor):
            Image with shape [B, C, H, W].
        kernel (Tensor):
            Positions of non-infinite elements of a flat structuring element.
            Non-zero values give the set of neighbors of the center over which
            the operation is applied. Its shape is [k_x, k_y]. For full
            structural elements use torch.ones_like(structural_element).
        structuring_element (Tensor, optional):
            Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin (list[int], optional):
            Origin of the structuring element. Default: `None` and uses the
            center of the structuring element as origin (rounding towards zero).
        border_type (str):
            It determines how the image borders are handled, where
            `border_value` is the value when `border_type` is equal to
            `constant`. Default: `geodesic` which ignores the values that are
            outside the image when applying the operation.
        border_value (float):
            Value to fill past edges of input if `border_type` is `constant`.
        max_val (float):
            Fvalue of the infinite elements in the kernel.
        engine (str):
            Convolution is faster and less memory hungry, and unfold is more
            stable numerically.
        
    Returns:
        output (Tensor):
            Closed image with shape [B, C, H, W].

    Example:
        >>> image     = torch.rand(1, 3, 5, 5)
        >>> kernel     = torch.ones(3, 3)
        >>> closed_img = closing(image, kernel)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. "
                        f"Got: {type(image)}")
    if len(image.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. "
                         f"Got: {image.dim()}")
    if not isinstance(kernel, Tensor):
        raise TypeError(f"Kernel type is not a Tensor. "
                        f"Got: {type(kernel)}")
    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. "
                         f"Got: {kernel.dim()}")

    return erosion(
        dilation(
            image,
            kernel              = kernel,
            structuring_element = structuring_element,
            origin              = origin,
            border_type         = border_type,
            border_value        = border_value,
            max_val             = max_val,
        ),
        kernel              = kernel,
        structuring_element = structuring_element,
        origin              = origin,
        border_type         = border_type,
        border_value        = border_value,
        max_val             = max_val,
        engine              = engine,
    )


@TRANSFORMS.register(name="closing")
class Closing(torch.nn.Module):

    def __init__(
        self,
        kernel             : Tensor,
        structuring_element: Optional[Tensor]    = None,
        origin             : Optional[list[int]] = None,
        border_type        : str                 = "geodesic",
        border_value       : float               = 0.0,
        max_val            : float               = 1e4,
        engine             : str                 = "unfold",
    ):
        super().__init__()
        self.kernel              = kernel
        self.structuring_element = structuring_element
        self.origin              = origin
        self.border_type         = border_type
        self.border_value        = border_value
        self.max_val             = max_val
        self.engine              = engine
    
    def forward(self, image: Tensor) -> Tensor:
        return closing(
            image, self.kernel, self.structuring_element, self.origin,
            self.border_type, self.border_value, self.max_val, self.engine
        )


# MARK: - Gradient
# Morphological Gradient
def morphology_gradient(
    image              : Tensor,
    kernel             : Tensor,
    structuring_element: Optional[Tensor]    = None,
    origin             : Optional[list[int]] = None,
    border_type        : str                 = "geodesic",
    border_value       : float               = 0.0,
    max_val            : float               = 1e4,
    engine             : str                 = "unfold",
) -> Tensor:
    """Return the morphological gradient of an image. That means,
    (dilation - erosion) applying the same kernel in each channel. Kernel
    must have 2 dimensions.

    Args:
        image (Tensor):
            Image with shape [B, C, H, W].
        kernel (Tensor):
            Positions of non-infinite elements of a flat structuring element.
            Non-zero values give the set of neighbors of the center over which
            the operation is applied. Its shape is [k_x, k_y]. For full
            structural elements use torch.ones_like(structural_element).
        structuring_element (Tensor, optional):
            Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin (list[int], optional):
            Origin of the structuring element. Default: `None` and uses the
            center of the structuring element as origin (rounding towards zero).
        border_type (str):
            It determines how the image borders are handled, where
            `border_value` is the value when `border_type` is equal to
            `constant`. Default: `geodesic` which ignores the values that are
            outside the image when applying the operation.
        border_value (float):
            Value to fill past edges of input if `border_type` is `constant`.
        max_val (float):
            Fvalue of the infinite elements in the kernel.
        engine (str):
            Convolution is faster and less memory hungry, and unfold is more
            stable numerically.
        
    Returns:
        output (Tensor):
            Gradient image with shape [B, C, H, W].
 
    Example:
        >>> image        = torch.rand(1, 3, 5, 5)
        >>> kernel       = torch.ones(3, 3)
        >>> gradient_img = morphology_gradient(image, kernel)
    """
    return dilation(
        image,
        kernel              = kernel,
        structuring_element = structuring_element,
        origin              = origin,
        border_type         = border_type,
        border_value        = border_value,
        max_val             = max_val,
        engine              = engine,
    ) - erosion(
        image,
        kernel              = kernel,
        structuring_element = structuring_element,
        origin              = origin,
        border_type         = border_type,
        border_value        = border_value,
        max_val             = max_val,
        engine              = engine,
    )


@TRANSFORMS.register(name="gradient")
class Gradient(torch.nn.Module):

    def __init__(
        self,
        kernel             : Tensor,
        structuring_element: Optional[Tensor]    = None,
        origin             : Optional[list[int]] = None,
        border_type        : str                 = "geodesic",
        border_value       : float               = 0.0,
        max_val            : float               = 1e4,
        engine             : str                 = "unfold",
    ):
        super().__init__()
        self.kernel              = kernel
        self.structuring_element = structuring_element
        self.origin              = origin
        self.border_type         = border_type
        self.border_value        = border_value
        self.max_val             = max_val
        self.engine              = engine
    
    def forward(self, image: Tensor) -> Tensor:
        return morphology_gradient(
            image, self.kernel, self.structuring_element, self.origin,
            self.border_type, self.border_value, self.max_val, self.engine
        )


# MARK: - TopHat

def top_hat(
    image              : Tensor,
    kernel             : Tensor,
    structuring_element: Optional[Tensor]    = None,
    origin             : Optional[list[int]] = None,
    border_type        : str                 = "geodesic",
    border_value       : float               = 0.0,
    max_val            : float               = 1e4,
    engine             : str                 = "unfold",
) -> Tensor:
    """Return the top hat transformation of an image. That means,
    (image - opened_image) applying the same kernel in each channel. Kernel
    must have 2 dimensions.

    Args:
        image (Tensor):
            Image with shape [B, C, H, W].
        kernel (Tensor):
            Positions of non-infinite elements of a flat structuring element.
            Non-zero values give the set of neighbors of the center over which
            the operation is applied. Its shape is [k_x, k_y]. For full
            structural elements use torch.ones_like(structural_element).
        structuring_element (Tensor, optional):
            Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin (list[int], optional):
            Origin of the structuring element. Default: `None` and uses the
            center of the structuring element as origin (rounding towards zero).
        border_type (str):
            It determines how the image borders are handled, where
            `border_value` is the value when `border_type` is equal to
            `constant`. Default: `geodesic` which ignores the values that are
            outside the image when applying the operation.
        border_value (float):
            Value to fill past edges of input if `border_type` is `constant`.
        max_val (float):
            Fvalue of the infinite elements in the kernel.
        engine (str):
            Convolution is faster and less memory hungry, and unfold is more
            stable numerically.
        
    Returns:
        output (Tensor):
            Top hat transformed image with shape [B, C, H, W].

    Example:
        >>> tensor      = torch.rand(1, 3, 5, 5)
        >>> kernel      = torch.ones(3, 3)
        >>> top_hat_img = top_hat(image, kernel)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. "
                        f"Got: {type(image)}")
    if len(image.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. "
                         f"Got: {image.dim()}")
    if not isinstance(kernel, Tensor):
        raise TypeError(f"Kernel type is not a Tensor. "
                        f"Got: {type(kernel)}")
    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. "
                         f"Got: {kernel.dim()}")

    return image - opening(
        image,
        kernel              = kernel,
        structuring_element = structuring_element,
        origin              = origin,
        border_type         = border_type,
        border_value        = border_value,
        max_val             = max_val,
        engine              = engine,
    )


@TRANSFORMS.register(name="top_hat")
class TopHat(torch.nn.Module):

    def __init__(
        self,
        kernel             : Tensor,
        structuring_element: Optional[Tensor]    = None,
        origin             : Optional[list[int]] = None,
        border_type        : str                 = "geodesic",
        border_value       : float               = 0.0,
        max_val            : float               = 1e4,
        engine             : str                 = "unfold",
    ):
        super().__init__()
        self.kernel              = kernel
        self.structuring_element = structuring_element
        self.origin              = origin
        self.border_type         = border_type
        self.border_value        = border_value
        self.max_val             = max_val
        self.engine              = engine
    
    def forward(self, image: Tensor) -> Tensor:
        return top_hat(
            image, self.kernel, self.structuring_element, self.origin,
            self.border_type, self.border_value, self.max_val, self.engine
        )


# MARK: - BottomHat

def bottom_hat(
    image              : Tensor,
    kernel             : Tensor,
    structuring_element: Optional[Tensor]    = None,
    origin             : Optional[list[int]] = None,
    border_type        : str                 = "geodesic",
    border_value       : float               = 0.0,
    max_val            : float               = 1e4,
    engine             : str                 = "unfold",
) -> Tensor:
    """Return the bottom hat transformation of an image. That means,
    (closed_image - image) applying the same kernel in each channel. Kernel
    must have 2 dimensions.

    Args:
        image (Tensor):
            Image with shape [B, C, H, W].
        kernel (Tensor):
            Positions of non-infinite elements of a flat structuring element.
            Non-zero values give the set of neighbors of the center over which
            the operation is applied. Its shape is [k_x, k_y]. For full
            structural elements use torch.ones_like(structural_element).
        structuring_element (Tensor, optional):
            Structuring element used for the grayscale dilation. It may be a
            non-flat structuring element.
        origin (list[int], optional):
            Origin of the structuring element. Default: `None` and uses the
            center of the structuring element as origin (rounding towards zero).
        border_type (str):
            It determines how the image borders are handled, where
            `border_value` is the value when `border_type` is equal to
            `constant`. Default: `geodesic` which ignores the values that are
            outside the image when applying the operation.
        border_value (float):
            Value to fill past edges of input if `border_type` is `constant`.
        max_val (float):
            Fvalue of the infinite elements in the kernel.
        engine (str):
            Convolution is faster and less memory hungry, and unfold is more
            stable numerically.
        
    Returns:
        output (Tensor):
            Top hat transformed image with shape [B, C, H, W].
 
    Example:
        >>> image          = torch.rand(1, 3, 5, 5)
        >>> kernel         = torch.ones(3, 3)
        >>> bottom_hat_img = bottom_hat(image, kernel)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. "
                        f"Got: {type(image)}")
    if len(image.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. "
                         f"Got: {image.dim()}")
    if not isinstance(kernel, Tensor):
        raise TypeError(f"Kernel type is not a Tensor. "
                        f"Got: {type(kernel)}")
    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. "
                         f"Got: {kernel.dim()}")

    return (
        closing(
            image,
            kernel              = kernel,
            structuring_element = structuring_element,
            origin              = origin,
            border_type         = border_type,
            border_value        = border_value,
            max_val             = max_val,
            engine              = engine,
        )
        - image
    )


@TRANSFORMS.register(name="bottom_hat")
class BottomHat(torch.nn.Module):

    def __init__(
        self,
        kernel             : Tensor,
        structuring_element: Optional[Tensor]    = None,
        origin             : Optional[list[int]] = None,
        border_type        : str                 = "geodesic",
        border_value       : float               = 0.0,
        max_val            : float               = 1e4,
        engine             : str                 = "unfold",
    ):
        super().__init__()
        self.kernel              = kernel
        self.structuring_element = structuring_element
        self.origin              = origin
        self.border_type         = border_type
        self.border_value        = border_value
        self.max_val             = max_val
        self.engine              = engine
    
    def forward(self, image: Tensor) -> Tensor:
        return bottom_hat(
            image, self.kernel, self.structuring_element, self.origin,
            self.border_type, self.border_value, self.max_val, self.engine
        )
