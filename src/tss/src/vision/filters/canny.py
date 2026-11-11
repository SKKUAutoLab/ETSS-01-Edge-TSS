#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from src.factory import TRANSFORMS
from src.type import Dim2
from src.type import ListOrTuple2T
from src.vision.color import rgb_to_grayscale
from .gaussian import gaussian_blur2d
from .kernels import get_canny_nms_kernel
from .kernels import get_hysteresis_kernel
from .sobel import spatial_gradient

__all__ = [
    "canny", "Canny"
]


# MARK: - Canny

def canny(
    image         : Tensor,
    low_threshold : float                = 0.1,
    high_threshold: float                = 0.2,
    kernel_size   : Dim2                 = (5, 5),
    sigma         : ListOrTuple2T[float] = (1, 1),
    hysteresis    : bool                 = True,
    eps           : float                = 1e-6,
) -> ListOrTuple2T[Tensor]:
    """Find edges of the input image and filters them using the Canny algorithm.

    Args:
        image (Tensor):
            Input image with shape [B, C, H, W].
        low_threshold (float):
            Lower threshold for the hysteresis procedure. Default: `0.1`.
        high_threshold (float):
            Upper threshold for the hysteresis procedure. Default: `0.2`.
        kernel_size (Dim2):
            Fsize of the kernel for the gaussian blur. Default: `(5, 5)`.
        sigma (ListOrTuple2T[float]):
            Standard deviation of the kernel for the gaussian blur.
            Default: `(1, 1)`.
        hysteresis (bool):
            If True, applies the hysteresis edge tracking. Otherwise, the edges
            are divided between weak (0.5) and strong (1) edges.
            Default: `True`.
        eps (float):
            Regularization number to avoid NaN during backprop. Default: `1e-6`.

    Returns:
        - the canny edge magnitudes map, shape of [B, 1, H, W].
        - the canny edge detection filtered by thresholds and hysteresis,
          of shape [B, 1, H, W].

    Example:
        >>> input            = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = canny(input)  # [5, 3, 4, 4]
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(image)}")
    if not len(image.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect [B, C, H, W]. "
                         f"Got: {image.shape}")
    if low_threshold > high_threshold:
        raise ValueError(
            f"Invalid input thresholds. low_threshold should be smaller than "
            f"the high_threshold. Got: {low_threshold}>{high_threshold}"
        )
    if 0 > low_threshold > 1:
        raise ValueError(f"Invalid input threshold. low_threshold should be "
                         f"in range (0,1). Got: {low_threshold}")
    if 0 > high_threshold > 1:
        raise ValueError(f"Invalid input threshold. high_threshold should be "
                         f"in range (0,1). Got: {high_threshold}")

    device = image.device
    dtype  = image.dtype

    # To Grayscale
    if image.shape[1] == 3:
        image = rgb_to_grayscale(image)

    # Gaussian filter
    blurred = gaussian_blur2d(image, kernel_size, sigma)

    # Compute the gradients
    gradients = spatial_gradient(blurred, normalized=False)

    # Unpack the edges
    gx = gradients[:, :, 0]
    gy = gradients[:, :, 1]

    # Compute gradient magnitude and angle
    magnitude = torch.sqrt(gx * gx + gy * gy + eps)
    angle     = torch.atan2(gy, gx)

    # Radians to Degrees
    angle = 180. * angle / math.pi

    # Round angle to the nearest 45 degree
    angle = torch.round(angle / 45) * 45

    # Non-maximal suppression
    nms_kernels   = get_canny_nms_kernel(device, dtype)
    nms_magnitude = F.conv2d(
        magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2
    )

    # Get the indices for both directions
    positive_idx = (angle / 45) % 8
    positive_idx = positive_idx.long()

    negative_idx = ((angle / 45) + 4) % 8
    negative_idx = negative_idx.long()

    # Apply the non-maximum suppression to the different directions
    channel_select_filtered_positive = torch.gather(
        nms_magnitude, 1, positive_idx
    )
    channel_select_filtered_negative = torch.gather(
        nms_magnitude, 1, negative_idx
    )
    channel_select_filtered = torch.stack(
        [channel_select_filtered_positive, channel_select_filtered_negative], 1
    )

    is_max    = channel_select_filtered.min(dim=1)[0] > 0.0
    magnitude = magnitude * is_max

    # Threshold
    edges = F.threshold(magnitude, low_threshold, 0.0)
    low   = magnitude > low_threshold
    high  = magnitude > high_threshold
    edges = low * 0.5 + high * 0.5
    edges = edges.to(dtype)

    # Hysteresis
    if hysteresis:
        edges_old = -torch.ones(edges.shape, device=edges.device, dtype=dtype)
        hysteresis_kernels = get_hysteresis_kernel(device, dtype)

        while ((edges_old - edges).abs() != 0).any():
            weak   = (edges == 0.5).float()
            strong = (edges == 1).float()

            hysteresis_magnitude = F.conv2d(
                edges, hysteresis_kernels,
                padding=hysteresis_kernels.shape[-1] // 2
            )
            hysteresis_magnitude = (
                (hysteresis_magnitude == 1).any(1, keepdim=True).to(dtype)
            )
            hysteresis_magnitude = hysteresis_magnitude * weak + strong

            edges_old = edges.clone()
            edges = (
                hysteresis_magnitude + (hysteresis_magnitude == 0) * weak * 0.5
            )

        edges = hysteresis_magnitude

    return magnitude, edges


@TRANSFORMS.register(name="canny")
class Canny(torch.nn.Module):
    """Module that finds edges of the input image and filters them using the
    Canny algorithm.

    Args:
        low_threshold (float):
            Lower threshold for the hysteresis procedure. Default: `0.1`.
        high_threshold (float):
            Upper threshold for the hysteresis procedure. Default: `0.2`.
        kernel_size (Dim2):
            Fsize of the kernel for the gaussian blur. Default: `(5, 5)`.
        sigma (ListOrTuple2T[float]):
            Standard deviation of the kernel for the gaussian blur.
            Default: `(1, 1)`.
        hysteresis (bool):
            If True, applies the hysteresis edge tracking. Otherwise, the edges
            are divided between weak (0.5) and strong (1) edges.
            Default: `True`.
        eps (float):
            Regularization number to avoid NaN during backprop. Default: `1e-6`.

    Example:
        >>> input            = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = Canny()(input)  # [5, 3, 4, 4]
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        low_threshold : float                = 0.1,
        high_threshold: float                = 0.2,
        kernel_size   : Dim2                 = (5, 5),
        sigma         : ListOrTuple2T[float] = (1, 1),
        hysteresis    : bool                 = True,
        eps           : float                = 1e-6,
    ):
        super().__init__()

        if low_threshold > high_threshold:
            raise ValueError(
                f"Invalid input thresholds. low_threshold should be smaller "
                f"than the high_threshold. "
                f"Got: {low_threshold}>{high_threshold}"
            )
        if low_threshold < 0 or low_threshold > 1:
            raise ValueError(f"Invalid input threshold. low_threshold should "
                             f"be in range (0,1). Got: {low_threshold}")
        if high_threshold < 0 or high_threshold > 1:
            raise ValueError(f"Invalid input threshold. high_threshold should "
                             f"be in range (0,1). Got: {high_threshold}")

        # Gaussian blur parameters
        self.kernel_size    = kernel_size
        self.sigma          = sigma
        # Double threshold
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        # Hysteresis
        self.hysteresis     = hysteresis
        self.eps            = eps

    def __repr__(self) -> str:
        return "".join(
            (
                f"{type(self).__name__}(",
                ", ".join(
                    f"{name}={getattr(self, name)}"
                    for name in sorted(self.__dict__)
                    if not name.startswith("_")
                ),
                ")",
            )
        )
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> ListOrTuple2T[Tensor]:
        return canny(
            image, self.low_threshold, self.high_threshold, self.kernel_size,
            self.sigma, self.hysteresis, self.eps
        )
