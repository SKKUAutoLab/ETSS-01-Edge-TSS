#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import warnings
from typing import Optional

import cv2
import numpy as np
import torch
from multipledispatch import dispatch
from torch import Tensor

from ..utils import to_channel_first

__all__ = [
    "blend_images"
]


# MARK: - Blend Images

@dispatch(Tensor, Tensor, float, float)
def blend_images(
    overlays: Tensor,
    images  : Tensor,
    alpha   : float,
    gamma   : float = 0.0
) -> Tensor:
    """Blends 2 images together. dst = image1 * alpha + image2 * beta + gamma

    Args:
        overlays (Tensor):
            Images we want to overlay on top of the original image.
        images (Tensor):
            Source images.
        alpha (float):
            Alpha transparency of the overlay.
        gamma (float):

    Returns:
        blend (Tensor):
            Blended image.
    """
    overlays_np = overlays.numpy()
    images_np   = images.numpy()
    blends      = blend_images(overlays_np, images_np, alpha, gamma)
    blends      = torch.from_numpy(blends)
    return blends


@dispatch(np.ndarray, np.ndarray, float, float)
def blend_images(
    overlays: np.ndarray,
    images  : np.ndarray,
    alpha   : float,
    gamma   : float = 0.0
) -> Optional[np.ndarray]:
    """Blends 2 images together. dst = image1 * alpha + image2 * beta + gamma

    Args:
        overlays (np.ndarray):
            Images we want to overlay on top of the original image.
        images (np.ndarray):
            Source images.
        alpha (float):
            Alpha transparency of the overlay.
        gamma (float):

    Returns:
        blend (np.ndarray, optional):
            Blended image.
    """
    # NOTE: Type checking
    if overlays.ndim != images.ndim:
        raise ValueError(f"image1 dims != image2 dims: "
                         f"{overlays.ndim} != {images.ndim}")
    
    # NOTE: Convert to channel-first
    overlays = to_channel_first(overlays)
    images   = to_channel_first(images)
    
    # NOTE: Unnormalize images
    from src.vision import denormalize_naive
    images = denormalize_naive(images)
    
    # NOTE: Convert overlays to same data type as images
    images   = images.astype(np.uint8)
    overlays = overlays.astype(np.uint8)
    
    # NOTE: If the images are of shape [CHW]
    if overlays.ndim == 3 and images.ndim == 3:
        return cv2.addWeighted(overlays, alpha, images, 1.0 - alpha, gamma)
    
    # NOTE: If the images are of shape [BCHW]
    if overlays.ndim == 4 and images.ndim == 4:
        if overlays.shape[0] != images.shape[0]:
            raise ValueError(
                f"Number of batch in image1 != Number of batch in image2: "
                f"{overlays.shape[0]} != {images.shape[0]}"
            )
        blends = []
        for overlay, image in zip(overlays, images):
            blends.append(cv2.addWeighted(overlay, alpha, image, 1.0 - alpha,
                                          gamma))
        blends = np.stack(blends, axis=0).astype(np.uint8)
        return blends
    
    warnings.warn(f"Cannot blend images and overlays with dimensions: "
                  f"{images.ndim} and {overlays.ndim}")
    return None
