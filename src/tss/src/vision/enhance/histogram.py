# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from src.type import ListOrTuple2T

__all__ = [
    "histogram",
    "histogram2d",
    "image_histogram2d",
    "joint_pdf",
    "marginal_pdf",
]


# MARK: - Histogram

def marginal_pdf(
    values : Tensor,
    bins   : Tensor,
    sigma  : Tensor,
    epsilon: float = 1e-10
) -> ListOrTuple2T[Tensor]:
    """Calculate the marginal probability distribution function of the input
    image based on the number of histogram bins.

    Args:
        values (Tensor):
            Shape [B, N, 1].
        bins (Tensor):
            Shape [NUM_BINS].
        sigma (Tensor):
            Shape [1], gaussian smoothing factor.
        epsilon (float):
            Scalar, for numerical stability.

    Returns:
        tuple[Tensor, Tensor]:
          - Tensor: shape [B, N].
          - Tensor: shape [B, N, NUM_BINS].
    """

    if not isinstance(values, Tensor):
        raise TypeError(f"Input values type is not a Tensor. "
                        f"Got: {type(values)}")
    if not isinstance(bins, Tensor):
        raise TypeError(f"Input bins type is not a Tensor. "
                        f"Got: {type(bins)}")
    if not isinstance(sigma, Tensor):
        raise TypeError(f"Input sigma type is not a Tensor. "
                        f"Got: {type(sigma)}")
    if not values.dim() == 3:
        raise ValueError(f"Input values must be a of the shape [B, N, 1]. "
                         f"Got: {values.shape}")
    if not bins.dim() == 1:
        raise ValueError(f"Input bins must be a of the shape [NUM_BINS]. "
                         f"Got: {bins.shape}")
    if not sigma.dim() == 0:
        raise ValueError(f"Input sigma must be a of the shape 1. "
                         f"Got: {sigma.shape}")

    residuals     = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

    pdf           = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
    pdf           = pdf / normalization

    return pdf, kernel_values


def joint_pdf(
    kernel_values1: Tensor,
    kernel_values2: Tensor,
    epsilon       : float = 1e-10
) -> Tensor:
    """Calculate the joint probability distribution function of the input
    tensors based on the number of histogram bins.

    Args:
        kernel_values1 (Tensor):
            Shape [B, N, NUM_BINS].
        kernel_values2 (Tensor):
            Shape [B, N, NUM_BINS].
        epsilon (float):
            Scalar, for numerical stability.

    Returns:
        pdf (Tensor):
            Shape [B, NUM_BINS, NUM_BINS].
    """
    if not isinstance(kernel_values1, Tensor):
        raise TypeError(f"Input kernel_values1 type is not a Tensor. "
                        f"Got: {type(kernel_values1)}")
    if not isinstance(kernel_values2, Tensor):
        raise TypeError(f"Input kernel_values2 type is not a Tensor. "
                        f"Got: {type(kernel_values2)}")
    if not kernel_values1.dim() == 3:
        raise ValueError(f"Input kernel_values1 must be a of the shape [B, N]."
                         f"Got: {kernel_values1.shape}")
    if not kernel_values2.dim() == 3:
        raise ValueError(f"Input kernel_values2 must be a of the shape [B, N]."
                         f"Got: {kernel_values2.shape}")
    if kernel_values1.shape != kernel_values2.shape:
        raise ValueError(
            f"Inputs kernel_values1 and kernel_values2 must have the same "
            f"shape. Got: {kernel_values1.shape} and {kernel_values2.shape}"
        )

    joint_kernel_values = torch.matmul(
        kernel_values1.transpose(1, 2), kernel_values2
    )
    normalization = torch.sum(
        joint_kernel_values, dim=(1, 2)
    ).view(-1, 1, 1) + epsilon
    
    pdf = joint_kernel_values / normalization
    return pdf


def histogram(
    x        : Tensor,
    bins     : Tensor,
    bandwidth: Tensor,
    epsilon  : float = 1e-10
) -> Tensor:
    """Estimate the histogram of the input image. Fcalculation uses kernel
    density estimation which requires a bandwidth (smoothing) parameter.

    Args:
        x (Tensor):
            Input image to compute the histogram with shape [B, D].
        bins (Tensor):
            Number of bins to use the histogram N_{bins}.
        bandwidth (Tensor):
            Gaussian smoothing factor with shape [1].
        epsilon (float):
            A scalar, for numerical stability.

    Returns:
        pdf (Tensor):
            Computed histogram of shape [B, N_{bins}].

    Examples:
        >>> x    = torch.rand(1, 10)
        >>> bins = torch.torch.linspace(0, 255, 128)
        >>> hist = histogram(x, bins, bandwidth=torch.tensor(0.9))
        >>> hist.shape
        torch.Size([1, 128])
    """
    pdf, _ = marginal_pdf(x.unsqueeze(2), bins, bandwidth, epsilon)
    return pdf


def histogram2d(
    x1       : Tensor,
    x2       : Tensor,
    bins     : Tensor,
    bandwidth: Tensor,
    epsilon  : float = 1e-10
) -> Tensor:
    """Estimate the 2d histogram of the input image. Fcalculation uses
    kernel density estimation which requires a bandwidth (smoothing) parameter.

    Args:
        x1 (Tensor):
            Input image to compute the histogram with shape [B, D1].
        x2 (Tensor):
        Input image to compute the histogram with shape [B, D2].
        bins (Tensor):
            Number of bins to use the histogram [N_{bins}].
        bandwidth (Tensor):
            Gaussian smoothing factor with shape [1].
        epsilon (float):
            A scalar, for numerical stability. Default: `1e-10`.

    Returns:
        pdf (Tensor):
            Computed histogram of shape [B, N_{bins}, N_{bins}].

    Examples:
        >>> x1   = torch.rand(2, 32)
        >>> x2   = torch.rand(2, 32)
        >>> bins = torch.torch.linspace(0, 255, 128)
        >>> hist = histogram2d(x1, x2, bins, bandwidth=torch.tensor(0.9))
        >>> hist.shape
        torch.Size([2, 128, 128])
    """
    _, kernel_values1 = marginal_pdf(x1.unsqueeze(2), bins, bandwidth, epsilon)
    _, kernel_values2 = marginal_pdf(x2.unsqueeze(2), bins, bandwidth, epsilon)
    pdf               = joint_pdf(kernel_values1, kernel_values2)
    return pdf


def image_histogram2d(
    image     : Tensor,
    min       : float                  = 0.0,
    max       : float                  = 255.0,
    n_bins    : int                    = 256,
    bandwidth : Optional[float]        = None,
    centers   : Optional[Tensor] = None,
    return_pdf: bool                   = False,
    kernel    : str                    = "triangular",
    eps       : float                  = 1e-10,
) -> ListOrTuple2T[Tensor]:
    """Estimate the histogram of the input image(s). Fcalculation uses
    triangular kernel density estimation.

    Args:
        image (Tensor):
            Input image to compute the histogram with shape [H, W], [C, H, W],
            or [B, C, H, W].
        min (float):
            Lower end of the interval (inclusive).
        max (float):
            Upper end of the interval (inclusive). Ignored when `centers` is
            specified.
        n_bins (int):
            Number of histogram bins. Ignored when `centers` is specified.
        bandwidth (float, optional):
            Smoothing factor. If not specified or equal to `-1`,
            (bandwidth = (max - min) / n_bins).
        centers (Tensor, optional):
            Centers of the bins with shape [n_bins,]. If not specified or empty,
            it is calculated as centers of equal width bins of [min, max] range.
        return_pdf (bool):
            If `True`, also return probability densities for each bin.
        kernel (str):
            Kernel to perform kernel density estimation. One of: [`triangular`,
            `gaussian`, `uniform`, `epanechnikov`]. Default: `triangular`.

    Returns:
        Computed histogram of shape [bins], [C, bins], [B, C, bins].
        Computed probability densities of shape [bins], [C, bins], [B, C, bins],
        if return_pdf is `True`. Tensor of zeros with shape of the histogram
        otherwise.
    """
    if image is not None and not isinstance(image, Tensor):
        raise TypeError(f"Input image type is not a Tensor. "
                        f"Got: {type(image)}.")
    if centers is not None and not isinstance(centers, Tensor):
        raise TypeError(f"Bins' centers type is not a Tensor. "
                        f"Got: {type(centers)}.")
    if centers is not None and len(centers.shape) > 0 and centers.dim() != 1:
        raise ValueError(f"Bins' centers must be a Tensor of the shape "
                         f"[n_bins,]. Got: {centers.shape}.")
    if not isinstance(min, float):
        raise TypeError(f"Type of lower end of the range is not a float. "
                        f"Got: {type(min)}.")
    if not isinstance(max, float):
        raise TypeError(f"Type of upper end of the range is not a float. "
                        f"Got: {type(min)}.")
    if not isinstance(n_bins, int):
        raise TypeError(f"Type of number of bins is not an int. "
                        f"Got: {type(n_bins)}.")
    if bandwidth is not None and not isinstance(bandwidth, float):
        raise TypeError(f"Bandwidth type is not a float. "
                        f"Got: {type(bandwidth)}.")
    if not isinstance(return_pdf, bool):
        raise TypeError(f"Return_pdf type is not a bool. "
                        f"Got: {type(return_pdf)}.")

    if bandwidth is None:
        bandwidth = (max - min) / n_bins
    if centers is None:
        centers = min + bandwidth * (torch.arange(
            n_bins, device=image.device, dtype=image.dtype
        ).float() + 0.5)
   
    centers = centers.reshape(-1, 1, 1, 1, 1)
    u       = torch.abs(image.unsqueeze(0) - centers) / bandwidth
    if kernel == "triangular":
        mask = (u <= 1).to(u.dtype)
        kernel_values = (1 - u) * mask
    elif kernel == "gaussian":
        kernel_values = torch.exp(-0.5 * u ** 2)
    elif kernel == "uniform":
        mask = (u <= 1).to(u.dtype)
        kernel_values = torch.ones_like(u, dtype=u.dtype, device=u.device) * mask
    elif kernel == "epanechnikov":
        mask = (u <= 1).to(u.dtype)
        kernel_values = (1 - u ** 2) * mask
    else:
        raise ValueError(f"Kernel must be `triangular`, `gaussian`, "
                         f"`uniform` or `epanechnikov`. Got: {kernel}.")

    hist = torch.sum(kernel_values, dim=(-2, -1)).permute(1, 2, 0)
    if return_pdf:
        normalization = torch.sum(hist, dim=-1, keepdim=True) + eps
        pdf           = hist / normalization
        if image.dim() == 2:
            hist = hist.squeeze()
            pdf  = pdf.squeeze()
        elif image.dim() == 3:
            hist = hist.squeeze(0)
            pdf  = pdf.squeeze(0)
        return hist, pdf

    if image.dim() == 2:
        hist = hist.squeeze()
    elif image.dim() == 3:
        hist = hist.squeeze(0)
    return hist, torch.zeros_like(hist, dtype=hist.dtype, device=hist.device)
