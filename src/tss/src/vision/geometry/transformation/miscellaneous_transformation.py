#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode

from src.factory import TRANSFORMS
from .affine_transformation import *

__all__ = [
	"shear_",
	"Shear",
	"shear_x",
	"shear_y",
	"ShearX",
	"ShearY"
]


# MARK: - Shear

def shear_(
	image        : Tensor,
	shear        : list[float],
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	fill         : Optional[list[float]] = None,
) -> Tensor:
	"""
	
	Args:
        image (PIL Image or Tensor):
            Image to transform.
        shear (list[float]):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x axis, while the second value
            corresponds to a shear parallel to the y axis.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported.
            For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
            are still acceptable.
        fill (list[float], optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.

    Returns:
        (PIL Image or Tensor):
            Transformed image.
	"""
	return affine(
		image, angle=0.0, translate=[0, 0], scale=1.0, shear=shear,
		interpolation=interpolation, fill=fill,
	)


def shear_x(
	image        : Tensor,
	magnitude    : float,
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	fill         : Optional[list[float]] = None,
) -> Tensor:
	return affine(
		image, angle=0.0, translate=[0, 0], scale=1.0,
		shear=[math.degrees(magnitude), 0.0], interpolation=interpolation,
		fill=fill,
	)


def shear_y(
	image        : Tensor,
	magnitude    : float,
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	fill         : Optional[list[float]] = None,
) -> Tensor:
	return affine(
		image, angle=0.0, translate=[0, 0], scale=1.0,
		shear=[0.0, math.degrees(magnitude)], interpolation=interpolation,
		fill=fill,
	)


@TRANSFORMS.register(name="shear")
class Shear(torch.nn.Module):
	"""
	
    Args:
    	shear (list[float]):
	        Shear angle value in degrees between -180 to 180, clockwise
	        direction. If a sequence is specified, the first value
	        corresponds to a shear parallel to the x axis, while the second
	        value corresponds to a shear parallel to the y axis.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        fill (list[float], optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		shear        : list[float],
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		fill         : Optional[list[float]] = None,
	):
		super().__init__()
		self.shear         = shear
		self.interpolation = interpolation
		self.fill          = fill
	
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		"""
		
		Args:
			image (PIL Image or Tensor):
				Image to transform.

		Returns:
			(PIL Image or Tensor):
				Transformed image.
		"""
		return shear_(image, self.shear, self.interpolation, self.fill)


@TRANSFORMS.register(name="shear_x")
class ShearX(torch.nn.Module):
	"""
	
    Args:
    	magnitude (float):
	        Shear angle value in degrees between -180 to 180, clockwise
	        direction.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        fill (list[float], optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		magnitude    : float,
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		fill         : Optional[list[float]] = None,
	):
		super().__init__()
		self.magnitude     = magnitude
		self.interpolation = interpolation
		self.fill          = fill
	
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		"""
		
		Args:
			image (PIL Image or Tensor):
				Image to transform.

		Returns:
			(PIL Image or Tensor):
				Transformed image.
		"""
		return shear_x(image, self.magnitude, self.interpolation, self.fill)


@TRANSFORMS.register(name="shear_y")
class ShearY(torch.nn.Module):
	"""
	
    Args:
    	magnitude (float):
	        Shear angle value in degrees between -180 to 180, clockwise
	        direction.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        fill (list[float], optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		magnitude    : float,
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		fill         : Optional[list[float]] = None,
	):
		super().__init__()
		self.magnitude     = magnitude
		self.interpolation = interpolation
		self.fill          = fill
	
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		"""
		
		Args:
			image (PIL Image or Tensor):
				Image to transform.

		Returns:
			(PIL Image or Tensor):
				Transformed image.
		"""
		return shear_y(image, self.magnitude, self.interpolation, self.fill)
