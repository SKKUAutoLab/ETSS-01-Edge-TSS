#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode

from src.factory import TRANSFORMS
from .affine_transformation import *

__all__ = [
	"translate_",
	"Translate",
	"translate_x",
	"translate_y",
	"TranslateX",
	"TranslateY"
]


# MARK: - Translate

def translate_(
	image        : Tensor,
	translate    : list[int],
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	fill         : Optional[list[float]] = None,
) -> Tensor:
	"""
	
	Args:
        image (PIL Image or Tensor):
            Image to transform.
        translate (list[int]):
            Horizontal and vertical translations (post-rotation translation)
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
		image, angle=0.0, translate=translate, scale=1.0, shear=[0.0, 0.0],
		interpolation=interpolation, fill=fill,
	)


def translate_x(
	image        : Tensor,
	magnitude    : int,
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	fill         : Optional[list[float]] = None,
) -> Tensor:
	"""
	
	Args:
        image (PIL Image or Tensor):
            Image to transform.
        magnitude (int):
            Horizontal translation (post-rotation translation)
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
		image, angle=0.0, translate=[magnitude, 0], scale=1.0, shear=[0.0, 0.0],
		interpolation=interpolation, fill=fill,
	)


def translate_y(
	image        : Tensor,
	magnitude    : int,
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	fill         : Optional[list[float]] = None,
) -> Tensor:
	"""
	
	Args:
        image (PIL Image or Tensor):
            Image to transform.
        magnitude (int):
            Vertical translation (post-rotation translation)
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
		image, angle=0.0, translate=[0, magnitude], scale=1.0, shear=[0.0, 0.0],
		interpolation=interpolation, fill=fill,
	)


@TRANSFORMS.register(name="translate")
class Translate(torch.nn.Module):
	"""
	
    Args:
    	translate (list[int]):
            Horizontal and vertical translations (post-rotation translation)
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
		translate    : list[int],
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		fill         : Optional[list[float]] = None,
	):
		super().__init__()
		self.translate     = translate
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
		return translate_(image, self.translate, self.interpolation, self.fill)


@TRANSFORMS.register(name="translate_x")
class TranslateX(torch.nn.Module):
	"""
	
    Args:
    	magnitude (int):
            Horizontal translation (post-rotation translation)
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
		magnitude    : int,
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
		return translate_x(image, self.magnitude, self.interpolation, self.fill)


@TRANSFORMS.register(name="translate_y")
class TranslateY(torch.nn.Module):
	"""
	
    Args:
    	magnitude (int):
            Vertical translation (post-rotation translation)
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
		magnitude    : int,
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
		return translate_y(image, self.magnitude, self.interpolation, self.fill)
