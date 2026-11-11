#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""An affine transformation is any transformation that preserves collinearity
(i.e., all points lying on a line initially still lie on a line after
transformation) and ratios of distances (e.g., the midpoint of a line segment
remains the midpoint after transformation). In this sense, affine indicates a
special class of projective transformations that do not move any objects from
the affine space R^3 to the plane at infinity or conversely. An affine
transformation is also called an affinity.

Geometric contraction, expansion, dilation, reflection, rotation, shear,
similarity transformations, spiral similarities, and translation are all
affine transformations, as are their combinations. In general, an affine
transformation is a composition of rotations, translations, dilations,
and shears.

While an affine transformation preserves proportions on lines, it does not
necessarily preserve angles or lengths. Any triangle can be transformed into
any other by an affine transformation, so all triangles are affine and,
in this sense, affine is a generalization of congruent and similar.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine

from src.factory import TRANSFORMS

__all__ = [
	"affine",
	"Affine"
]


# MARK: - Affine

@TRANSFORMS.register(name="affine")
class Affine(torch.nn.Module):
	"""Apply affine transformation on the image keeping image center invariant.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    
    Args:
    	angle (float):
    	    Rotation angle in degrees between -180 and 180, clockwise direction.
	    translate (list[int]):
	        Horizontal and vertical translations (post-rotation translation).
	    scale (float):
	        Overall scale
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
		angle        : float,
		translate    : list[int],
		scale        : float,
		shear        : list[float],
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		fill         : Optional[list[float]] = None,
	):
		super().__init__()
		self.angle         = angle
		self.translate     = translate
		self.scale         = scale
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
		return affine(
			image, self.angle, self.translate, self.scale, self.shear,
			self.interpolation, self.fill,
		)
