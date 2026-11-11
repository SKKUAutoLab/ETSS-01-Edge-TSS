#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import vflip

from src.factory import TRANSFORMS

__all__ = [
	"rotate",
	"rotate_fliplr",
	"rotate_flipud",
	"random_rotate",
	"random_rotate_fliplr",
	"random_rotate_flipud",
	"Rotate",
	"RotateFliplr",
	"RotateFlipud",
	"RandomRotate",
	"RandomRotateFliplr",
	"RandomRotateFlipud"
]


# MARK: - Rotate

def rotate(
	image        : Tensor,
	angle        : float,
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	expand       : bool                  = False,
	center       : Optional[list[int]]   = None,
	fill         : Optional[list[float]] = None,
) -> Tensor:
	"""Rotate a tensor image or a batch of tensor images. Input must be a
	tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].
	
	Args:
		image (Tensor):
			Image to be rotated.
		angle (float):
			Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
   
    Returns:
		(Tensor):
			Rotated image.
	"""
	return F.rotate(image, angle, interpolation, expand, center, fill)


def rotate_fliplr(
	image        : Tensor,
	angle        : float,
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	expand       : bool                  = False,
	center       : Optional[list[int]]   = None,
	fill         : Optional[list[float]] = None,
) -> Tensor:
	"""Rotate a tensor image or a batch of tensor images and then horizontally
	flip. Input must be a tensor of shape [C, H, W] or a batch of
	tensors [*, C, H, W].
	
	Args:
		image (Tensor):
			Image to be rotated and flipped.
		angle (float):
			Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
   
    Returns:
		(Tensor):
			Rotated and flipped image.
	"""
	image = rotate(image, angle, interpolation, expand, center, fill)
	return hflip(image)


def rotate_flipud(
	image        : Tensor,
	angle        : float,
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	expand       : bool                  = False,
	center       : Optional[list[int]]   = None,
	fill         : Optional[list[float]] = None,
) -> Tensor:
	"""Rotate a tensor image or a batch of tensor images and then vertically
	flip. Input must be a tensor of shape [C, H, W] or a batch of tensors
	[*, C, H, W].
	
	Args:
		image (Tensor):
			Image to be rotated and flipped.
		angle (float):
			Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
   
    Returns:
		(Tensor):
			Rotated and flipped image.
	"""
	image = rotate(image, angle, interpolation, expand, center, fill)
	return vflip(image)


def random_rotate(
	image        : Tensor,
	angle        : float,
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	expand       : bool                  = False,
	center       : Optional[list[int]]   = None,
	fill         : Optional[list[float]] = None,
	p            : float                 = 0.5
) -> Tensor:
	"""Random rotate a tensor image or a batch of tensor images. Input must be
	a tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].
	
	Args:
		image (Tensor):
			Image to be rotated.
		angle (float):
			Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
		p (float):
            Probability of the image being flipped. Default: `0.5`.
    
    Returns:
		(Tensor):
			Randomly rotated image.
	"""
	if torch.rand(1) < p:
		return rotate(image, angle, interpolation, expand, center, fill)
	return image


def random_rotate_fliplr(
	image        : Tensor,
	angle        : float,
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	expand       : bool                  = False,
	center       : Optional[list[int]]   = None,
	fill         : Optional[list[float]] = None,
	p            : float                 = 0.5
) -> Tensor:
	"""Randomly rotate a tensor image or a batch of tensor images and then
	horizontally flip. Input must be a tensor of shape [C, H, W] or a batch of
	tensors [*, C, H, W].
	
	Args:
		image (Tensor):
			Image to be rotated and flipped.
		angle (float):
			Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
		p (float):
            Probability of the image being flipped. Default: `0.5`.
    
	Returns:
		(PIL Image or Tensor):
			Randomly rotated and flipped image.
	"""
	if torch.rand(1) < p:
		image = rotate(image, angle, interpolation, expand, center, fill)
		return hflip(image)
	return image


def random_rotate_flipud(
	image        : Tensor,
	angle        : float,
	interpolation: InterpolationMode     = InterpolationMode.NEAREST,
	expand       : bool                  = False,
	center       : Optional[list[int]]   = None,
	fill         : Optional[list[float]] = None,
	p            : float                 = 0.5
) -> Tensor:
	"""Rotate a tensor image or a batch of tensor images and then vertically
	flip. Input must be a tensor of shape [C, H, W] or a batch of tensors
	[*, C, H, W].
	
	Args:
		image (Tensor):
			Image to be rotated and flipped.
		angle (float):
			Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        p (float):
            Probability of the image being flipped. Default: `0.5`.
   
    Returns:
		(Tensor):
			Rotated and flipped image.
	"""
	if torch.rand(1) < p:
		image = rotate(image, angle, interpolation, expand, center, fill)
		return vflip(image)
	return image


@TRANSFORMS.register(name="rotate")
class Rotate(torch.nn.Module):
	"""Rotate a tensor image or a batch of tensor images. Input must be a
	tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].
    
    Args:
    	angle (float):
    	    Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		angle        : float,
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		expand       : bool                  = False,
		center       : Optional[list[int]]   = None,
		fill         : Optional[list[float]] = None,
	):
		super().__init__()
		self.angle         = angle
		self.interpolation = interpolation
		self.expand        = expand
		self.center        = center
		self.fill          = fill
	
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		"""
		
		Args:
			image (Tensor):
				Image to be rotated.
				
		Returns:
			(Tensor):
				Frotated image.
		"""
		return rotate(
			image, self.angle, self.interpolation, self.expand, self.center,
			self.fill
		)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rotate_fliplr")
class RotateFliplr(torch.nn.Module):
	"""Rotate a tensor image or a batch of tensor images and then horizontally
	flip. Input must be a tensor of shape [C, H, W] or a batch of tensors
	[*, C, H, W].
	
	Args:
		angle (float):
    	    Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
	"""
	
	# MARK: Magic Functions
	def __init__(
		self,
		angle        : float,
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		expand       : bool                  = False,
		center       : Optional[list[int]]   = None,
		fill         : Optional[list[float]] = None,
	):
		super().__init__()
		self.angle         = angle
		self.interpolation = interpolation
		self.expand        = expand
		self.center        = center
		self.fill          = fill
		
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		"""
		
		Args:
			image (Tensor):
				Image to be rotated and flipped.
				
		Returns:
			(Tensor):
				Rotated and flipped image.
		"""
		return rotate_fliplr(
			image, self.angle, self.interpolation, self.expand, self.center,
			self.fill
		)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="rotate_flipud")
class RotateFlipud(torch.nn.Module):
	"""Rotate a tensor image or a batch of tensor images and then vertically
	flip. Input must be a tensor of shape [C, H, W] or a batch of tensors
	[*, C, H, W].
	
	Args:
		angle (float):
    	    Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
	"""
	
	# MARK: Magic Functions
	def __init__(
		self,
		angle        : float,
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		expand       : bool                  = False,
		center       : Optional[list[int]]   = None,
		fill         : Optional[list[float]] = None,
	):
		super().__init__()
		self.angle         = angle
		self.interpolation = interpolation
		self.expand        = expand
		self.center        = center
		self.fill          = fill
		
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		"""
		
		Args:
			image (Tensor):
				Image to be rotated and flipped.
				
		Returns:
			(Tensor):
				Rotated and flipped image.
		"""
		return rotate_flipud(
			image, self.angle, self.interpolation, self.expand, self.center,
			self.fill
		)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="random_rotate")
class RandomRotate(torch.nn.Module):
	"""Random rotate a tensor image or a batch of tensor images. Input must be
	a tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].
	
	Args:
		angle (float):
    	    Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        p (float):
            Probability of the image being flipped. Default: `0.5`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		angle        : float,
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		expand       : bool                  = False,
		center       : Optional[list[int]]   = None,
		fill         : Optional[list[float]] = None,
		p            : float                 = 0.5
	):
		super().__init__()
		self.angle         = angle
		self.interpolation = interpolation
		self.expand        = expand
		self.center        = center
		self.fill          = fill
		self.p             = p
		
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		"""
		
		Args:
			image (Tensor):
				Image to be rotated.
				
		Returns:
			(Tensor):
				Randomly rotated image.
		"""
		return random_rotate(
			image, self.angle, self.interpolation, self.expand, self.center,
			self.fill, self.p
		)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="random_rotate_fliplr")
class RandomRotateFliplr(torch.nn.Module):
	"""Randomly rotate a tensor image or a batch of tensor images and then
	horizontally flip. Input must be a tensor of shape [C, H, W] or a batch of
	tensors [*, C, H, W].
	
	Args:
		angle (float):
    	    Angle to rotate the image.
		interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        p (float):
            Probability of the image being flipped. Default: `0.5`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		angle        : float,
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		expand       : bool                  = False,
		center       : Optional[list[int]]   = None,
		fill         : Optional[list[float]] = None,
		p            : float                 = 0.5
	):
		super().__init__()
		self.angle         = angle
		self.interpolation = interpolation
		self.expand        = expand
		self.center        = center
		self.fill          = fill
		self.p             = p
		
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		"""
		
		Args:
			image (Tensor):
				Image to be rotated and flipped.
				
		Returns:
			(Tensor):
				Randomly rotated and flipped image.
		"""
		return random_rotate_fliplr(
			image, self.angle, self.interpolation, self.expand, self.center,
			self.fill, self.p
		)


# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="random_rotate_flipud")
class RandomRotateFlipud(torch.nn.Module):
	"""Rotate a tensor image or a batch of tensor images and then vertically
	flip. Input must be a tensor of shape [C, H, W] or a batch of tensors
	[*, C, H, W].
	
	Args:
		angle (float):
    	    Angle to rotate the image.
    	interpolation (InterpolationMode):
    	    Desired interpolation enum defined by
    	    :class:`torchvision.transforms.InterpolationMode`.
    	    Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        expand (bool, optional):
            Optional expansion flag.
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and no
            translation.
        center (sequence, optional):
            Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        p (float):
            Probability of the image being flipped. Default: `0.5`.
	"""
	
	# MARK: Magic Functions
	def __init__(
		self,
		angle        : float,
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		expand       : bool                  = False,
		center       : Optional[list[int]]   = None,
		fill         : Optional[list[float]] = None,
		p            : float                 = 0.5
	):
		super().__init__()
		self.angle         = angle
		self.interpolation = interpolation
		self.expand        = expand
		self.center        = center
		self.fill          = fill
		self.p             = p
		
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		"""
		
		Args:
			image (Tensor):
				Image to be rotated and flipped.
				
		Returns:
			(Tensor):
				Rotated and flipped image.
		"""
		return random_rotate_flipud(
			image, self.angle, self.interpolation, self.expand, self.center,
			self.fill, self.p
		)
