#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A flip is a motion in geometry in which an object is turned over a straight
line to form a mirror image. Every point of an object and the corresponding
point on the image are equidistant from the flip line. A flip is also called
a reflection.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import vflip

from src.factory import TRANSFORMS

__all__ = [
	"hflip",
	"Hflip",
	"vflip",
	"Vflip",
]


# MARK: - Hflip

@TRANSFORMS.register(name="hflip")
@TRANSFORMS.register(name="horizontal_flip")
class Hflip(torch.nn.Module):
	"""Horizontally flip a tensor image or a batch of tensor images. Input must
	be a tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].

	Examples:
		>>> hflip = Hflip()
		>>> input = torch.tensor([[[
		...    [0., 0., 0.],
		...    [0., 0., 0.],
		...    [0., 1., 1.]
		... ]]])
		>>> hflip(input)
		image([[[[0., 0., 0.],
				  [0., 0., 0.],
				  [1., 1., 0.]]]])
	"""
	
	# MARK: Magic Functions
	
	def __repr__(self):
		return self.__class__.__name__
	
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		return hflip(image)


# MARK: - Vflip

@TRANSFORMS.register(name="vflip")
@TRANSFORMS.register(name="vertical_flip")
class Vflip(torch.nn.Module):
	"""Vertically flip a tensor image or a batch of tensor images. Input must
	be a tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].

	Examples:
		>>> vflip = Vflip()
		>>> input = torch.tensor([[[
		...    [0., 0., 0.],
		...    [0., 0., 0.],
		...    [0., 1., 1.]
		... ]]])
		>>> vflip(input)
		image([[[[0., 1., 1.],
				  [0., 0., 0.],
				  [0., 0., 0.]]]])
	"""
	
	# MARK: Magic Functions
	
	def __repr__(self):
		return self.__class__.__name__
	
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		return vflip(image)
