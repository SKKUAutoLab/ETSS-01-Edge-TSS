#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
try:
    # 👇️ using Python 3.10+
    from collections.abc import Callable
except ImportError:
    # 👇️ using Python 3.10-
    from collections import Callable
from typing import Optional

import torch.nn
from torch import nn
from torch import Tensor
from torchvision.transforms import InterpolationMode

from src.factory import TRANSFORMS
from src.file import load
from src.type import Transforms_

__all__ = [
	"BaseAugment",
	"BaseAugmentModule",
]


# MARK: - BaseAugment

class BaseAugment(nn.Module, metaclass=ABCMeta):
	r"""Base augmentation perform a sequence of transformations.

	Attributes:
		interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default: `InterpolationMode.NEAREST`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` are
            supported.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
		to_tensor (bool):
			Convert a PIL Image or numpy.ndarray [H, W, C] in the range
			[0, 255] to a torch.FloatTensor of shape [C, H, W] in the range
			[0.0, 1.0]. Default: `True`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		fill         : Optional[list[float]] = None,
		to_tensor    : bool                  = True,
	):
		super().__init__()
		self.interpolation = interpolation
		self.fill          = fill
		self.to_tensor     = to_tensor
	
	# MARK: Configure
	
	@abstractmethod
	def _augmentation_space(
		self, *args, **kwargs
	) -> dict[str, tuple[Tensor, bool]]:
		pass
	
	# MARK: Forward Pass
	
	@abstractmethod
	def forward(self, *args, **kwargs):
		r"""Forward pass to transform the image and target."""
		pass


# MARK: - BaseAugmentModule

class BaseAugmentModule(nn.Module, metaclass=ABCMeta):
	r"""Build a torch.nn.Sequential module of transformations.

	Args:
		transforms (Transforms_, optional):
			Transformation sequential module or list/dict of transformation
			parameters.
		
	Attributes:
		num_transforms (int):
			Number of operations in transformation sequential.
		interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default: `InterpolationMode.NEAREST`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` are
            supported.
        fill (sequence or number, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
		to_tensor (bool):
			Convert a PIL Image or numpy.ndarray [H, W, C] in the range
			[0, 255] to a torch.FloatTensor of shape [C, H, W] in the range
			[0.0, 1.0]. Default: `True`.
		apply_random (bool):
			If `True`, randomly apply a transformation among the list.
			Else, apply all transformations.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		transforms   : Optional[Transforms_],
		interpolation: InterpolationMode     = InterpolationMode.NEAREST,
		fill         : Optional[list[float]] = None,
		to_tensor    : bool                  = True,
		apply_random : bool                  = False,
	):
		super().__init__()
		self.transforms     = transforms
		self.num_transforms = len(list(self.transforms.children()))
		self.interpolation  = interpolation
		self.fill           = fill
		self.to_tensor      = to_tensor
		self.apply_random   = apply_random
	
	# MARK: Properties
	
	@property
	def with_transforms(self) -> bool:
		return hasattr(self, "transforms") and self.transforms is not None
	
	@property
	def transforms(self) -> nn.Sequential:
		r"""Input and target transformation sequence."""
		return self._transforms
	
	@transforms.setter
	def transforms(self, transforms: Optional[Transforms_]):
		self._transforms = self.create_transform(transforms)
		
	# MARK: Configure
	
	@classmethod
	def load_from_file(cls, path: str):
		"""

		Args:
			path (str):
				Filepath to the config file that stores the augmentation
				configs.
		"""
		config = load(path=path)
		if config is None:
			raise ValueError(f"No configs can be found in {path}.")
		return cls(**config)
	
	@staticmethod
	def create_transform(transforms: Optional[Transforms_]) -> nn.Sequential:
		if transforms is None:
			transforms = [nn.Identity()]
		elif isinstance(transforms, (dict, Callable)):
			transforms = [transforms]
		
		if isinstance(transforms, list):
			transforms = [TRANSFORMS.build_from_dict(cfg=t)
			              if isinstance(t, dict) else t for t in transforms]
			transforms = nn.Sequential(*transforms)

		if not isinstance(transforms, torch.nn.Sequential):
			raise ValueError(f"transforms must be of type `torch.nn.Sequential` ."
			                 f"But got: {type(transforms)}.")
		
		return transforms
	
	# MARK: Forward Pass
	
	@abstractmethod
	def forward(self, *args, **kwargs):
		r"""Forward pass to transform the image and target."""
		pass
