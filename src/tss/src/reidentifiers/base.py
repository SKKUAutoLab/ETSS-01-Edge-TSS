#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class to store persistent objects
"""

from __future__ import annotations

import abc
from typing import Optional

import torch
from PIL import Image

from src.type import Dim2
from src.utils import select_device

__all__ = [
	"BaseReIndentifier"
]


# MARK: - BaseReIndentifier

class BaseReIndentifier(metaclass=abc.ABCMeta):
	"""Base class for all Re-Indentifier.
	"""

	# MARK: Magic Functions

	def __init__(
		self,
		name          : str,
		model_cfg     : Optional[dict] = None,
		reid_backbone : Optional[str]  = None,
		reid_model    : Optional[str]  = None,
		reid_size_test: Optional[Dim2] = None,
		device        : Optional[str]  = None,
		*args, **kwargs
	):
		super().__init__()
		self.name           = name
		self.model          = None
		self.model_cfg      = model_cfg
		self.reid_backbone  = reid_backbone
		self.reid_model     = reid_model
		self.reid_size_test = reid_size_test
		self.device         = select_device(device = device)
		self.val_transforms = None

		# NOTE: Load model
		self.init_model()

	# MARK: Configure

	@abc.abstractmethod
	def init_model(self):
		"""Create and load model from weights."""
		pass

	# MARK: Properties

	# MARK: Extract

	def extract(self, img_path_list: list) -> list:
		"""Detect objects in the images.

		Args:

		Returns:
			instances (list):
				List of `Instance` objects.
		"""
		# NOTE: Safety check
		if self.model is None:
			print("Model has not been defined yet!")
			raise NotImplementedError

		# NOTE: Preprocess
		img       = self.preprocess(img_path_list)
		# NOTE: Forward
		feat      = self.forward(img)
		# NOTE: Postprocess
		instances = self.postprocess(
			img_path_list, feat
		)
		return instances

	@abc.abstractmethod
	def forward(self, img):
		"""Forward pass.

		Args:

		Returns:
			pred (Tensor):
				Predictions.
		"""
		pass

	@abc.abstractmethod
	def preprocess(self, img_path_list):
		"""Preprocess the input images to model's input image.

		Args:
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			input (Tensor):
				Models' input.
		"""
		pass

	@abc.abstractmethod
	def postprocess(
			self,
			img_path_list,
			feat
	) -> list:
		"""Postprocess the prediction.

		Args:

		Returns:
			instances (list):
				List of `Instance` objects.
		"""
		pass

	# MARK: Visualize

	def clear_model_memory(self):
		"""Free the memory of model

		Returns:
			None
		"""
		if self.model is not None:
			self.model.cpu()
			del self.model
			torch.cuda.empty_cache()

