#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class to store persistent objects
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from src.builder import REIDENTIFIER
from src.configuration import models_zoo_dir
from src.reidentifiers.base import BaseReIndentifier
from src.reidentifiers.baseline.config import cfg
from src.reidentifiers.baseline.model import make_model

__all__ = [
	"ReidFeature"
]


@REIDENTIFIER.register(name="re_identifier_feature")
class ReidFeature(BaseReIndentifier):
	"""Extract reid feature."""

	# MARK: Magic Functions

	def __init__(self, name: str = "re_identifier_feature", *args, **kwargs):
		super().__init__(name, *args, **kwargs)

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		self.model, self.reid_cfg = self.build_reid_model()

		# NOTE: Eval
		self.model.to(device=self.device)
		self.model.eval()

		mean = [0.485, 0.456, 0.406]
		std  = [0.229, 0.224, 0.225]
		self.val_transforms = T.Compose([
			T.Resize(self.reid_cfg.INPUT.SIZE_TEST, interpolation=3),
			T.ToTensor(),
			T.Normalize(mean=mean, std=std)
		])

	def build_reid_model(self):
		"""Build the Re-Identifier Model"""
		# NOTE: get the absolute path of file
		abs_file = __file__
		abs_dir = abs_file[:abs_file.rfind("/")]

		cfg.merge_from_file(os.path.join(abs_dir, 'configs/aictest.yml'))
		cfg.INPUT.SIZE_TEST = self.reid_size_test
		cfg.MODEL.NAME = self.reid_backbone

		# DEBUG:
		# print(cfg)
		# print(type(cfg))
		# sys.exit()

		model = make_model(cfg, num_class=100)
		model.load_param(os.path.join(models_zoo_dir, self.reid_model))

		return model, cfg

	# MARK: Extract

	def preprocess(self, img_path_list):
		"""Preprocess the input images to model's input image.

		Args:
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			input (Tensor):
				Models' input.
		"""
		img_batch = []
		for img_path in img_path_list:
			img = Image.open(img_path).convert('RGB')
			img = self.val_transforms(img)
			img = img.unsqueeze(0)
			img_batch.append(img)

		return torch.cat(img_batch, dim=0)

	def forward(self, img):
		"""Extract image feature with given image path.
		Feature shape (2048,) float32.

		Args:
			img_path_list (list):
				list of Absolute path of images
		Returns:
			feat (list)
				list of feature
		"""
		with torch.no_grad():
			img = img.to(self.device)

			flip_feats = False

			if self.reid_cfg.TEST.FLIP_FEATS == 'yes':
				flip_feats = True

			if flip_feats:
				for i in range(2):
					if i == 1:
						inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
						img = img.index_select(3, inv_idx)
						feat1 = self.model(img)
					else:
						feat2 = self.model(img)
				feat = feat2 + feat1
			else:
				feat = self.model(img)

		feat = feat.cpu().detach().numpy()

		return feat

	def postprocess(self, image_path_list, reid_feat_numpy):
		"""Process_input_by_worker_process."""
		feat_dict = {}
		for index, image_path in enumerate(image_path_list):
			feat_dict[image_path] = reid_feat_numpy[index]
		return feat_dict
