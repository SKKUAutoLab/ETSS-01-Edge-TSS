#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YOLOv5 object_detectors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import platform
from torch import Tensor
import torch
from core.utils.image import is_channel_first
from core.factory.builder import DETECTORS
from core.objects.instance import Instance
from detectors.detector import BaseDetector

# NOTE: add PATH of YOLOv9 source to here
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLO root directory
# if str(ROOT) not in sys.path:
# 	sys.path.append(str(ROOT))  # add ROOT to PATH
# if platform.system() != 'Windows':
# 	ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from motordriver.detectors.yolov9.yolov9.models.common import DetectMultiBackend
from motordriver.detectors.yolov9.yolov9.utils.torch_utils import select_device
from motordriver.detectors.yolov9.yolov9.utils.general import (check_img_size, non_max_suppression, scale_boxes)
from motordriver.detectors.yolov9.yolov9.utils.augmentations import (letterbox)

__all__ = [
	"YOLOv9"
]


# MARK: - YOLOv9

@DETECTORS.register(name="yolov9")
class YOLOv9(BaseDetector):
	"""YOLOv9 object detector."""

	# MARK: Magic Functions

	def __init__(self,
				 name: str = "yolov9",
				 *args, **kwargs):
		super().__init__(name=name, *args, **kwargs)

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""

		# NOTE: Simple check
		if self.weights is None or self.weights == "":
			print("No weights file has been defined!")
			raise ValueError

		# NOTE: Create model
		# Get image size of detector
		if is_channel_first(np.ndarray(self.shape)):
			self.img_size = self.shape[1:]
		else:
			self.img_size = self.shape[:2]

		# NOTE: load model
		# Load model
		device = select_device(self.device)
		self.model = DetectMultiBackend(
			weights = self.weights,
			device  = device
		)

		stride, names, pt = self.model.stride, self.model.names, self.model.pt
		self.img_size     = check_img_size(self.img_size, s=self.stride)  # check image size
		self.model.warmup(imgsz=(1 if pt or self.model.triton else self.batch_size, 3, *self.img_size))  # warmup

	# MARK: Detection

	def detect(self, indexes: np.ndarray, images: np.ndarray) -> list:
		"""Detect objects in the images.

		Args:
			indexes (np.ndarray):
				Image indexes.
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			instances (list):
				List of `Instance` objects.
		"""
		# NOTE: Safety check
		if self.model is None:
			print("Model has not been defined yet!")
			raise NotImplementedError

		# NOTE: Preprocess
		input_imgs = self.preprocess(images=images)
		# NOTE: Forward
		preds     = self.forward(input_imgs)
		# NOTE: Postprocess
		instances = self.postprocess(
			indexes=indexes, images=images, input_imgs=input_imgs, preds=preds
		)
		# NOTE: Suppression
		instances = self.suppress_wrong_labels(instances=instances)

		return instances

	def preprocess(self, images: np.ndarray):
		"""Preprocess the input images to model's input image.

		Args:
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			input (Tensor):
				Models' input.
		"""
		input_imgs = []
		for im0 in images:
			im = letterbox(im0, self.img_size, stride=self.stride, auto=self.model.pt)[0]  # padded resize
			im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
			im = np.ascontiguousarray(im)  # contiguous

			im = torch.from_numpy(im).to(self.model.device)
			im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
			im /= 255  # 0 - 255 to 0.0 - 1.0
			if len(im.shape) == 3:
				im = im[None]  # expand for batch dim
			input_imgs.append(im)

		return input_imgs

	def forward(self, input_imgs: Tensor):
		"""Forward pass.

		Args:
			input_imgs (Tensor):
				Input image of shape [B, C, H, W].

		Returns:
			preds (Tensor):
				Predictions.
		"""
		preds = []
		for input_img in input_imgs:
			preds.append(self.model(input_img, augment=False, visualize=False)[0])
		return preds

	def postprocess(
			self,
			indexes    : np.ndarray,
			images     : np.ndarray,
			input_imgs : Tensor,
			preds      : Tensor,
			*args, **kwargs
	) -> list:
		"""Postprocess the prediction.

		Args:
			indexes (np.ndarray):
				Image indexes.
			images (np.ndarray):
				Images of shape [B, H, W, C].
			input_imgs (Tensor):
				Input image of shape [B, C, H, W].
			preds (Tensor):
				Prediction.

		Returns:
			instances (list):
				List of `Instances` objects.
		"""
		# NOTE: Create Detection objects
		instances = []

		for idx, (im0, im, pred) in enumerate(zip(images, input_imgs, preds)):
			pred = non_max_suppression(
				pred,
				conf_thres = self.min_confidence,
				iou_thres  = self.nms_max_overlap,
				classes    = None,
				agnostic   = False,
				max_det    = 300
			)

			inst = []
			for i, det in enumerate(pred):
				if len(det) < 1:
					continue

				det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

				for *xyxy, conf, cls in reversed(det):
					confident   = float(conf)
					class_id    = int(cls)
					class_label = self.class_labels.get_class_label(
						key="train_id", value=class_id
					)
					inst.append(
						Instance(
							frame_index = indexes[0] + idx,
							bbox        = xyxy,
							confidence  = confident,
							class_label = class_label,
							label       = class_label
						)
					)

			instances.append(inst)
		return instances
