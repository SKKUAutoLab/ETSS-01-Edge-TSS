#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Label handler for YOLO label/data format.
"""

from __future__ import annotations

import numpy as np

from src.factory import LABEL_HANDLERS
from src.file import is_txt_file
from src.vision import bbox_area
from src.vision import bbox_cxcywh_norm_to_xyxy
from .base import BaseLabelHandler
from ..data_class import ImageInfo
from ..data_class import ObjectAnnotation
from ..data_class import VisionData

__all__ = [
	"YoloLabelHandler"
]


# MARK: - YoloLabelHandler

@LABEL_HANDLERS.register(name="yolo")
class YoloLabelHandler(BaseLabelHandler):
	"""Handler for loading and dumping labels from Yolo label format to our
	custom label format defined in `src.data.vision_data`.
	
	YOLO format:
		<object_category> <x_center> <y_center> <bbox_width> <bbox_height>
		<score> ...
		
		Where:
			<object_category> : Object category indicates the type of
							    annotated object.
			<x_center_norm>   : Fx coordinate of the center of rectangle.
			<y_center_norm>   : Fy coordinate of the center of rectangle.
			<bbox_width_norm> : Width in pixels of the predicted object
								bounding box.
			<bbox_height_norm>: Height in pixels of the predicted object
								bounding box.
			<score>           : Fscore in the DETECTION result file
								indicates the confidence of the predicted
							    bounding box enclosing an object instance.
	"""
	
	# MARK: Load
	
	def load_from_file(
		self, image_path: str, label_path: str, **kwargs
	) -> VisionData:
		"""Load data from file.

		Args:
			image_path (str):
				Image file.
			label_path (str):
				Label file.
				
		Return:
			visual_data (VisualData):
				A `VisualData` item.
		"""
		# NOTE: Parse image info
		image_info = ImageInfo.from_file(image_path=image_path)
		shape0     = image_info.shape0
		
		# NOTE: Load content from file
		if is_txt_file(path=label_path):
			with open(label_path, "r") as f:
				labels = np.array([x.split() for x in f.read().splitlines()],
								  np.float32)  # labels
		if len(labels) == 0:
			return VisionData(image_info=image_info)
		
		# NOTE: Parse all annotations
		objs = []
		for i, l in enumerate(labels):
			class_id 		 = int(l[0])
			bbox_cxcywh_norm = l[1:5]
			bbox_xyxy		 = bbox_cxcywh_norm_to_xyxy(
				bbox_cxcywh_norm, shape0[0], shape0[1]
			)
			confidence 		 = l[5]
			area      	     = bbox_area(bbox_xyxy)
			objs.append(
				ObjectAnnotation(
					class_id   = class_id,
					bbox       = bbox_cxcywh_norm,
					area       = area,
					confidence = confidence
				)
			)
			
		return VisionData(image_info=image_info, objects=objs)
	
	# MARK: Dump
	
	def dump_to_file(self, data: VisionData, path: str, **kwargs):
		"""Dump data from object to file.
		
		Args:
			data (VisualData):
				`VisualData` item.
			path (str):
				Label filepath to dump the data.
		"""
		if not is_txt_file(path=path):
			path += ".txt"
		
		# NOTE: Dump to file
		with open(path, "w") as f:
			for b in data.bbox_labels:
				ss = f"{b[1]} {b[2]} {b[3]} {b[4]} {b[5]} {b[6]}\n"
				f.writelines(ss)
