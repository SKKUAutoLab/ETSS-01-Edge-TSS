# ==================================================================== #
# Copyright (C) 2022 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ==================================================================== #
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
import threading
import uuid
import glob
import copy
import random
from queue import Queue
from operator import itemgetter
from timeit import default_timer as timer
from typing import Union, Optional

import cv2
import torch
import numpy as np
from tqdm import tqdm

from core.data.class_label import ClassLabels
from core.io.filedir import is_basename
from core.io.filedir import is_json_file
from core.io.frame import FrameLoader
from core.io.frame import FrameWriter
from core.io.picklewrap import PickleLoader
from core.io.video import VideoLoader
from core.utils.rich import console
from core.utils.constants import AppleRGB
from core.objects.instance import Instance
from core.factory.builder import CAMERAS
from core.factory.builder import DETECTORS
from detectors.detector import BaseDetector

from configuration import (
	data_dir,
	config_dir,
	result_dir
)
from cameras.base import BaseCamera

__all__ = [
	"TrafficSafetyCamera"
]


# NOTE: only for ACI23_Track_5
classes_aic24 = ['motorbike',
				 'DHelmet', 'DNoHelmet',
				 'P1Helmet', 'P1NoHelmet',
				 'P2Helmet', 'P2NoHelmet',
				 'P0Helmet', 'P0NoHelmet']


# MARK: - TrafficSafetyCamera


# noinspection PyAttributeOutsideInit

@CAMERAS.register(name="traffic_safety_camera")
class TrafficSafetyCamera(BaseCamera):

	# MARK: Magic Functions

	def __init__(
			self,
			data         : dict,
			dataset      : str,
			name         : str,
			detector     : dict,
			identifier   : dict,
			heuristic    : dict,
			data_loader  : dict,
			data_writer  : Union[FrameWriter,  dict],
			process      : dict,
			id_          : Union[int, str] = uuid.uuid4().int,
			verbose      : bool            = False,
			drawing      : bool            = False,
			queue_size   : int             = 10,
			*args, **kwargs
	):
		"""

		Args:
			dataset (str):
				Dataset name. It is also the name of the directory inside
				`data_dir`.
			subset (str):
				Subset name. One of: [`dataset_a`, `dataset_b`].
			name (str):
				Camera name. It is also the name of the camera's config
				files.
			class_labels (ClassLabels, dict):
				ClassLabels object or a config dictionary.
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
			data_loader (FrameLoader, dict):
				Data loader object or a data loader's config dictionary.
			data_writer (VideoWriter, dict):
				Data writer object or a data writer's config dictionary.
			id_ (int, str):
				Camera's unique ID.
			verbose (bool):
				Verbosity mode. Default: `False`.
			queue_size (int):
				Size of queue store the information
		"""
		super().__init__(id_=id_, dataset=dataset, name=name)
		# NOTE: Init attributes
		self.start_time = None
		self.pbar       = None
		self.detector   = None
		self.identifier = None
		self.heuristic  = None

		# NOTE: Define attributes
		self.process         = process
		self.verbose         = verbose
		self.drawing         = drawing

		# NOTE: Define configurations
		self.data_cfg        = data
		self.detector_cfg    = detector
		self.identifier_cfg  = identifier
		self.heuristic_cfg   = heuristic
		self.data_loader_cfg = data_loader
		self.data_writer_cfg = data_writer

		# NOTE: Queue
		self.frames_queue                 = Queue(maxsize = self.data_loader_cfg['queue_size'])
		self.detections_queue_identifier  = Queue(maxsize = self.detector_cfg['queue_size'])
		self.identifications_queue        = Queue(maxsize = self.identifier_cfg['queue_size'])
		self.writer_queue                 = Queue(maxsize = self.data_writer_cfg['queue_size'])

		# NOTE: Init modules
		self.init_dirs()

		# NOTE: Init for output
		self.init_data_output()

	# MARK: Configure

	def init_dirs(self):
		"""Initialize dirs.
		"""
		self.root_dir    = os.path.join(data_dir)
		self.configs_dir = os.path.join(config_dir)
		self.outputs_dir = os.path.join(result_dir, self.data_writer_cfg["dst"])
		self.video_dir   = os.path.join(self.root_dir, self.data_loader_cfg["data"])
		self.image_dir   = os.path.join(self.root_dir, self.data_loader_cfg["data"])

	def init_class_labels(self, class_labels: Union[ClassLabels, dict]):
		"""Initialize class_labels.

		Args:
			class_labels (ClassLabels, dict):
				ClassLabels object or a config dictionary.
		"""
		if isinstance(class_labels, ClassLabels):
			self.class_labels = class_labels
		elif isinstance(class_labels, dict):
			file = class_labels["file"]
			if is_json_file(file):
				self.class_labels = ClassLabels.create_from_file(file)
			elif is_basename(file):
				file              = os.path.join(self.root_dir, file)
				self.class_labels = ClassLabels.create_from_file(file)
		else:
			file              = os.path.join(self.root_dir, f"class_labels.json")
			self.class_labels = ClassLabels.create_from_file(file)
			print(f"Cannot initialize class_labels from {class_labels}. "
				  f"Attempt to load from {file}.")

	def init_detector(self, detector: Union[BaseDetector, dict]):
		"""Initialize detector.

		Args:
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
		"""
		console.log(f"Initiate Detector.")
		if isinstance(detector, BaseDetector):
			self.detector = detector
		elif isinstance(detector, dict):
			detector["class_labels"] = self.class_labels
			self.detector = DETECTORS.build(**detector)
		else:
			raise ValueError(f"Cannot initialize detector with {detector}.")

	def init_identifier(self, identifier: Union[BaseDetector, dict]):
		"""Initialize identifier.

		Args:
			identifier (BaseDetector, dict):
				Identifier object or a identifier's config dictionary.
		"""

		# DEBUG:
		weight_cfgs = self.identifier_cfg['weights']
		shape_cfgs = self.identifier_cfg['shape']
		self.identifier = []
		for index, (weight_cfg, shape_cfg) in enumerate(zip(weight_cfgs, shape_cfgs)):
			console.log(f"Initiate Identifier. {index}")
			identifier['weights'] = weight_cfg
			identifier['shape']   = shape_cfg
			if isinstance(identifier, BaseDetector):
				self.identifier.append(identifier)
			elif isinstance(identifier, dict):
				identifier["class_labels"] = self.class_labels
				self.identifier.append(DETECTORS.build(**identifier))
			else:
				raise ValueError(f"Cannot initialize detector with {identifier}.")

	def init_data_loader(self, data_loader_cfg: dict):
		"""Initialize data loader.

		Args:
			data_loader_cfg (dict):
				Data loader object or a data loader's config dictionary.
		"""
		if self.process["run_image"]:
			self.data_loader = FrameLoader(data=os.path.join(data_dir, "images", data_loader_cfg["data_path"]), batch_size=data_loader_cfg['batch_size'])
		else:
			self.data_loader = VideoLoader(data=os.path.join(data_dir, "videos", data_loader_cfg["data_path"]), batch_size=data_loader_cfg['batch_size'])

	def check_and_create_folder(self, folder_path):
		"""Check and create the folder to store the result

		Args:
			folder_path (str):
				path to folder
		Returns:
			None
		"""
		if not os.path.isdir(folder_path):
			os.makedirs(folder_path)

	def init_data_output(self):
		"""Initialize data writer.
		"""

		# NOTE: save detections
		self.data_writer_cfg["detector"]  = os.path.join(self.outputs_dir, "detector", self.detector_cfg['folder_out'])
		self.check_and_create_folder(self.data_writer_cfg["detector"])

		# NOTE: save identifier
		self.data_writer_cfg["identifier"] = os.path.join(self.outputs_dir, "identifier", self.identifier_cfg['folder_out'])
		self.check_and_create_folder(self.data_writer_cfg["identifier"])

		# NOTE: save heuristic
		self.data_writer_cfg["heuristic"] = os.path.join(self.outputs_dir, "heuristic", self.heuristic_cfg['folder_out'])
		self.check_and_create_folder(self.data_writer_cfg["heuristic"])

	# MARK: Run

	def run_detector(self):
		"""Run detection model with videos"""
		# NOTE: create directory to store result
		folder_output = f"{self.data_writer_cfg['detector']}/{self.data_loader_cfg['data_path']}/detection/"
		make_directory(folder_output)

		# DEBUG: get labels
		# folder_img_out = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_test/labels_1_classes_filtered_clean/images/"
		# folder_lbl_out = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_test/labels_1_classes_filtered_clean/labels/"
		# make_directory(folder_img_out)
		# make_directory(folder_lbl_out)

		# init value
		height_img, width_img = None, None

		# NOTE: init data loader for load images/video
		self.init_data_loader(self.data_loader_cfg)
		detections_queue_identifier = []
		areas                       = []
		heights                     = []
		widths                      = []

		pbar = tqdm(total=self.data_loader.num_frames, desc=f"Detection: {self.data_loader_cfg['data_path']}")

		# NOTE: run detection
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			for images, indexes, _, _ in self.data_loader:

				if len(indexes) == 0:
					break

				# if finish loading
				if indexes is None:
					break

				# get size of image
				if height_img is None:
					height_img, width_img, _ = images[0].shape

				# NOTE: Detect batch of instances
				batch_instances = self.detector.detect(
					indexes=indexes, images=images
				)

				# NOTE: Process the detection result
				for index_b, (index_image, batch) in enumerate(zip(indexes, batch_instances)):
				# for index_b, batch in enumerate(batch_instances):
					# store result each frame
					batch_detections_identifier = []
					batch_detections_tracker    = []

					# DEBUG: get labels
					# bbox_xyxy_list = []

					# DEBUG: draw
					if self.drawing:
						image_draw = images[index_b].copy()

					# NOTE: Process each detection
					for index_in, instance in enumerate(batch):
						bbox_xyxy = [int(i) for i in instance.bbox]
						crop_id   = [int(index_image), int(index_in)]  # frame_index, bounding_box index

						# if size of bounding box is very small
						# because the heuristic need the bigger bounding box
						# if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 40 \
						# 		or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 40:
						# 	continue

						# NOTE: store for distribution
						# calculate the parameters of bounding box
						if instance.confidence >= 0.1:
							w = abs(bbox_xyxy[2] - bbox_xyxy[0])
							h = abs(bbox_xyxy[3] - bbox_xyxy[1])
							area = w * h
							widths.append(w)
							heights.append(h)
							areas.append(area)

						# DEBUG: get labels
						# bbox_xyxy_list.append(bbox_xyxy)

						# NOTE: crop the bounding box for IDENTIFIER, add 60 or 1.5 scale
						bbox_xyxy  = scaleup_bbox(
							bbox_xyxy,
							height_img,
							width_img,
							ratio   = 1.5,
							padding = 60
						)

						# NOTE: for reduce the processing time
						# only process the bbox_xyxy have value more than 40
						if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 40 or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 40:
							continue

						crop_image = images[index_b][bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]
						detection_result = {
							'roi_uuid'    : instance.roi_uuid,
							'video_name'  : self.data_loader_cfg['data_path'],
							'frame_index' : index_image,
							'image'       : None,
							'bbox'        : np.array((bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3])),
							'class_id'    : instance.class_label["train_id"],
							'class_label' : instance.class_label,
							'label'       : instance.label,
							'id'          : crop_id,
							'confidence'  : instance.confidence,
							'image_size'  : [width_img, height_img]
						}
						# detection_instance = Instance(**detection_result)
						batch_detections_identifier.append(Instance(**detection_result))

						# DEBUG: draw
						if self.drawing:
							image_draw = plot_one_box(
								bbox = bbox_xyxy,
								img  = image_draw,
								label= instance.label.name
							)

						# NOTE: crop the bounding box for TRACKER, add 40 or 1.2 scale
						# bbox_xyxy = [int(i) for i in instance.bbox]
						# bbox_xyxy = scaleup_bbox(
						# 	bbox_xyxy,
						# 	height_img,
						# 	width_img,
						# 	ratio   = 1.0,
						# 	padding = 0
						# )
						# crop_image = images[index_b][bbox_xyxy[1]: bbox_xyxy[3], bbox_xyxy[0]: bbox_xyxy[2]]
						# detection_result = {
						# 	'roi_uuid'    : instance.roi_uuid,
						# 	'video_name'  : self.data_loader_cfg['data_path'],
						# 	'frame_index' : index_image,
						# 	'image'       : crop_image,
						# 	'bbox'        : np.array((bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3])),
						# 	'class_id'    : instance.class_label["train_id"],
						# 	'class_label' : instance.class_label,
						# 	'label'       : instance.label,
						# 	'id'          : crop_id,
						# 	'confidence'  : instance.confidence,
						# 	'image_size'  : [width_img, height_img]
						# }
						# batch_detections_tracker.append(Instance(**detection_result))

					# DEBUG: draw
					if self.drawing:
						cv2.imwrite(os.path.join(folder_output, f"{index_image:04d}.jpg"), image_draw)

					# DEBUG:get labels
					# if int(self.data_loader_cfg['data_path']) == 31:
					# 	with open(os.path.join(folder_lbl_out, f"_{self.data_loader_cfg['data_path']}{index_image:04d}.txt"), "w") as f_write:
					# 		img_h, img_w, c = images[index_b].shape
					# 		for bbox_xyxy_temp in bbox_xyxy_list:
					# 			xyxy_xywh = convert_voc_to_yolo((img_w, img_h), bbox_xyxy_temp)
					# 			f_write.write(f"{0} {xyxy_xywh[0]} {xyxy_xywh[1]} {xyxy_xywh[2]} {xyxy_xywh[3]}\n")
					# 	cv2.imwrite(os.path.join(folder_img_out, f"_{self.data_loader_cfg['data_path']}{index_image:04d}.jpg"), images[index_b])

					# NOTE: Push detections to array
					detections_queue_identifier.append([index_image, images[index_b], batch_detections_identifier])

				# update pbar
				pbar.update(len(indexes))

		pbar.close()

		# NOTE: save pickle
		pickle.dump(
			detections_queue_identifier,
			open(f"{self.data_writer_cfg['detector']}/{self.data_loader_cfg['data_path']}/detections_queue_identifier.pkl", 'wb')
		)

		# NOTE: save json for distribution
		dict_distribution = {
			"widths": {
				"min" : min(widths),
				"max" : max(widths),
				"mean": np.mean(widths),
				"std" : np.std(widths)
			},
			"heights": {
				"min" : min(heights),
				"max" : max(heights),
				"mean": np.mean(heights),
				"std" : np.std(heights)
			},
			"areas": {
				"min" : min(areas),
				"max" : max(areas),
				"mean": np.mean(areas),
				"std" : np.std(areas)
			}
		}
		json_path = os.path.join(self.data_writer_cfg['detector'], self.data_loader_cfg['data_path'], "distribution.json")
		with open(json_path, "w") as json_file:
			json.dump(dict_distribution, json_file)

	def run_identifier(self):
		"""Run identification model"""
		# NOTE: create directory to store result
		folder_output = f"{self.data_writer_cfg['identifier']}/{self.data_loader_cfg['data_path']}/identification/"
		make_directory(folder_output)

		# NOTE: select the right identifier
		dict_distribution = json.load(
			open(f"{self.data_writer_cfg['detector']}/{self.data_loader_cfg['data_path']}/distribution.json"))
		with open(self.identifier_cfg['cluster_weights'], "rb") as f:
			cluster_method = pickle.load(f)
		X = [[
			dict_distribution["areas"]["max"],
			dict_distribution["areas"]["mean"],
			dict_distribution["areas"]["std"]
		]]
		identifier_index = cluster_method.predict(X)[0]

		# DEBUG:
		# identifier_index = 0

		# NOTE: init pickle loader
		pickle_loader = PickleLoader(
			data=f"{self.data_writer_cfg['detector']}/{self.data_loader_cfg['data_path']}/detections_queue_identifier.pkl",
			batch_size=self.identifier_cfg['batch_size']
		)

		identifications_queue = []
		frame_current         = None
		frame_current_index   = None

		pbar = tqdm(total=len(pickle_loader), desc=f"Identification: {self.data_loader_cfg['data_path']} -- Cluster: {identifier_index}")
		indentifier_queue_size = int(self.identifier_cfg['queue_size'])

		# NOTE: Run identification
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			for pickles, indexes_img in pickle_loader:

				for index_frame_dect, frame, batch_detections_full in pickles:

					# NOTE: read the current frame
					if frame_current is None or frame_current_index is None or frame_current_index != index_frame_dect:
						frame_current = cv2.imread(
							os.path.join(
								self.image_dir,
								self.data_loader_cfg['data_path'],
								f"{self.data_loader_cfg['data_path']}{(index_frame_dect + 1):05d}.jpg")
						)
						frame_current_index = index_frame_dect

						# DEBUG: draw
						if self.drawing:
							image_draw = frame_current.copy()

					# NOTE: Process the each detector result
					batch_detections = []
					# store result each crop image
					batch_identifications = []

					# for batch_index, batch_detection in enumerate(tqdm(batch_detections_full, desc=f"Video : {self.data_loader_cfg['data_path']} -- Frame: {index_frame_dect}")):
					for batch_index, batch_detection in enumerate(batch_detections_full):
						if len(batch_detections) == indentifier_queue_size or batch_index == len(batch_detections_full) - 1:

							# Load crop images
							crop_images = []
							indexes     = []
							for detection_instance in batch_detections:
								crop_image = frame_current[
								             detection_instance.bbox[1]: detection_instance.bbox[3],
								             detection_instance.bbox[0]: detection_instance.bbox[2]
								             ]
								crop_images.append(crop_image)
								indexes.append(detection_instance.id[1])

							# if the crop images is empty
							if len(indexes) > 0 and len(crop_images) > 0:
								batch_instances = self.identifier[identifier_index].detect(
									indexes=indexes, images=crop_images
								)
							else:
								continue

							# NOTE: Process the full identify result
							for index_b, (detection_instance, batch_instance) in enumerate(zip(batch_detections, batch_instances)):
								for index_in, instance in enumerate(batch_instance):
									bbox_xyxy     = [int(i) for i in instance.bbox]
									instance_id   = detection_instance.id + [int(index_in)]  # frame_index, bounding_box index, instance_index

									# add the coordinate from crop image to original image
									# DEBUG: comment doan nay neu extract anh nho
									bbox_xyxy[0] += int(detection_instance.bbox[0])
									bbox_xyxy[1] += int(detection_instance.bbox[1])
									bbox_xyxy[2] += int(detection_instance.bbox[0])
									bbox_xyxy[3] += int(detection_instance.bbox[1])

									# if size of bounding box0 is very small
									if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 40 \
											or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 40:
										continue

									# crop the bounding box, add 60 or 1.5 scale
									# bbox_xyxy = scaleup_bbox(
									# 	bbox_xyxy,
									# 	detection_instance.image_size[1],
									# 	detection_instance.image_size[0],
									# 	ratio   = 1.1,
									# 	padding = 20
									# )
									# DEBUG: draw
									if self.drawing:
										image_draw = plot_one_box(
											bbox = bbox_xyxy,
											img  = image_draw,
											color= AppleRGB.values()[instance.class_label["train_id"]],
											label= instance.label.name
										)

									# Extract the instance image
									# if instance.class_label["train_id"] in [1, 2]:  # if the object is driver (DHelmet, DNoHelmet)
									# 	instance_image = frame[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]
									# else:
									# 	instance_image = None

									identification_result = {
										'video_name'    : detection_instance.video_name,
										'frame_index'   : detection_instance.frame_index,
										'bbox'          : np.array((bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3])),
										'class_id'      : instance.class_label["train_id"],
										'id'            : instance_id,
										'confidence'    : (float(detection_instance.confidence) * instance.confidence),
										'image_size'    : detection_instance.image_size
									}

									identification_instance = Instance(**identification_result)
									batch_identifications.append(identification_instance)

							# reset batch_detections
							batch_detections = []
						else:
							batch_detections.append(batch_detection)

					# DEBUG: draw
					if self.drawing:
						cv2.imwrite(os.path.join(folder_output, f"{index_frame_dect:04d}.jpg"), image_draw)

					# Push identifications to array
					identifications_queue.append([index_frame_dect, None, batch_identifications])

				pbar.update(len(indexes_img))

		pbar.close()

		# NOTE: save pickle
		pickle.dump(
			identifications_queue,
			open(f"{self.data_writer_cfg['identifier']}/{self.data_loader_cfg['data_path']}/identifications_queue.pkl", 'wb')
		)

	def run_heuristic(self):
		# NOTE: init parameter
		folder_output = f"{self.data_writer_cfg['heuristic']}/{self.data_loader_cfg['data_path']}/heuristic/"
		make_directory(folder_output)

		# DEBUG: get labels
		# folder_img_out = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_test/labels_9_classes_crop_filtered/images/"
		# folder_lbl_out = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_test/labels_9_classes_crop_filtered/labels/"
		# make_directory(folder_img_out)
		# make_directory(folder_lbl_out)

		# NOTE: init
		detection_pickle_loader = pickle.load(
			open(f"{self.data_writer_cfg['detector']}/{self.data_loader_cfg['data_path']}/detections_queue_identifier.pkl", 'rb'))
		identifications_pickle_loader = pickle.load(
			open(f"{self.data_writer_cfg['identifier']}/{self.data_loader_cfg['data_path']}/identifications_queue.pkl", 'rb'))

		heuristic_queues = []
		frame_current    = None
		frame_index      = None

		pbar = tqdm(total=len(detection_pickle_loader), desc=f"Heuristic: {self.data_loader_cfg['data_path']}")

		# NOTE: Run heuristic
		for index_frame_dect, frame, batch_detections in detection_pickle_loader:
			frame_heuristic_queues = []

			# load all bounding box of detector
			for detection_instance in batch_detections:
				instance_heuristic_queue = []

				# DEBUG: get labels, bounding box image from detector
				# img_crop_w = abs(detection_instance.bbox[2] - detection_instance.bbox[0])
				# img_crop_h = abs(detection_instance.bbox[3] - detection_instance.bbox[1])
				# img_detector_crop         = detection_instance.image.copy()
				# bboxesn = []
				# cls_ids = []

				# load instance in one bounding box
				for index_frame_ident, _, batch_identifications in identifications_pickle_loader:
					if index_frame_ident == index_frame_dect:
						for identification_instance in batch_identifications:
							if detection_instance.id[0] == identification_instance.id[0] and \
									detection_instance.id[1] == identification_instance.id[1]:
								instance_heuristic_queue.append(identification_instance)

								# DEBUG: get labels
								# if float(identification_instance.confidence) >= self.data_writer_cfg['min_confidence']:
								# 	bbox_temp = [
								# 		identification_instance.bbox[0] - detection_instance.bbox[0],
								# 		identification_instance.bbox[1] - detection_instance.bbox[1],
								# 		identification_instance.bbox[2] - detection_instance.bbox[0],
								# 		identification_instance.bbox[3] - detection_instance.bbox[1]
								# 	]
								# 	bboxesn.append(convert_voc_to_yolo((img_crop_w, img_crop_h), bbox_temp))
								# 	cls_ids.append(identification_instance.class_id)

				# NOTE: run heuristic
				# if frame_current is None or frame_index is None or frame_index != index_frame_dect:
				# 	frame_current = cv2.imread(
				# 		os.path.join(
				# 			self.image_dir,
				# 			self.data_loader_cfg['data_path'],
				# 			f"{self.data_loader_cfg['data_path']}{(index_frame_dect + 1):05d}.jpg")
				# 	)
				# 	frame_index = index_frame_dect
				# instance_heuristic_queue = self.heuristic.run(detection_instance, instance_heuristic_queue, frame_current)

				# DEBUG: get labels, filter the result, confident score must higher than x
				# if len(cls_ids) > 1 and img_crop_h >= 40 and img_crop_w >= 40:
				# 	basename_crop = f"{detection_instance.video_name}{detection_instance.id[0]:03d}{detection_instance.id[1]:05d}"
				# 	with open(os.path.join(folder_lbl_out, f"{basename_crop}.txt"), "w") as f_write:
				# 		for cls_id, bboxn in zip(cls_ids, bboxesn):
				# 			f_write.write(f"{int(cls_id)} {float(bboxn[0])} {float(bboxn[1])} {float(bboxn[2])} {float(bboxn[3])}\n")
				# 	cv2.imwrite(os.path.join(folder_img_out, f"{basename_crop}.jpg"), img_detector_crop)

				# save to queue
				frame_heuristic_queues.append(instance_heuristic_queue)

			# overall queue
			heuristic_queues.append(frame_heuristic_queues)

			pbar.update(1)

		pbar.close()

		# NOTE: save pickle
		pickle.dump(
			heuristic_queues,
			open(
				f"{self.data_writer_cfg['heuristic']}/{self.data_loader_cfg['data_path']}/heuristic_queue.pkl",
				'wb')
		)

	def writing_final_result(self, data_path_start, data_path_end):
		# NOTE: run writing
		with open(os.path.join(self.outputs_dir, self.data_writer_cfg["final_file"]), "w") as f_write:
			for video_index in tqdm(range(data_path_start, data_path_end + 1), desc=f"Writing final result: "):
				video_name    = f"{video_index:03d}"
				pkl_path      = f"{self.data_writer_cfg['heuristic']}/{video_name}/heuristic_queue.pkl"
				pickle_loader = pickle.load(open(pkl_path, 'rb'))

				for result_instances_frame in pickle_loader:
					for result_instances_bbox in result_instances_frame:
						for result_instances in result_instances_bbox:

							# <video_id>, <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
							# print(result_instances.id)

							# NOTE: crop the bounding box, add 60 or 1.5 scale
							# result_instances.bbox = scaleup_bbox(
							# 	result_instances.bbox,
							# 	result_instances.image_size[1],
							# 	result_instances.image_size[0],
							# 	ratio=1.15,
							# 	padding=15
							# )

							x = int(result_instances.bbox[0])
							y = int(result_instances.bbox[1])
							w = int(abs(result_instances.bbox[2] - result_instances.bbox[0]))
							h = int(abs(result_instances.bbox[3] - result_instances.bbox[1]))
							conf = float(result_instances.confidence)

							# NOTE: filter the wrong bbox
							# only check the frame from 1 -> 200 in the challenge, based on evaluation code
							if int(result_instances.id[0]) + 1 > 200:
								continue

							# filter the result, confident score must higher than x
							if conf < self.data_writer_cfg['min_confidence']:
								continue

							# Filter width and height
							if w < 40 or h < 40:
								continue

							# fix size
							x = min(x, result_instances.image_size[0], 1920)
							w = min(w, result_instances.image_size[0], 1920)
							y = min(y, result_instances.image_size[1], 1080)
							h = min(h, result_instances.image_size[1], 1080)

							# x, y, w, h
							# frame in process start from 0, but in challenge start from 1
							f_write.write(f"{int(video_index)},"
										  f"{int(result_instances.id[0]) + 1},"
										  f"{x},"
										  f"{y},"
										  f"{w},"
										  f"{h},"
										  f"{int(result_instances.class_id) + 1},"
										  f"{conf:.8f}\n")

	def run(self):
		"""Main run loop."""
		self.run_routine_start()

		# NOTE: running all videos
		data_path_start = int(self.data_loader_cfg['data_path'][0])
		data_path_end   = int(self.data_loader_cfg['data_path'][1])

		for data_path_index in range(data_path_start, data_path_end + 1):
			self.data_loader_cfg['data_path'] = f"{data_path_index:03d}"

			# DEBUG: run only some videos
			# videos_gr_1 = ['003', '009', '012', '016', '032', '043', '045', '046',
			#                '056', '059', '062', '074', '082', '084', '087', '092']
			# if self.data_loader_cfg['data_path'] not in ['045']:
			# 	continue

			# NOTE: detection
			if self.process["function_detection"]:
				self.init_class_labels(class_labels=self.detector_cfg['class_labels'])
				if self.detector is None:
					self.init_detector(detector=self.detector_cfg)
				self.run_detector()

			# NOTE: identification
			if self.process["function_identify"]:
				self.init_class_labels(class_labels=self.identifier_cfg['class_labels'])
				if self.identifier is None:
					self.init_identifier(identifier=self.identifier_cfg)
				self.run_identifier()

			# NOTE: identification
			if self.process["function_heuristic"]:
				# if self.heuristic is None:
				# 	self.init_heuristic(heuristic=self.heuristic_cfg)
				self.run_heuristic()

		# NOTE: writing final result
		if self.process["function_writing_final"]:
			self.writing_final_result(data_path_start, data_path_end)

		self.run_routine_end()

	def run_routine_start(self):
		"""Perform operations when run routine starts. We start the timer."""
		self.start_time = timer()
		if self.verbose:
			cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

	def run_routine_end(self):
		"""Perform operations when run routine ends."""
		# NOTE: clear detector
		if self.detector is not None:
			self.detector.clear_model_memory()
			self.detector = None

		# NOTE: clear identifier
		if self.identifier is not None:
			for identifier in self.identifier:
				identifier.clear_model_memory()
			self.identifier = None

		cv2.destroyAllWindows()
		self.stop_time = timer()
		if self.pbar is not None:
			self.pbar.close()

	def postprocess(self, image: np.ndarray, *args, **kwargs):
		"""Perform some postprocessing operations when a run step end.

		Args:
			image (np.ndarray):
				Image.
		"""
		if not self.verbose and not self.save_image and not self.save_video:
			return

		torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

		elapsed_time = timer() - self.start_time
		if self.verbose:
			# cv2.imshow(self.name, result)
			cv2.waitKey(1)

	# MARK: Visualize

	def draw(
			self,
			drawing     : np.ndarray,
			gmos        : list       = None,
			rois        : list       = None,
			mois        : list       = None,
			elapsed_time: float      = None,
	) -> np.ndarray:
		"""Visualize the results on the drawing.

		Args:
			drawing (np.ndarray):
				Drawing canvas.
			elapsed_time (float):
				Elapsed time per iteration.

		Returns:
			drawing (np.ndarray):
				Drawn canvas.
		"""
		pass


# MARK - Ultilies


def scaleup_bbox(bbox_xyxy, height_img, width_img, ratio, padding):
	"""Scale up 1.2% or +-40

	Args:
		bbox_xyxy (np.ndarray):
		height_img (int):
		width_img (int):
		ratio (float):
		padding (int):

	Returns:
		bbox_xyxy (np.ndarray):

	"""
	cx = 0.5 * bbox_xyxy[0] + 0.5 * bbox_xyxy[2]
	cy = 0.5 * bbox_xyxy[1] + 0.5 * bbox_xyxy[3]
	w = abs(bbox_xyxy[2] - bbox_xyxy[0])
	w = min(w * ratio, w + padding)
	h = abs(bbox_xyxy[3] - bbox_xyxy[1])
	h = min(h * ratio, h + padding)
	bbox_xyxy[0] = int(max(0, cx - 0.5 * w))
	bbox_xyxy[1] = int(max(0, cy - 0.5 * h))
	bbox_xyxy[2] = int(min(width_img - 1, cx + 0.5 * w))
	bbox_xyxy[3] = int(min(height_img - 1, cy + 0.5 * h))
	return bbox_xyxy


def plot_one_box(bbox, img, color=None, label=None, line_thickness=1):
	"""Plots one bounding box on image img

	Returns:

	"""
	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
	cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

	return img


def make_directory(folder_path):
	''' Create directory

	Args:
		folder_path (str, Path): the path to directory
	'''
	if not os.path.isdir(folder_path):
		os.makedirs(folder_path)


def heuristic_processing(frame_heuristic_queue):
	heuristic_queue = frame_heuristic_queue.copy()
	results   = []

	# NOTE: Check driver helmet and NoHelmet id = [2, 3]  [DHelMet, DNoHelmet]
	conf_max = 0
	for det_crop in heuristic_queue:
		if det_crop.id[2] in [2, 3]:
			conf_max = max(conf_max, det_crop.confidence)
	for det_crop in heuristic_queue:
		if (det_crop.id[2] in [2, 3] and det_crop.confidence == conf_max) or \
				det_crop.id[2] not in [2, 3]:
			results.append(det_crop)

	# NOTE: extend, scale up the bounding box size result
	# heuristic_queue = results.copy()
	# results = []
	# for det_crop in heuristic_queue:
	# 	bbox_xyxy    = np.array(det_crop['bbox'])
	# 	# width_img  = det_crop['width_img']
	# 	# height_img = det_crop['height_img']
	# 	width_img  = 1920
	# 	height_img = 1080
	# 	det_crop['bbox'] = scaleup_bbox(bbox_xyxy, height_img, width_img, ratio=1.1, padding=20)
	# 	results.append(det_crop)

	return results


def convert_voc_to_yolo(size, box):
	"""
	Converts bounding box coordinates from VOC format to YOLO format.

	VOC format is defined as (xmin, ymin, xmax, ymax) where:
	xmin and ymin are the coordinates of the top left corner of the bounding box,
	xmax and ymax are the coordinates of the bottom right corner of the bounding box.

	YOLO format is defined as (x, y, w, h) where:
	x and y are the coordinates of the center of the bounding box,
	w and h are the width and height of the bounding box respectively.

	All values are normalized in respect to the image size.

	Args:
		size (tuple):
			A tuple containing the size of the image in the format (width, height).
		box (tuple):
			A tuple containing the bounding box coordinates in VOC format.

	Returns:
		list:
			A list containing the bounding box coordinates in YOLO format.
	"""
	dw = 1./(size[0])
	dh = 1./(size[1])
	x  = (box[0] + box[2])/2.0 - 1
	y  = (box[1] + box[3])/2.0 - 1
	w  = float(abs(box[2] - box[0]))
	h  = float(abs(box[3] - box[1]))
	x  = x * dw
	w  = w * dw
	y  = y * dh
	h  = h * dh
	return [x,y,w,h]
