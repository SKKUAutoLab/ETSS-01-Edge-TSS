#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
import uuid
import glob
from operator import itemgetter
from timeit import default_timer as timer
from typing import Union

import pickle
import cv2
import torch
import numpy as np
from tqdm import tqdm

from core.data.class_label import ClassLabels
from core.io.filedir import is_basename
from core.io.filedir import is_json_file
from core.io.filedir import is_stem
from core.utils.bbox import bbox_xyxy_to_cxcywh_norm
from core.utils.rich import console
from core.utils.constants import AppleRGB
from core.io.frame import FrameLoader
from core.io.frame import FrameWriter
from core.io.video import is_video_file
from core.io.video import VideoLoader
from core.io.picklewrap import PickleLoader
from core.factory.builder import CAMERAS
from core.factory.builder import DETECTORS
from detectors.base import BaseDetector
from configuration import (
	data_dir,
	config_dir
)
from cameras.base import BaseCamera

__all__ = [
	"AICTrafficSafetyCameraS2"
]

# MARK: - AICTrafficSafetyCameraS2


# noinspection PyAttributeOutsideInit
@CAMERAS.register(name="aic_traffic_safety_camera_solution_2")
class AICTrafficSafetyCameraS2(BaseCamera):

	# MARK: Magic Functions

	def __init__(
			self,
			data         : dict,
			dataset      : str,
			name         : str,
			detector     : Union[BaseDetector, dict],
			identifier   : dict,
			data_loader  : dict,
			data_writer  : Union[FrameWriter,  dict],
			process      : dict,
			id_          : Union[int, str] = uuid.uuid4().int,
			verbose      : bool            = False,
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
		"""
		super().__init__(id_=id_, dataset=dataset, name=name)
		self.process      = process
		self.verbose      = verbose

		self.data_cfg        = data
		self.detector_cfg    = detector
		self.identifier_cfg  = identifier
		self.data_loader_cfg = data_loader
		self.data_writer_cfg = data_writer

		self.init_dirs()
		self.init_data_writer(data_writer_cfg=self.data_writer_cfg)

		if self.process["function_dets"]:
			self.init_class_labels(class_labels=self.detector_cfg['class_labels'])
			self.init_detector(detector=detector)
		if self.process["function_identify"]:
			self.init_class_labels(class_labels=self.identifier_cfg['class_labels'])
			self.init_identifier(identifier=identifier)

		self.start_time = None
		self.pbar       = None

	# MARK: Configure

	def init_dirs(self):
		"""Initialize dirs.

		Returns:

		"""
		self.root_dir    = os.path.join(data_dir)
		self.configs_dir = os.path.join(config_dir)
		self.outputs_dir = os.path.join(self.root_dir, self.data_writer_cfg["dst"])
		self.video_dir   = os.path.join(self.root_dir, self.data_loader_cfg["data"])

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

	def init_identifier(self, identifier: dict):
		"""Initialize identifier.

		Args:
			identifier (dict):
				Identifier object or a identifier's config dictionary.
		"""
		console.log(f"Initiate Identifier.")
		if isinstance(identifier, BaseDetector):
			self.identifier = identifier
		elif isinstance(identifier, dict):
			identifier["class_labels"] = self.class_labels
			self.identifier = DETECTORS.build(**identifier)
		else:
			raise ValueError(f"Cannot initialize identifier with {identifier}.")

	def init_data_loader(self, data_loader_cfg: dict):
		"""Initialize data loader.

		Args:
			data_loader_cfg (dict):
				Data loader object or a data loader's config dictionary.
		"""
		# if self.process["function_dets"]:
		# 	return sorted(glob.glob(os.path.join(self.video_dir, self.data_cfg["type"])))
		pass

	def check_and_create_folder(self, attr, data_writer_cfg: dict):
		"""CHeck and create the folder to store the result

		Args:
			attr (str):
				the type of function/saving/creating
			data_writer_cfg (dict):
				configuration of camera
		Returns:
			None
		"""
		path = os.path.join(self.outputs_dir, f"{data_writer_cfg[attr]}")
		if not os.path.isdir(path):
			os.makedirs(path)
		data_writer_cfg[attr] = path

	def init_data_writer(self, data_writer_cfg: dict):
		"""Initialize data writer.

		Args:
			data_writer_cfg (FrameWriter, dict):
				Data writer object or a data writer's config dictionary.
		"""
		# NOTE: save detections crop
		data_writer_cfg["dets_crop_pkl"] = f'{data_writer_cfg["dets_crop_pkl"]}/{self.detector_cfg["folder_out"]}'
		self.check_and_create_folder("dets_crop_pkl", data_writer_cfg=data_writer_cfg)

		# NOTE: save heuristic detections
		data_writer_cfg["dets_crop_heuristic_pkl"] = f'{data_writer_cfg["dets_crop_heuristic_pkl"]}/{self.detector_cfg["folder_out"]}'
		self.check_and_create_folder("dets_crop_heuristic_pkl", data_writer_cfg=data_writer_cfg)

		# NOTE: save full detail detection
		data_writer_cfg["dets_full_pkl"] = f'{data_writer_cfg["dets_full_pkl"]}/{self.identifier_cfg["folder_out"]}'
		self.check_and_create_folder("dets_full_pkl", data_writer_cfg=data_writer_cfg)

	# MARK: Run

	def run_detection_image(self):
		"""Run detection model with images
		"""
		# NOTE: Load dataset
		self.data_loader_cfg["batch_size"] = self.detector_cfg["batch_size"]
		list_dir = sorted(os.listdir(self.video_dir))

		for dir in tqdm(list_dir, desc="Detection process: "):

			# DEBUG: run 002 video
			# if not "002" in dir:
			# 	continue

			dir_path       = os.path.join(self.video_dir, dir)
			basename_noext = dir
			index_image    = 0
			out_dict       = []
			height_img, width_img = None, None

			if not os.path.isdir(dir_path):
				continue

			data_loader = FrameLoader(data=dir_path, batch_size=self.data_loader_cfg["batch_size"])
			pbar_video  = tqdm(total=len(data_loader), desc=f"Video {basename_noext}:")

			# NOTE: run each video
			for images, indexes, _, _ in data_loader:
				# NOTE: pre process
				# if finish loading
				if len(indexes) == 0:
					break

				# get size of image
				if height_img is None:
					height_img, width_img, _ = images[0].shape

				# NOTE: Detect batch of instances
				batch_instances = self.detector.detect(
					indexes=indexes, images=images
				)

				# NOTE: Write the detection result
				# for index_b, (index_image, batch) in enumerate(zip(indexes, batch_instances)):
				for index_b, batch in enumerate(batch_instances):
					image_draw = images[index_b].copy()
					index_image += 1
					name_index_image = f"{index_image:06d}"

					for index_in, instance in enumerate(batch):
						name_index_in = f"{index_in:08d}"
						bbox_xyxy = [int(i) for i in instance.bbox]

						# if size of bounding box is very small
						# because the heuristic need the bigger bounding box
						if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 40 \
								or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 40:
							continue

						# NOTE: crop the bounding box, add 60 or 1.5 scale
						bbox_xyxy = scaleup_bbox(bbox_xyxy, height_img, width_img, ratio=1.5, padding=60)
						crop_image = images[index_b][bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]

						# DEBUG:
						# if instance.confidence < 0.1:
						# 	continue
						# print(bbox_xyxy)
						# cv2.rectangle(image_draw, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (125, 125, 125), 4, cv2.LINE_AA)  # filled



						result_dict = {
							'video_name': basename_noext,
							'frame_id'  : name_index_image,
							'crop_img'  : crop_image,
							'bbox'      : (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
							'class_id'  : instance.class_label["train_id"],
							'id'        : instance.class_label["id"],
							'conf'      : instance.confidence,
							'width_img' : width_img,
							'height_img': height_img
						}
						out_dict.append(result_dict)

					# DEBUG:
					# cv2.imwrite(
					# 	f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test/outputs_s2_v8_det_v8_iden/dets_crop_images/" \
					# 	f"{name_index_image}.jpg",
					# 	image_draw
					# )

				pbar_video.update(len(indexes))

			# NOTE: save pickle
			pickle.dump(
				out_dict,
				open(f"{os.path.join(self.data_writer_cfg['dets_crop_pkl'], basename_noext)}.pkl", 'wb')
			)

			pbar_video.close()

			# DEBUG: run 1 video
			# break

	def run_detection_video(self):
		"""Run detection model with videos
		"""
		# NOTE: Load dataset
		self.data_loader_cfg["batch_size"] = self.detector_cfg["batch_size"]
		data_loader = sorted(glob.glob(os.path.join(self.video_dir, self.data_cfg["type"])))

		# NOTE: run detection
		pbar = tqdm(total=len(data_loader), desc="Detection process: ")
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			for video_path in data_loader:
				# Init parameter
				basename       = os.path.basename(video_path)
				basename_noext = os.path.splitext(basename)[0]
				height_img    , width_img = None, None
				index_image    = 0
				out_dict       = []

				# DEBUG: run 002 video
				# if not "002" in basename_noext:
				# 	continue

				video_loader = VideoLoader(data=video_path, batch_size=self.data_loader_cfg["batch_size"])
				pbar_video = tqdm(total=len(video_loader), desc=f"Video {basename}:")

				# NOTE: run each video
				for images, indexes, _, _ in video_loader:
					# NOTE: pre process
					# if finish loading
					if len(indexes) == 0:
						break

					# get size of image
					if height_img is None:
						height_img, width_img, _ = images[0].shape

					# NOTE: Detect batch of instances
					batch_instances = self.detector.detect(
						indexes=indexes, images=images
					)

					# NOTE: Write the detection result
					for index_b, batch in enumerate(batch_instances):
						image_draw       = images[index_b].copy()
						index_image     += 1
						name_index_image = f"{index_image:06d}"

						for index_in, instance in enumerate(batch):
							name_index_in = f"{index_in:08d}"
							bbox_xyxy     = [int(i) for i in instance.bbox]

							# if size of bounding box is very small
							# because the heuristic need the bigger bounding box
							if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 40  \
									or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 40:
								continue

							# NOTE: crop the bounding box, add 60 or 1.5 scale
							bbox_xyxy = scaleup_bbox(bbox_xyxy, height_img, width_img, ratio=1.5, padding=60)
							crop_image = images[index_b][bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]

							# DEBUG:
							# cv2.rectangle(image_draw, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]),
							# 			  (125, 125, 125), 4, cv2.LINE_AA)  # filled

							result_dict = {
								'video_name' : basename_noext,
								'frame_id'   : name_index_image,
								'crop_img'   : crop_image,
								'bbox'       : (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
								'class_id'   : instance.class_label["train_id"],
								'id'         : instance.class_label["id"],
								'conf'       : instance.confidence,
								'width_img'  : width_img,
								'height_img' : height_img
							}
							out_dict.append(result_dict)

						# DEBUG:
						# cv2.imwrite(
						# 	f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test/outputs_s2_v8_det_v8_iden/dets_crop_images/" \
						# 	f"{name_index_image}.jpg",
						# 	image_draw
						# )

					pbar_video.update(len(indexes))

				# NOTE: save pickle
				pickle.dump(
					out_dict,
					open(f"{os.path.join(self.data_writer_cfg['dets_crop_pkl'], basename_noext)}.pkl", 'wb')
				)

				# Post process
				del video_loader
				pbar_video.close()
				pbar.update(1)

				# DEBUG: run 1 video
				# break

		pbar.close()

	def run_identification_scale(self):
		"""Run detection model

		Returns:

		"""
		# NOTE: Load dataset
		self.data_loader_cfg["batch_size"] = self.identifier_cfg["batch_size"]
		data_loader = sorted(glob.glob(os.path.join(self.data_writer_cfg['dets_crop_pkl'], "*.pkl")))

		# NOTE: Run identification
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			for pickle_path in data_loader:

				# Init parameter
				basename       = os.path.basename(pickle_path)
				basename_noext = os.path.splitext(basename)[0]
				out_dict       = []

				# DEBUG:
				# if "029" not in basename_noext:
				# 	continue

				# Load pickle
				pickle_loader = PickleLoader(data=pickle_path, batch_size=self.data_loader_cfg["batch_size"])
				# dets_crop = pickle.load(open(pickle_path, 'rb'))
				pbar_pickle = tqdm(total=len(pickle_loader), desc=f"Identifying {basename_noext}: ")
				# NOTE: run each video
				for pickles, indexes in pickle_loader:

					crop_images = []
					# Load crop images
					for pkl in pickles:
						crop_images.append(pkl['crop_img'])

					# NOTE: Identify batch of instances
					batch_instances = self.identifier.detect(
						indexes=indexes, images=crop_images
					)

					# NOTE: Write the full detection result
					for index_b, (crop_dict, batch) in enumerate(zip(pickles, batch_instances)):
						for index_in, instance in enumerate(batch):
							bbox_xyxy     = [int(i) for i in instance.bbox]

							# NOTE: add the coordinate from crop image to original image
							# DEBUG: comment doan nay neu extract anh nho
							bbox_xyxy[0] += int(crop_dict["bbox"][0])
							bbox_xyxy[1] += int(crop_dict["bbox"][1])
							bbox_xyxy[2] += int(crop_dict["bbox"][0])
							bbox_xyxy[3] += int(crop_dict["bbox"][1])

							# if size of bounding box is very small
							if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 40 \
									or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 40:
								continue

							result_dict = {
								'video_name': crop_dict['video_name'],
								'frame_id'  : crop_dict['frame_id'],
								'crop_id'   : index_b,
								'crop_img'  : crop_dict['crop_img'],
								'bbox'      : (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
								'class_id'  : instance.class_label["train_id"],
								'id'        : instance.class_label["id"],
								'conf'      : (float(crop_dict["conf"]) * instance.confidence),
								'width_img' : crop_dict['width_img'],
								'height_img': crop_dict['height_img']
							}
							# result_dict = {
							# 	'video_name': crop_dict['video_name'],
							# 	'frame_id'  : crop_dict['frame_id'],
							# 	'crop_id'   : index_b,
							# 	'bbox'      : (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
							# 	'class_id'  : instance.class_label["train_id"],
							# 	'id'        : instance.class_label["id"],
							# 	'conf'      : (float(crop_dict["conf"]) * instance.confidence),
							# 	'width_img' : crop_dict['width_img'],
							# 	'height_img': crop_dict['height_img']
							# }
							out_dict.append(result_dict)

					pbar_pickle.update(len(indexes))

				# Post process
				pbar_pickle.close()

				# NOTE: save pickle
				pickle.dump(
					out_dict,
					open(f"{os.path.join(self.data_writer_cfg['dets_full_pkl'], basename_noext)}.pkl", 'wb')
				)

				# DEBUG: run 1 video
				# break

	def run_identification_padding(self):
		"""Run identification model

		Returns:

		"""
		# NOTE: Load dataset
		self.data_loader_cfg["batch_size"] = self.identifier_cfg["batch_size"]
		data_loader = sorted(glob.glob(os.path.join(self.data_writer_cfg['dets_crop_pkl'], "*.pkl")))

		# NOTE: run detection
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			for pickle_path in tqdm(data_loader, desc="Identification process"):
				# Init parameter
				basename       = os.path.basename(pickle_path)
				basename_noext = os.path.splitext(basename)[0]
				out_dict       = []

				# DEBUG:
				# if "029" not in basename_noext:
				# 	continue

				# Load pickle
				dets_crop = pickle.load(open(pickle_path, 'rb'))
				for index_crop, crop_dict in enumerate(tqdm(dets_crop, desc=f"Identifying {basename_noext}: ")):

					# Load parameter
					crop_image = crop_dict['crop_img']

					# Add crop into black image
					img = np.zeros((crop_dict['height_img'], crop_dict['width_img'], 3), np.uint8)
					img[crop_dict["bbox"][1]:crop_dict["bbox"][1] + crop_image.shape[0], crop_dict["bbox"][0]:crop_dict["bbox"][0] + crop_image.shape[1]] = crop_image

					# NOTE: Identify batch of instances
					batch_instances = self.identifier.detect(
						indexes=[index_crop], images=[img]
					)

					# NOTE: Write the full detection result
					for index_b, batch in enumerate(batch_instances):
						for index_in, instance in enumerate(batch):
							bbox_xyxy     = [int(i) for i in instance.bbox]

							# if size of bounding box is very small
							if abs(bbox_xyxy[2] - bbox_xyxy[0]) < 2  \
									or abs(bbox_xyxy[3] - bbox_xyxy[1]) < 2:
								continue

							result_dict = {
								'video_name': crop_dict['video_name'],
								'frame_id'  : crop_dict['frame_id'],
								'crop_id'   : index_b,
								'crop_img'  : crop_dict['crop_img'],
								'bbox'      : (bbox_xyxy[0]                                   , bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
								'class_id'  : instance.class_label["train_id"],
								'id'        : instance.class_label["id"],
								'conf'      : (float(crop_dict["conf"]) * instance.confidence),
								'width_img' : crop_dict['width_img'],
								'height_img': crop_dict['height_img']
							}
							# result_dict = {
							# 	'video_name': crop_dict['video_name'],
							# 	'frame_id'  : crop_dict['frame_id'],
							#   'crop_id'   : index_b,
							# 	'bbox'      : (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
							# 	'class_id'  : instance.class_label["train_id"],
							# 	'id'        : instance.class_label["id"],
							# 	'conf'      : (float(crop_dict["conf"]) * instance.confidence),
							# 	'width_img' : crop_dict['width_img'],
							# 	'height_img': crop_dict['height_img']
							# }
							out_dict.append(result_dict)

				# NOTE: save pickle
				pickle.dump(
					out_dict,
					open(f"{os.path.join(self.data_writer_cfg['dets_full_pkl'], basename_noext)}.pkl", 'wb')
				)

				# DEBUG: run 1 video
				# break

	def run_heuristic_filter(self):
		"""Add more object in result

		Returns:

		"""
		data_loader = sorted(glob.glob(os.path.join(self.data_writer_cfg['dets_full_pkl'],'*.pkl')))
		for pkl_path in tqdm(data_loader, desc="Heuristic process"):
			# NOTE: Init parameter
			basename = os.path.basename(pkl_path)
			basename_noext = os.path.splitext(basename)[0]

			# NOTE: load pickle
			dets_crop = pickle.load(open(pkl_path, 'rb'))

			# NOTE: process pickle
			out_dict       = []  # the output dict
			img_crop_index = set()
			for det_crop in tqdm(dets_crop, desc=f"Process {basename_noext}"):

				# NOTE: filter the result, confident score must higher than x
				if det_crop['conf'] < self.data_writer_cfg['min_confidence']:
					continue

				# get index crop
				baseindex = f"{det_crop['frame_id']}_{det_crop['crop_id']}"

				if baseindex not in img_crop_index:
					img_crop_index.add(baseindex)
					in_dict = []

					# Get list of dict
					for det_crop_temp in dets_crop:
						baseindex_temp = f"{det_crop_temp['frame_id']}_{det_crop_temp['crop_id']}"
						if baseindex_temp == baseindex:
							in_dict.append(det_crop_temp)

					# heuristic process
					results_dict = heuristic_processing(in_dict)

					# output process
					for result_dict in results_dict:
						out_dict.append(result_dict)

			# NOTE: save pickle
			pickle.dump(
				out_dict,
				open(f"{os.path.join(self.data_writer_cfg['dets_crop_heuristic_pkl'], basename_noext)}.pkl", 'wb')
			)

	def run_write_final_result(self):
		"""Group and write the final result

		Returns:

		"""
		# [dataset]/[where the final pickle]/[detector]/*.pkl
		# data_loader = sorted(glob.glob(os.path.join(self.data_writer_cfg['dets_full_pkl'], "*.pkl")))
		data_loader = sorted(glob.glob(os.path.join(
			self.outputs_dir,
			self.data_writer_cfg['final_result'],
			self.identifier_cfg['folder_out'],
			'*.pkl'
		)))

		# NOTE: run writing
		with open(os.path.join(self.outputs_dir, self.data_writer_cfg["final_file"]), "w") as f_write:
			for pkl_path in tqdm(data_loader, desc="Writing process"):
				dets_crop = pickle.load(open(pkl_path, 'rb'))
				# dets_crop = sorted(dets_crop, key=itemgetter('frame_id', 'conf'), reverse=True)
				# sorted frame_id increase, conf decrease
				dets_crop = sorted(dets_crop, key=lambda x: (x['frame_id'], - x['conf']))
				for result_dict in dets_crop:
					# <video_id>, <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
					w    = int(abs(result_dict['bbox'][2] - result_dict['bbox'][0]))
					h    = int(abs(result_dict['bbox'][3] - result_dict['bbox'][1]))
					conf = float(result_dict['conf'])

					# NOTE: filter the result, confident score must higher than x
					if conf < self.data_writer_cfg['min_confidence']:
						continue

					# NOTE: Filter width and height
					if w < 40 or h < 40:
						continue

					# x, y, w, h
					f_write.write(f"{int(result_dict['video_name'])},"
								f"{int(result_dict['frame_id'])},"
								f"{int(result_dict['bbox'][0])},"
								f"{int(result_dict['bbox'][1])},"
								f"{w},"
								f"{h},"
								f"{int(result_dict['id'])},"
							  	f"{conf:.8f}\n")

	def run_track(self):
		"""Run tracking

			Returns:
		"""
		data_loader = sorted(glob.glob(os.path.join(self.data_writer_cfg['dets_crop_heuristic_pkl'], '*.pkl')))
		for pkl_path in tqdm(data_loader, desc="Heuristic process"):
			# NOTE: Init parameter
			basename = os.path.basename(pkl_path)
			basename_noext = os.path.splitext(basename)[0]

			# NOTE: load pickle
			dets_crop = pickle.load(open(pkl_path, 'rb'))

			for det_crop in tqdm(dets_crop, desc=f"Process {basename_noext}"):
				pass

	def run(self):
		"""Main run loop."""
		self.run_routine_start()

		# NOTE: run detection
		if self.process["function_dets"]:
			if self.process["run_image"]:
				self.run_detection_image()
			else:
				self.run_detection_video()
			self.detector.clear_model_memory()
			self.detector = None

		# NOTE: run identification
		if self.process["function_identify"]:
			self.run_identification_scale()
			# self.run_identification_padding()

		# NOTE: run heuristic process
		if self.process["function_heuristic"]:
			self.run_heuristic_filter()

		# NOTE: run write final result
		if self.process["function_write_final"]:
			self.run_write_final_result()

		self.run_routine_end()

	def run_routine_start(self):
		"""Perform operations when run routine starts. We start the timer."""
		self.start_time = timer()
		if self.verbose:
			cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

	def run_routine_end(self):
		"""Perform operations when run routine ends."""
		cv2.destroyAllWindows()
		self.stop_time = timer()

	def postprocess(self, image: np.ndarray, *args, **kwargs):
		"""Perform some postprocessing operations when a run step end.

		Args:
			image (np.ndarray):
				Image.
		"""
		if not self.verbose and not self.save_image and not self.save_video:
			return

		elapsed_time = timer() - self.start_time
		if self.verbose:
			# cv2.imshow(self.name, result)
			cv2.waitKey(1)

	# MARK: Visualize

	def draw(self, drawing: np.ndarray, elapsed_time: float) -> np.ndarray:
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
		return drawing


# MARK - Ultilies

def heuristic_processing(in_dict):
	dict_temp = in_dict.copy()
	results   = []

	# NOTE: Check driver helmet and NoHelmet id = [2, 3]  [DHelMet, DNoHelmet]
	conf_max = 0
	for det_crop in dict_temp:
		if det_crop['id'] in [2, 3]:
			conf_max = max(conf_max, det_crop['conf'])
	for det_crop in dict_temp:
		if (det_crop['id'] in [2, 3] and det_crop['conf'] == conf_max) or \
				det_crop['id'] not in [2, 3]:
			results.append(det_crop)

	# NOTE: extend, scale up the bounding box size result
	dict_temp = results.copy()
	results = []
	for det_crop in dict_temp:
		bbox_xyxy    = np.array(det_crop['bbox'])
		# width_img  = det_crop['width_img']
		# height_img = det_crop['height_img']
		width_img  = 1920
		height_img = 1080
		det_crop['bbox'] = scaleup_bbox(bbox_xyxy, height_img, width_img, ratio=1.1, padding=20)
		results.append(det_crop)

	return results


def scaleup_bbox(bbox_xyxy, height_img, width_img, ratio, padding):
	"""Scale up 1.2% or +-40

	Args:
		bbox_xyxy:
		height_img:
		width_img:

	Returns:

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
