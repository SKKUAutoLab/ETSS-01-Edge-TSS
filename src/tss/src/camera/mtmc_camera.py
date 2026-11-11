#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import colorsys
import glob
import os
import pickle
import sys
import time
import uuid
from operator import attrgetter, itemgetter
from timeit import default_timer as timer
from typing import Optional, Union

import cv2
import numpy as np
import torch
from loguru import logger
from sklearn import preprocessing
from src.builder import CAMERAS, DETECTORS, REIDENTIFIER, TRACKERS
from src.camera.base import BaseCamera
from src.configuration import data_dir
from src.data import ClassLabels
from src.detectors import BaseDetector
from src.file import is_basename, is_json_file, is_stem
from src.io import MTMCDrawer
from src.matchers import ReidMatching
from src.reidentifiers import BaseReIndentifier
from src.trackers import BaseTracker
from src.trackers.utils.filter import FilterBoxes
from src.type import is_list_of
from src.vision import (
	AppleRGB,
	FrameLoader,
	FrameWriter,
	bbox_xyxy_to_cxcywh_norm,
	is_video_file,
)
from torchvision.ops import nms
from tqdm import tqdm

# MARK: - MTMCTrackingCamera



@CAMERAS.register(name="aic_mtmc_camera")
class MTMCTrackingCamera(BaseCamera):

	# MARK: Magic Functions

	def __init__(
			self,
			data          : str,
			dataset       : str,
			subset        : str,
			name          : str,
			class_labels  : ClassLabels       | dict,
			detector      : BaseDetector      | dict,
			tracker       : BaseTracker       | dict,
			reidentifier  : BaseReIndentifier | dict,
			matching      : ReidMatching      | dict,
			data_loader   : FrameLoader       | dict,
			data_writer   : FrameWriter       | dict,
			process       : dict,
			id_           : int  | str  = uuid.uuid4().int,
			featuremerger : dict | None = None,
			*args, **kwargs
	):
		super().__init__(id_=id_, dataset=dataset, name=name)
		self.subset           = subset
		self.process          = process

		self.data_cfg         = data
		self.data_loader_cfg  = data_loader
		self.detector_cfg     = detector
		self.reidentifier_cfg = reidentifier
		self.featuremerge_cfg = featuremerger
		self.tracker_cfg      = tracker
		self.data_writer      = data_writer

		self.detector         = None
		self.reidentifier     = None
		self.tracker          = None
		self.mtmc_drawer      = MTMCDrawer()
		self.video_len        = None
		self.video_fps        = None

		self.data_writer_dets_debug   = None
		self.data_writer_dets_crop    = None
		self.data_writer_tracks_debug = None
		self.data_writer_tracks_zone  = None

		self.init_dirs()
		self.init_class_labels(class_labels=class_labels)
		self.init_data_writer(data_writer=data_writer)

		if self.process["function_dets"]:
			self.init_detector(detector=detector)

		if self.process["function_dets_crop_feat"]:
			self.init_reidentifier(reidentifier=reidentifier)

		if self.process["function_tracking"] or \
				self.process["function_tracks_postprocess"] or \
				self.process["save_tracks_img"] or \
				self.process["save_tracks_zone"]:
			self.init_tracker(tracker=tracker)

		if self.process["function_matching_zone"] or \
				self.process["function_matching_scene"] or \
				self.process["function_write_result"]:
			self.init_reidentifier_matching(matching=matching)

		self.mos        = []
		self.start_time = None
		self.stop_time  = None
		self.path_video = self.data_loader_cfg["data"]

	# MARK: Configure

	def init_dirs(self):
		"""Initialize dirs."""
		self.root_dir      = os.path.join(data_dir     , self.dataset)
		self.configs_dir   = os.path.join(self.root_dir, "configs")
		self.video_dir     = os.path.join(self.root_dir, self.subset)
		self.outputs_dir   = os.path.join(self.root_dir, "temps")
		self.timestamp_dir = os.path.join(self.root_dir, "timestamp")

	def init_class_labels(self, class_labels: ClassLabels | dict):
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

	def init_detector(self, detector: BaseDetector | dict):
		"""Initialize detector.

		Args:
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
		"""
		logger.info(f"Initiate Detector.")
		if isinstance(detector, BaseDetector):
			self.detector = detector
		elif isinstance(detector, dict):
			detector["class_labels"] = self.class_labels
			self.detector = DETECTORS.build(**detector)
		else:
			raise ValueError(f"Cannot initialize detector with {detector}.")

	def init_detector_filter(self):
		"""Initialize the detection filter."""
		pass

	def init_reidentifier(self, reidentifier: BaseReIndentifier | dict):
		"""Initialize reidentifier.

		Args:
			reidentifier (BaseReIndentifier, dict):
				ReIndentifier object or a reidentifier's config dictionary.
		"""
		logger.info(f"Initiate Re-Indentifier.")
		if isinstance(reidentifier, BaseReIndentifier):
			self.reidentifier = reidentifier
		elif isinstance(reidentifier, dict):
			self.reidentifier = REIDENTIFIER.build(**reidentifier)
		else:
			raise ValueError(f"Cannot initialize detector with {reidentifier}.")

	def init_tracker(self, tracker: BaseDetector | dict):
		"""Initialize tracker.

		Args:
			tracker (BaseTracker, dict):
				Tracker object or a tracker's config dictionary.
		"""
		logger.info(f"Initiate Tracker.")
		if isinstance(tracker, BaseTracker):
			self.tracker = tracker
		elif isinstance(tracker, dict):
			self.tracker = TRACKERS.build(**tracker)
		else:
			raise ValueError(f"Cannot initialize tracker with {tracker}.")

	def init_reidentifier_matching(self, matching: dict):
		"""Initialize reidentifier.

		Args:
			reidentifier (BaseReIndentifier, dict):
				ReIndentifier object or a reidentifier's config dictionary.
		"""
		logger.info(f"Initiate Re-Indentifier.")
		if isinstance(matching, dict):
			self.matching = REIDENTIFIER.build(**matching)
		else:
			raise ValueError(f"Cannot initialize detector with {matching}.")

	def init_data_loader(self, data_loader: FrameLoader | dict):
		"""Initialize data loader.

		Args:
			data_loader (FrameLoader, dict):
				Data loader object or a data loader's config dictionary.
		"""
		if isinstance(data_loader, FrameLoader):
			self.data_loader = data_loader
		elif isinstance(data_loader, dict):
			data = data_loader.get("data", "")
			if data_loader.get("ignore_region", "") is not None:
				data_loader["ignore_region"] = os.path.join(self.video_dir, data_loader.get("ignore_region", ""))

			if is_video_file(data):
				data_loader["data"] = data
			elif is_basename(data):
				data_loader["data"] = os.path.join(self.video_dir, f"{data}")
			elif is_stem(data):
				data_loader["data"] = os.path.join(
					self.video_dir, f"{data}.mp4"
				)
			else:
				data_loader["data"] = os.path.join(
					self.video_dir, data
				)
			self.data_loader = FrameLoader(**data_loader)
		else:
			raise ValueError(f"Cannot initialize data loader with {data_loader}.")

	def check_and_create_folder(self, attr, data_writer: FrameWriter | dict):
		"""CHeck and create the folder to store the result

		Args:
			attr (str):
				the type of function/saving/creating
			data_writer (dict):
				configuration of camera
		Returns:
			None
		"""
		path = os.path.join(self.outputs_dir, f"{data_writer[attr]}", self.name)
		if not os.path.isdir(path):
			os.makedirs(path)
		data_writer[attr] = path

	def init_data_writer(self, data_writer: FrameWriter | dict):
		"""Initialize data writer.

		Args:
			data_writer (FrameWriter, dict):
				Data writer object or a data writer's config dictionary.
		"""
		# NOTE: always save image for processing
		data_writer["save_image"] = True
		data_writer["save_video"] = False
		self.check_and_create_folder("images", data_writer=data_writer)
		data_writer["dst"] = data_writer["images"]
		self.data_writer_images_with_roi = FrameWriter(**data_writer)

		# NOTE: always save image for processing
		data_writer["save_image"] = True
		data_writer["save_video"] = False
		data_writer["dst_crop"] = f'{data_writer["dst_crop"]}/{self.detector_cfg["folder_out"]}'
		self.check_and_create_folder("dst_crop", data_writer=data_writer)
		data_writer["dst"] = data_writer["dst_crop"]
		self.data_writer_dets_crop = FrameWriter(**data_writer)

		# NOTE: always save images for debug
		data_writer["save_image"] = False
		data_writer["save_video"] = True
		data_writer["dst_debug"] = os.path.join(
			self.outputs_dir,
			data_writer["dst_debug"],
			self.detector_cfg["folder_out"],
			f"{self.name}_dets"
		)
		# self.data_writer_dets_debug("dst_debug", data_writer=data_writer)
		data_writer["dst"] = data_writer["dst_debug"]
		if self.process["save_dets_img"]:
			self.data_writer_dets_debug = FrameWriter(**data_writer)

		# NOTE: save detections txt
		data_writer["dst_label"] = f'{data_writer["dst_label"]}/{self.detector_cfg["folder_out"]}'
		self.check_and_create_folder("dst_label", data_writer=data_writer)

		# NOTE: save detections txt
		data_writer["dst_crop_pkl"] = f'{data_writer["dst_crop_pkl"]}/{self.detector_cfg["folder_out"]}'
		self.check_and_create_folder("dst_crop_pkl", data_writer=data_writer)

		# NOTE: save detections feature
		data_writer["dst_dets_feat"] = f'{data_writer["dst_dets_feat"]}/{self.reidentifier_cfg["folder_out"]}'
		self.check_and_create_folder("dst_dets_feat", data_writer=data_writer)

		# NOTE: save merge feature of all reidentifier
		data_writer["dst_dets_feat_merge"] = f'{data_writer["dst_dets_feat_merge"]}'
		self.check_and_create_folder("dst_dets_feat_merge", data_writer=data_writer)

		# NOTE: save tracking txt result
		data_writer["dst_mots"] = f'{data_writer["dst_mots"]}/{self.tracker_cfg["folder_out"]}'
		self.check_and_create_folder("dst_mots", data_writer=data_writer)

		# NOTE: save tracking txt result after post process
		data_writer["dst_mots_res"] = f'{data_writer["dst_mots"]}/res'
		if not os.path.isdir(data_writer["dst_mots_res"]):
			os.makedirs(data_writer["dst_mots_res"])

		# NOTE: save video for detection filter debug
		data_writer["save_image"] = False
		data_writer["save_video"] = True
		data_writer["dst_debug_filter"] = f"{data_writer['dst_debug']}_filter"
		data_writer["dst"] = data_writer["dst_debug_filter"]
		if self.process["save_dets_img_filter"]:
			self.data_writer_dets_filter_debug = FrameWriter(**data_writer)

		# NOTE: save video for debug
		data_writer["save_image"] = False
		data_writer["save_video"] = True
		video_path = os.path.join(
			self.outputs_dir,
			data_writer["dst_mots_debug"],
			self.tracker_cfg["folder_out"],
			f"{self.name}_mots_feat_raw"
		)
		# self.check_and_create_folder("dst_mots_debug", data_writer=data_writer)
		data_writer["dst"] = video_path
		if self.process["save_tracks_img"]:
			self.data_writer_tracks_debug = FrameWriter(**data_writer)

		# NOTE: save video tracking result with zone
		data_writer["save_image"] = False
		data_writer["save_video"] = True
		video_path = os.path.join(
			self.outputs_dir,
			data_writer["dst_mots_debug"],
			self.tracker_cfg["folder_out"],
			f"{self.name}_mots_feat_zone"
			# f"{self.name}_mots_feat_break"  # DEBUG:
		)
		# self.check_and_create_folder("dst_mots_debug", data_writer=data_writer)
		data_writer["dst"] = video_path
		if self.process["save_tracks_zone"]:
			self.data_writer_tracks_zone = FrameWriter(**data_writer)

		# NOTE: save tracking feature
		data_writer["dst_mots_feat"] = f'{data_writer["dst_mots_feat"]}/{self.tracker_cfg["folder_out"]}'
		self.check_and_create_folder("dst_mots_feat", data_writer=data_writer)

		self.data_writer = data_writer

		# NOTE: get length of video and fps
		self.init_data_loader(data_loader=self.data_loader_cfg)
		self.video_len = self.data_loader.num_frames
		self.video_fps = self.data_loader.video_fps

	# MARK: Run

	def run_images_extraction(self):
		"""Extract and save frames from the video as images."""
		# NOTE: Load dataset
		self.data_loader_cfg["batch_size"] = 1
		self.init_data_loader(data_loader=self.data_loader_cfg)
		pbar = tqdm(total=len(self.data_loader), desc=f"Image extraction: {self.name}")
		index_image = -1
		for images, indexes, _, _ in self.data_loader:
			if len(indexes) == 0:
				break

			image = images[0]
			index_image += 1
			name_index_image = f"{index_image:06d}"
			self.data_writer_images_with_roi.write_frame(image, name_index_image)
			pbar.update(len(indexes))  # Update pbar
		pbar.close()

	def run_detection(self):
		"""Run the object detection model on the extracted frames."""
		# NOTE: Load dataset
		self.data_loader_cfg["batch_size"] = self.detector_cfg["batch_size"]
		self.init_data_loader(data_loader=self.data_loader_cfg)

		# NOTE: run detection
		pbar = tqdm(total=len(self.data_loader), desc=f"Detection: {self.name}")
		with torch.no_grad():
			height_img, width_img = None, None
			index_image           = -1
			out_dict              = dict()

			for images, indexes, _, _ in self.data_loader:
				# NOTE: if finish loading
				if len(indexes) == 0:
					break

				# NOTE: get size of image
				if height_img is None:
					height_img, width_img, _ = images[0].shape

				# NOTE: Detect batch of instances
				batch_instances = self.detector.detect(
					indexes=indexes, images=images
				)

				# NOTE: Write the detection result
				for index_b, batch in enumerate(batch_instances):
					image_draw = images[index_b].copy()
					index_image       += 1
					name_index_image  = f"{index_image:06d}"

					if self.process["save_dets_txt"]:
						with open(f'{self.data_writer["dst_label"]}/{name_index_image}.txt', 'w') as f_write:
							pass

					for index_in, instance in enumerate(batch):
						name_index_in = f"{index_in:08d}"
						bbox_xyxy     = instance.bbox

						# NOTE: avoid out of image bound
						# if int(bbox_xyxy[0]) < 0 or \
						# 		int(bbox_xyxy[1]) < 0 or \
						# 		int(bbox_xyxy[2]) > image_draw.shape[1] - 1 or \
						# 		int(bbox_xyxy[3]) > image_draw.shape[0] - 1:
						# 	print(bbox_xyxy)
						# 	continue

						# NOTE: small than 1000, removed, base on rules
						# if abs((bbox_xyxy[3] - bbox_xyxy[1]) * (bbox_xyxy[2] - bbox_xyxy[0])) < 1000:
						# 	continue

						crop_image    = images[index_b][bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]

						# NOTE: write crop object image
						if self.process["save_dets_crop"]:
							dets_crop_name = f"{name_index_image}_{name_index_in}"
							self.data_writer_dets_crop.write_frame(crop_image, dets_crop_name)

							if self.process["save_dets_pkl"]:
								out_dict[dets_crop_name] = {
									'bbox'   : (bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]),
									'frame'  : name_index_image,
									'id'     : name_index_in,
									'imgname': f"{dets_crop_name}.png",
									'class'  : instance.class_label["train_id"],
									'conf'   : instance.confidence
								}

						# NOTE: write txt
						if self.process["save_dets_txt"]:
							bbox_cxcywh_norm = bbox_xyxy_to_cxcywh_norm(bbox_xyxy, height_img, width_img)
							with open(f'{self.data_writer["dst_label"]}/{name_index_image}.txt', 'a') as f_write:
								f_write.write(f'{instance.class_label["train_id"]} '
											  f'{instance.confidence} '
											  f'{bbox_cxcywh_norm[0]} '
											  f'{bbox_cxcywh_norm[1]} '
											  f'{bbox_cxcywh_norm[2]} '
											  f'{bbox_cxcywh_norm[3]}\n')

						if self.process["save_dets_img"]:
							instance.draw(image_draw, bbox=True, score=True)

					# DEBUG: show detection result
					# cv2.imshow("result", images[index_b])
					# cv2.waitKey(1)

					# NOTE: write result image
					if self.process["save_dets_img"]:
						self.data_writer_dets_debug.write_frame(image_draw, name_index_image)

				# NOTE: get feature of all crop images

				pbar.update(len(indexes))  # Update pbar

			if self.process["save_dets_pkl"]:
				pickle.dump(
					out_dict,
					open(f"{os.path.join(self.data_writer['dst_crop_pkl'], self.name)}_dets_crop.pkl", 'wb')
				)

		pbar.close()

	def run_feature_detection(self):
		"""
		Extract feature of each cropped object image using the re-identifier.
		Loads the cropped bounding boxes, extracts their features, and 
		saves them into a pickle file.
		"""
		dets_pkl_file = f"{os.path.join(self.data_writer['dst_crop_pkl'], self.name)}_dets_crop.pkl"
		dets_dict_feat= pickle.load(open(dets_pkl_file, 'rb'))

		# NOTE: Load dataset
		self.data_loader_cfg["batch_size"] = self.reidentifier_cfg["batch_size"]
		self.data_loader_cfg["data"]       = self.data_writer['dst_crop']
		self.data_loader_cfg["ignore_region"] = None
		self.init_data_loader(data_loader=self.data_loader_cfg)

		# NOTE: run reidentifier
		pbar = tqdm(total=len(self.data_loader), desc=f"Feature extraction: {self.name}")
		for images, indexes, ab_paths, rel_paths in self.data_loader:

			# NOTE: extract feature
			batch_feat = self.reidentifier.extract(ab_paths)

			# NOTE: save to dictionary
			for image_path, feat in batch_feat.items():
				name_crop_image = os.path.splitext(os.path.basename(image_path))[0]
				if name_crop_image not in dets_dict_feat:
					print(f"Crop image is not in dets_dict_feat of {self.name}")
				dets_dict_feat[name_crop_image]['feat'] = feat

			pbar.update(len(indexes))

		pickle.dump(
			dets_dict_feat,
			open(f"{os.path.join(self.data_writer['dst_dets_feat'], self.name)}_dets_feat.pkl", 'wb')
		)
		pbar.close()

	def run_feature_merge(self):
		"""
		Merge features extracted from multiple re-identifier models.
		Averages the normalized features from all models for each detection 
		and saves the merged features to a pickle file.
		"""
		pbar = tqdm(total=len(self.featuremerge_cfg["camera"]))
		for cam in self.featuremerge_cfg["camera"]:
			pbar.set_description(f"Feature merge: {cam}")
			feat_dic_list = []

			# NOTE: load all feature of camera
			# DEBUG:
			# print(len(self.featuremerge_cfg["reidentifier"]))
			for feat_model in self.featuremerge_cfg["reidentifier"]:

				feat_pkl_file = os.path.join(
					self.outputs_dir,
					self.featuremerge_cfg["data_writer"]["dst_dets_feat"],
					feat_model,
					f"{cam}",
					f"{cam}_dets_feat.pkl"
				)
				feat_mode_dict = pickle.load(open(feat_pkl_file, 'rb'))
				feat_dic_list.append(feat_mode_dict)

			# NOTE: merge all feature
			merged_dict = feat_dic_list[0].copy()
			for crop_detection in merged_dict:
				patch_feature_list = []
				for feat_mode_dict in feat_dic_list:
					patch_feature_list.append(feat_mode_dict[crop_detection]['feat'])

				patch_feature_array = np.array(patch_feature_list)
				patch_feature_array = preprocessing.normalize(
					patch_feature_array,
					norm='l2',
					axis=1
				)

				patch_feature_mean = np.mean(patch_feature_array, axis=0)
				merged_dict[crop_detection]['feat'] = patch_feature_mean

			# NOTE: save the merge result
			merge_feat_pkl_folder = os.path.join(
				self.outputs_dir,
				self.featuremerge_cfg["data_writer"]["dst_dets_feat_merge"],
				f"{cam}"
			)
			if not os.path.isdir(merge_feat_pkl_folder):
				os.makedirs(merge_feat_pkl_folder)
			merged_pkl_file = os.path.join(merge_feat_pkl_folder, f'{cam}_dets_feat.pkl')
			pickle.dump(merged_dict, open(merged_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
			pbar.update(1)

		pbar.close()

	def run_detection_filter(self):
		"""
		Filter the detected bounding boxes based on multiple criteria:
		threshold, area, size, position, and overlap.
		"""
		merge_feat_pkl_folder = os.path.join(
			self.outputs_dir,
			self.featuremerge_cfg["data_writer"]["dst_dets_feat_merge"],
			f"{self.name}"
		)
		pkl_path = os.path.join(merge_feat_pkl_folder, f'{self.name}_dets_feat.pkl')
		filter_box = FilterBoxes(camera_name=self.name, pickle_path=pkl_path)
		filter_box.to_string_length_pickle()
		filter_box.filter_thres_bbox()
		filter_box.to_string_length_pickle()
		filter_box.filter_area_bbox()
		filter_box.to_string_length_pickle()
		filter_box.filter_size_bbox()
		filter_box.to_string_length_pickle()
		filter_box.filter_position_bbox()
		filter_box.to_string_length_pickle()
		filter_box.filter_bbox_overlap_bbox()
		filter_box.to_string_length_pickle()
		filter_box.save_pickle()

	def run_tracking(self):
		"""
		Run the multi-object tracking (MOT) algorithm.
		Updates tracklets based on bounding boxes and merged features.
		"""
		# NOTE: Load dataset
		self.data_loader_cfg["batch_size"] = self.tracker_cfg["batch_size"]
		self.data_loader_cfg["ignore_region"] = None
		self.init_data_loader(data_loader=self.data_loader_cfg)

		# NOTE: load merge feature
		if self.process["use_dets_feat_filter"]:
			print("Use Filter dets feat")
			dets_pkl_file = f"{os.path.join(self.data_writer['dst_dets_feat_merge'], self.name)}_dets_feat_filter.pkl"
		else:
			print("Use Original dets feat")
			dets_pkl_file  = f"{os.path.join(self.data_writer['dst_dets_feat_merge'], self.name)}_dets_feat.pkl"
		dets_dict_feat = pickle.load(open(dets_pkl_file, 'rb'))

		# NOTE: get information from merge feature
		bbox_dic, feat_dic = self.tracker.get_merge_feature(dets_dict_feat, self.name)

		# NOTE: Define default
		height_img, width_img, _ = self.data_cfg["shape"]
		feature_dim     = None
		min_box_area    = self.tracker_cfg["min_box_area"]
		min_confidence  = self.tracker_cfg["min_confidence"]
		nms_max_overlap = self.tracker_cfg["nms_max_overlap"]

		# NOTE: Tracking
		results = []
		for frame_idx in tqdm(range(len(self.data_loader)), desc=f"Tracking {self.name}: "):
			if frame_idx not in bbox_dic:
				continue
			detections = bbox_dic[frame_idx]
			feats      = feat_dic[frame_idx]

			# Run non-maxima suppression.
			boxes    = np.array([d[:4] for d in detections], dtype=float)
			scores   = np.array([d[4] for d in detections], dtype=float)
			nms_keep = nms(torch.from_numpy(boxes),
						   torch.from_numpy(scores),
						   iou_threshold=nms_max_overlap).numpy()
			detections = np.array([detections[i] for i in nms_keep if scores[i] >= min_confidence], dtype=float)
			feats      = np.array([feats[i] for i in nms_keep if scores[i] >= min_confidence], dtype=float)

			# Update tracker.
			online_targets = self.tracker.update(detections, feats, frame_idx)
			# Store results.
			for t in online_targets:
				tlwh     = t.det_tlwh
				track_id = t.track_id
				score    = t.score
				cls      = t.cls
				# feature = t.smooth_feat
				feature  = t.features[-1]
				# vertical = tlwh[2] / tlwh[3] > 1.6
				feature  = t.smooth_feat
				# Filter - track xong xoa het nhung bounding box nao nho hon min_box_area
				# if tlwh[2] * tlwh[3] > min_box_area:
				results.append([
					frame_idx, track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], score, cls, feature
				])

		# NOTE: write pickle result
		self.tracker.write_pickle_results(
			f"{os.path.join(self.data_writer['dst_mots_feat'],self.name)}_mot_feat_raw.pkl",
			results,
			self.name
		)

		# NOTE: print txt result of tracking
		self.tracker.write_txt_results(
			f"{os.path.join(self.data_writer['dst_mots'], self.name)}_mot.txt",
			results,
			self.name
		)

		# NOTE: post process for tracking, use this for fast run
		if self.process["function_tracks_postprocess"]:
			self.run_tracking_post_process(results)
			# After run post process, turn it off, because in self.run() we have one more call
			self.process["function_tracks_postprocess"] = False

	def run_tracking_post_process(self, results = None):
		"""Post process for tracking

		Args:
			results (list):
				Result from task tracking
				If results is None, we can load from mots_raw.pkl
		Returns:

		"""
		if results is None:  # check to load results
			results = self.tracker.load_pickle_result(
				f"{os.path.join(self.data_writer['dst_mots_feat'], self.name)}_mot_feat_raw.pkl"
			)

		# NOTE: post process of tracking
		results = self.tracker.post_process(results, self.name)

		# NOTE: write pickle result
		self.tracker.write_pickle_results(
			f"{os.path.join(self.data_writer['dst_mots_feat'], self.name)}_mot_feat.pkl",
			results,
			self.name
		)

		# NOTE: print txt result of tracking
		self.tracker.write_txt_results(
			f"{os.path.join(self.data_writer['dst_mots_res'], self.name)}_mot.txt",
			results,
			self.name
		)

	def run(self):
		"""Main run loop."""
		self.run_routine_start()

		# NOTE: run image extraction
		if self.process["images_extraction"]:
			self.run_images_extraction()

		# NOTE: run detection
		if self.process["function_dets"]:
			self.run_detection()
			self.detector.clear_model_memory()
			self.detector = None

		# NOTE: run feature detection
		if self.process["function_dets_crop_feat"]:
			self.run_feature_detection()
			self.reidentifier.clear_model_memory()
			self.reidentifier = None

		# NOTE: run merge all feature
		if self.process["function_dets_crop_feat_merge"]:
			self.run_feature_merge()

		# NOTE: run filter all detection result
		if self.process["detection_filter"]:
			self.run_detection_filter()

		# NOTE: draw detection after filter result
		if self.process["save_dets_img_filter"]:
			path_feat_merge = f"{os.path.join(self.data_writer['dst_dets_feat_merge'], self.name)}_dets_feat_filter.pkl"
			if os.path.isfile(path_feat_merge):
				self.draw_dets_filter(
					path_feat_merge,
					self.name
				)

		# NOTE: run motion tracking
		if self.process["function_tracking"]:
			self.run_tracking()

		# NOTE: run post process motion tracking
		# it will be False if the we turn on both function_tracking and post_process_tracking
		# because the post process has run in self.run_tracking()
		if self.process["function_tracks_postprocess"]:
			self.run_tracking_post_process()

		# NOTE: draw tracking result
		if self.process["save_tracks_img"]:
			path_feat_raw = f"{os.path.join(self.data_writer['dst_mots_feat'], self.name)}_mot_feat_raw.pkl"
			if os.path.isfile(path_feat_raw):
				self.draw_mots(
					path_feat_raw,
					self.name
				)

		# NOTE: run matching zone
		if self.process["function_matching_zone"]:
			self.matching.match_zone(
				self.name,
				f"{os.path.join(self.data_writer['dst_mots_feat'])}",
				self.process["use_mots_feat_raw"],
				self.featuremerge_cfg,
			)

		# NOTE: draw tracking result
		if self.process["save_tracks_zone"]:
			path_feat_raw = f"{os.path.join(self.data_writer['dst_mots_feat'], self.name)}_mot_feat_zone.pkl"
			# DEBUG: get result from original
			# path_feat_raw = f"{os.path.join(self.data_writer['dst_mots_feat'], self.name)}_mot_feat_break.pkl"
			if os.path.isfile(path_feat_raw):
				self.draw_mots_zone(
					path_feat_raw,
					self.name
				)

		# NOTE: run matching scene
		if self.process["function_matching_scene"]:
			self.matching.sub_cluster(
				self.featuremerge_cfg,
				f"{os.path.join(self.data_writer['dst_mots_feat'])}",
				self.outputs_dir
			)

		# NOTE: run writing result
		if self.process["function_write_result"]:
			self.matching.writing_result(
				self.featuremerge_cfg,
				self.video_dir,
				f"{os.path.join(self.data_writer['dst_mots_feat'])}",
				self.outputs_dir
			)

		# NOTE: draw tracking result
		if self.process["save_mtmc_result"]:
			self.draw_mtmc()

		self.run_routine_end()

	def run_routine_start(self):
		"""Perform operations when run routine starts. We start the timer."""
		self.start_time = timer()

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
		pass

	# MARK: Visualize

	def draw_dets_filter(
			self,
			pkl_path: str,
			camera_name: str
	):
		"""
		Draw the filtered detections on the frames for debugging.

		Args:
			pkl_path (str): Path to the pickle file containing filtered detections.
			camera_name (str): Name of the camera.
		"""
		# NOTE: create load data
		self.data_loader_cfg["batch_size"] = 1
		self.data_loader_cfg["data"] = self.path_video
		self.data_loader_cfg["ignore_region"] = None
		self.init_data_loader(data_loader=self.data_loader_cfg)

		# NOTE: load data
		# merge_feat_pkl_folder = os.path.join(
		# 	self.outputs_dir,
		# 	self.featuremerge_cfg["data_writer"]["dst_dets_feat_merge"],
		# 	f"{self.name}"
		# )
		# pkl_path   = os.path.join(merge_feat_pkl_folder, f'{self.name}_dets_feat_filter.pkl')
		filter_box = FilterBoxes(camera_name=self.name, pickle_path=pkl_path)
		results    = filter_box.get_list_data()

		self.mtmc_drawer.draw_dets(
			camera_name,
			self.data_loader,
			results,
			self.data_writer_dets_filter_debug
		)

	def draw_mots(
			self,
			path_pickle: str,
			camera_name: str
	):
		"""
		Draw the multi-object tracking (MOT) results on the frames.

		Args:
			path_pickle (str): Path to the MOT pickle file.
			camera_name (str): Name of the camera.
		"""
		# NOTE: create load data
		self.data_loader_cfg["batch_size"]    = 1
		self.data_loader_cfg["data"]          = self.path_video
		self.data_loader_cfg["ignore_region"] = None
		self.init_data_loader(data_loader=self.data_loader_cfg)

		results = self.tracker.load_pickle_result(path_pickle)
		results = sorted(results, key=itemgetter(0))  # Sort by first column, "frame" column

		self.mtmc_drawer.draw_mots(
			camera_name,
			self.data_loader,
			results,
			self.data_writer_tracks_debug
		)

	def draw_mots_zone(
			self,
			path_pickle: str,
			camera_name: str
	):
		"""
		Draw the multi-object tracking (MOT) results with zone information on the frames.

		Args:
			path_pickle (str): Path to the MOT pickle file.
			camera_name (str): Name of the camera.
		"""
		# NOTE: create load data
		self.data_loader_cfg["batch_size"] = 1
		self.data_loader_cfg["data"] = self.path_video
		self.data_loader_cfg["ignore_region"] = None
		self.init_data_loader(data_loader=self.data_loader_cfg)

		results = self.tracker.load_pickle_result(path_pickle)
		results = sorted(results, key=itemgetter(0))  # Sort by first column, "frame" column

		self.mtmc_drawer.draw_mots_zone(
			camera_name,
			self.data_loader,
			results,
			self.data_writer_tracks_zone
		)

	def draw_mtmc(self):
		"""
		Draw the final Multi-Target Multi-Camera (MTMC) tracking results, 
		linking tracklets across different cameras.
		"""
		self.mtmc_drawer.draw_mtmc(
			self.video_dir,
			self.data_writer,
			os.path.join(self.outputs_dir, "mtmc_result.txt"),
			self.outputs_dir
		)

	def draw(self, drawing: np.ndarray, elapsed_time: float) -> np.ndarray:
		"""Visualize the results on the drawing.

		Args:
			drawing (np.ndarray):
				Drawing canvas.
			elapsed_time (float):
				Elapsed time per iteration.
		"""
		if not os.path.exists(os.path.join(self.outputs_dir, "mtmc_result.txt")):
			print(f"Result of mtmc_result.txt is not exist")
		else:
			self.draw_mtmc()
		return None


# MARK: Utils
