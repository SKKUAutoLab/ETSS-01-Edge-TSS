#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SORT tracker.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from typing import Union

import cv2
import numpy as np
from src.builder import TRACKERS
from src.objects import Instance
from src.type import ListOrTuple3T
from tqdm import tqdm

__all__ = [
	"FairMOT"
]

from src.utils import console

from .fairmot.fm_tracker.multitracker import JDETracker
from .fairmot.post_processing.post_association import associate
from .fairmot.post_processing.track_nms import track_nms

np.random.seed(0)


# MARK: - FairMOT

@TRACKERS.register(name="fairmot_track_mtmc")
class FairMOT(JDETracker):
	"""FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking."""

	# MARK: Magic Functions

	def __init__(self, name: str = "fairmot_track_mtmc", *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.name = name

	# MARK: Update

	# MARK: Method
	def get_merge_feature(
			self,
			dets_dict_feat,
			camera_name
	):
		"""Get all feature from merge

		Args:
			dets_dict_feat (dict):
				The current merge feature of camera
			camera_name (str):
				Name of camera we are processing
		Returns:
			bbox_dic (dict)
				store the bounding box result and confident score
			feat_dic (dict)
				store the feature of each bounding box
		"""
		bbox_dic = {}
		feat_dic = {}
		for key, value in tqdm(dets_dict_feat.items(), desc=f"Collect merge feature {camera_name}: "):
			frame_index = int(re.sub("[^0-9]", "", value['frame']))
			det_bbox    = np.array(value['bbox']).astype('float32')
			det_feat    = value['feat']
			cls         = int(value['class']) if 'class' in feat_dic else 2
			cls         = np.array((cls, ))
			score       = value['conf']
			score       = np.array((score, ))
			det_bbox    = np.concatenate((det_bbox, score, cls)).astype('float32')
			if frame_index not in bbox_dic:
				bbox_dic[frame_index] = [det_bbox]
				feat_dic[frame_index] = [det_feat]
			else:
				bbox_dic[frame_index].append(det_bbox)
				feat_dic[frame_index].append(det_feat)
		return bbox_dic, feat_dic

	def post_process(
			self,
			results,
			camera_name
	):
		results = np.array(results)
		loaded_trk_ids = np.unique(results[:, 1])

		mark_interpolation = False
		# post associate

		# DEBUG: dang test cai write MOTs after post processs
		results = associate(results, 0.1, camera_name)
		results = associate(results, 0.1, camera_name)

		# remove len 1 track and interpolate, help on reducing FNs.
		# results = interpolate_traj(results, drop_len=1)

		# track nms can help reduce FPs.
		results = track_nms(results, 0.65)

		trk_ids = np.unique(results[:, 1])
		console.log('after all PP, merging ', len(loaded_trk_ids) - len(trk_ids), ' tracks')

		return results

	# MARK: input/output

	def load_pickle_result(
			self,
			path_pickle: str,
	):
		mot_feat_dic = pickle.load(open(path_pickle, 'rb'))
		results = []
		for image_name in list(mot_feat_dic.keys()):
			feat_dic = mot_feat_dic[image_name]
			# sub("[^0-9]", "", "!1a2;b3c?") to remove letter in string
			frame    = int(re.sub("[^0-9]", "", feat_dic['frame']))
			track_id = int(feat_dic['id'])
			bbox     = np.array(feat_dic['bbox']).astype('float32')
			score    = float(feat_dic['score']) if 'score' in feat_dic else None
			cls      = int(feat_dic['class']) if 'class' in feat_dic else 2  # 2 is class id of car in COCO dataset
			feat     = feat_dic['feat']
			zone     = feat_dic.get('zone', None)

			results.append([frame, track_id,
							bbox[0], bbox[1],
							bbox[2] - bbox[0], bbox[3] - bbox[1],
							score,
							cls,
							zone,
							feat])
		return results

	def write_pickle_results(
			self,
			feat_pkl_file : str,
			results,
			camera_name   : str
	):
		mot_feat_dic = {}
		for row in tqdm(results, desc=f"Write tracking pickle {camera_name}: "):
			[frame_idx, track_id, x, y, w, h] = row[:6]  # pylint: disable=invalid-name
			score      = row[6]
			feat       = row[-1]
			image_name = f'{camera_name}_{track_id:08d}_{frame_idx:04d}.png'
			bbox       = (x, y, x + w, y + h)
			cls        = row[7]
			frame      = f'{int(frame_idx):06d}'
			mot_feat_dic[image_name] = {
				'bbox'   : bbox,
				'frame'  : frame,
				'id'     : int(track_id),
				'imgname': image_name,
				'class'  : cls,
				'score'  : score,
				'feat'   : feat
			}
		pickle.dump(mot_feat_dic, open(feat_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)

	def write_txt_results(
			self,
			path_write  : str,
			results,
			camera_name : str
	):
		with open(path_write, 'w') as f:
			for row in tqdm(results, desc=f"Write tracking txt {camera_name}: "):
				# frame_idx, track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], score, (Optional: feature)
				f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1\n' % (
					row[0], row[1], row[2], row[3], row[4], row[5], row[6]))

	# MARK: visualize


# MARK: - Utils
