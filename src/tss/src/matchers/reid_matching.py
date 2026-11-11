#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class to store persistent objects
"""
from __future__ import annotations

import abc
import os
import pickle
import sys
from typing import Optional

import numpy as np
from loguru import logger
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from src.builder import REIDENTIFIER
from src.configuration import data_dir
from src.matchers.utils.filter import *
from src.matchers.utils.visual_rr import visual_rerank
from src.matchers.utils.zone_intra import ZoneIntra
from tqdm import tqdm

__all__ = [
	"ReidMatching"
]

from yacs.config import CfgNode as CN

cfg = CN()

cfg.CHALLENGE_DATA_DIR = ''
cfg.DET_SOURCE_DIR = ''
cfg.REID_MODEL = ''
cfg.REID_BACKBONE = ''
cfg.REID_SIZE_TEST = [256, 256]

cfg.DET_IMG_DIR = ''
cfg.DATA_DIR = ''
cfg.ROI_DIR = ''
cfg.CID_BIAS_DIR = ''

cfg.USE_RERANK = False
cfg.USE_FF = False
cfg.SCORE_THR = 0.5

cfg.MCMT_OUTPUT_TXT = ''


# MARK: - ReidMatching

@REIDENTIFIER.register(name="re_identifier_matching")
class ReidMatching(metaclass=abc.ABCMeta):
	"""Extract reid feature."""

	# MARK: Magic Functions

	def __init__(
			self,
			timestamp_dir: str,
			scene_name   : str,
			dataset      : str,
			zone         : str,
			name         : Optional[str] = "re_identifier_matching",
			*args, **kwargs
	):
		"""
		Initialize the ReidMatching component.

		Args:
			timestamp_dir (str): Directory containing timestamp bias files.
			scene_name (str): Name of the current scene (e.g., S01, S02).
			dataset (str): Name of the dataset being used.
			zone (str): Directory/file specific to spatial zone boundaries.
			name (Optional[str]): The identifier name of the component.
		"""
		self.timestamp_dir    = timestamp_dir
		self.scene_name       = scene_name
		self.dataset          = dataset
		self.zone_folder      = zone

		self.camera_timestamp = self.load_timestamp()
		self.zones            = ZoneIntra(os.path.join(data_dir, dataset))

	# MARK: Configure

	def load_timestamp(self):
		"""Load the timestamp of the scene

		Returns:
			camera_timestamp (dict):
				store the timestamp of each camera, the bias we need to count in beginning
		"""
		camera_timestamp = dict()
		with open(os.path.join(data_dir, self.dataset, self.timestamp_dir, f'{self.scene_name}.txt')) as f:
			lines = f.readlines()

			for line in lines:
				line        = line.strip().split(' ')
				# camera_name = int(line[0][2:])
				camera_name = line[0]
				bias        = float(line[1])

				if camera_name not in camera_timestamp:
					camera_timestamp[camera_name] = bias
		return camera_timestamp

	# MARK: Fusion - Finding crossed zone

	def match_zone(
			self,
			camera_name,
			mot_feat_folder,
			use_feat_raw,
			merge_config,
	):
		"""
		Orchestrates the process of finding and assigning spatial zones to a camera's MOT 
		tracklets and then aggregating/merging those trajectories.

		Args:
			camera_name (str): The name of the camera we are working on (e.g., "c041").
			mot_feat_folder (str): The path storing the extracted MOT feature pickle files.
			use_feat_raw (bool): Whether to use the raw motion tracking results or 
			                     the post-processed ones.
		"""
		mot_list = self.find_zone(camera_name, mot_feat_folder, use_feat_raw, merge_config)
		self.merge_trajectory_zone(camera_name, mot_feat_folder, mot_list)

	def find_zone(
			self,
			camera_name,
			mot_feat_folder,
			use_feat_raw,
			merge_config,
	):
		"""Find the crossed zone for each vehicle with its own trajectory

		Args:
			camera_name (str):
				The name of camera we are working one.
				e.g., "c041"
			mot_feat_folder (str):
				The path to store the merged feature of vehicle
				which store in pickle file
			use_feat_raw (bool):
				There are two option in use of
		Returns:
			mot_list (dict)
				Motion tracking result which include one more attribute
				Including the zone of the vehicle at the moment on the trajectory
		"""
		# camera_name_int = int(camera_name[-3:])
		# self.zones.set_cam(camera_name_int)
		self.zones.set_cam(camera_name)
		# NOTE: choose to use raw motion tracking or post process of motion tracking
		if use_feat_raw:
			mot_feat_pkl = os.path.join(mot_feat_folder, f'{camera_name}_mot_feat_raw.pkl')  # result of original
		else:
			mot_feat_pkl = os.path.join(mot_feat_folder, f'{camera_name}_mot_feat.pkl')  # result of post process

		mot_feat_fusion_pkl = os.path.join(mot_feat_folder, f'{camera_name}_mot_feat_zone.pkl')

		with open(mot_feat_pkl, 'rb') as f:
			lines = pickle.load(f)

		# NOTE: filter trajectory with zone
		mot_list = dict()
		for line in tqdm(lines, desc=f"Find zone of track {camera_name}: "):
			fid  = int(lines[line]['frame'])
			tid  = lines[line]['id']
			bbox = list(map(lambda x: int(float(x)), lines[line]['bbox']))
			if tid not in mot_list:
				mot_list[tid] = dict()
			out_dict           = lines[line]
			out_dict['zone']   = self.zones.get_zone(bbox)
			mot_list[tid][fid] = out_dict

		# DEBUG:
		# logger.info("HELLO WORLD")

		# NOTE: process the trajectory with zone
		print(f"Length trajectory before process the trajectory with zone - {len(mot_list)}")
		# mot_list = self.zones.break_mot(mot_list, camera_name)
		# print(f"Length trajectory after break MOT- {len(mot_list)}")
		mot_list = self.zones.filter_mot(mot_list, camera_name, merge_config)  # filter by zone
		print(f"Length trajectory after filter MOT- {len(mot_list)}")
		# mot_list = self.zones.filter_bbox(mot_list, camera_name)  # filter bbox
		# print(f"Length trajectory after filter bbox- {len(mot_list)}")

		# NOTE: save the trajectory with zone
		out_dict = dict()
		for tracklet in mot_list:
			tracklet = mot_list[tracklet]
			for f in tracklet:
				out_dict[tracklet[f]['imgname']] = tracklet[f]

		# NOTE: write the trajectory with zone
		pickle.dump(out_dict, open(mot_feat_fusion_pkl, 'wb'))

		return mot_list

	def merge_trajectory_zone(
			self,
			camera_name,
			mot_feat_folder,
			mot_list,
	):
		"""Get the result from the finding zone of vehicle
		Write it down into other format to help following process
			*_feat_zone -> *_feat_merge
		Args:
			camera_name (str):
				The name of camera we are working on
			mot_feat_folder (str):
				The folder to store the pickle file
			mot_list (dict):
				The motion tracking result with zone
		"""
		# NOTE: merge into one file
		# camera_id = int(camera_name[-3:])
		# camera_id = int(camera_name[1:])
		camera_id = camera_name
		cur_bias = self.camera_timestamp[camera_id]
		mot_feat_merge_pkl = os.path.join(mot_feat_folder, f'{camera_name}_mot_feat_merge.pkl')
		track_id_data = dict()

		for track_id in tqdm(mot_list, desc=f"Merge track with zone into one file {camera_name}: "):
			tracklet = mot_list[track_id]
			if len(tracklet) <= 1:
				continue

			frame_list = list(tracklet.keys())
			frame_list.sort()
			# if track_id==11 and camera_id==44:
			#     print(track_id)
			zone_list = [tracklet[f]['zone'] for f in frame_list]

			# NOTE: Only get feature with bounding box more than 2000.
			feature_list = [tracklet[f]['feat'] for f in frame_list if (tracklet[f]['bbox'][3] - tracklet[f]['bbox'][1]) * (
						tracklet[f]['bbox'][2] - tracklet[f]['bbox'][0]) > 2000]
			if len(feature_list) < 2:
				feature_list = [tracklet[f]['feat'] for f in frame_list]

			# NOTE: 10 mean video has 10fps
			video_fps = 10.
			io_time   = [cur_bias + frame_list[0] / video_fps, cur_bias + frame_list[-1] / video_fps]
			all_feat  = np.array([feat for feat in feature_list])
			mean_feat = np.mean(all_feat, axis=0)

			track_id_data[track_id] = {
				'cam'       : camera_id,
				'tid'       : track_id,
				'mean_feat' : mean_feat,
				'zone_list' : zone_list,
				'frame_list': frame_list,
				'tracklet'  : tracklet,
				'io_time'   : io_time
			}

		# NOTE: write the merge feature
		pickle.dump(track_id_data, open(mot_feat_merge_pkl, 'wb'), pickle.HIGHEST_PROTOCOL)

	# MARK: Sub-Cluster

	def sub_cluster(
			self,
			merge_config,
			mot_feat_folder,
			output_folder
	):
		"""
		Perform multi-target multi-camera (MTMC) clustering to associate 
		tracklets across different cameras into global identities.

		Args:
			merge_config (dict): Configuration mapping containing the target cameras and MTMC rules.
			mot_feat_folder (str): Directory where the merged MOT feature pickles are stored.
			output_folder (str): Directory where the final clustered MTMC results will be saved.
		"""
		camera_orders     = merge_config['camera']  # order of camera, thu tu camera trong config, thu tu nay se anh huong den ket qua clustering, nen can xac dinh dung
		cfg.CAMERA_ORDERS = camera_orders

		# NOTE: load merge pickle
		cameraId_trackId_dict = self.load_merge_trajectory_zone_pickle(merge_config, mot_feat_folder)
		cameraId_trackIds     = sorted([key for key in cameraId_trackId_dict.keys()])

		# NOTE: clustering
		clu = self.get_labels(cfg, cameraId_trackId_dict, cameraId_trackIds, score_thr=cfg.SCORE_THR)

		# NOTE: filtering the MOT after clustering
		print('all_clu:', len(clu))
		new_clu = list()
		for c_list in clu:
			# Filter, if there is only one camera id.
			if len(c_list) <= 1:
				continue
			cam_list = [cameraId_trackIds[c][0] for c in c_list]
			# Filter, if camera id appears more than once.
			if len(cam_list) != len(set(cam_list)):
				continue
			new_clu.append([cameraId_trackIds[c] for c in c_list])
		print('new_clu: ', len(new_clu))

		all_clu = new_clu
		cameraId_trackId_label = dict()
		for i, c_list in enumerate(all_clu):
			for c in c_list:
				cameraId_trackId_label[c] = i + 1
		pickle.dump({'cluster': cameraId_trackId_label}, open(os.path.join(output_folder, 'mtmc_result.pkl'), 'wb'))

	def load_merge_trajectory_zone_pickle(self,	merge_config, mot_feat_folder):
		"""
		Load and aggregate all merged trajectory pickle files across specified cameras.

		Args:
			merge_config (dict): MTMC configuration specifying the valid cameras.
			mot_feat_folder (str): Base directory containing camera subfolders with merge pickles.

		Returns:
			cameraid_trackid_dict (dict): A mapping from (camera_id, track_id) to tracklet data.
		"""
		cameraid_trackid_dict = dict()
		mot_feat_folder_temp = os.path.dirname(mot_feat_folder)
		for camera_name in tqdm(merge_config['camera'], desc="Load merge trajectory: "):
			# camera_id = int(camera_name[-3:])
			# camera_id = int(camera_name[1:])
			camera_id = camera_name
			pkl_path  = os.path.join(mot_feat_folder_temp, camera_name, f'{camera_name}_mot_feat_merge.pkl')

			with open(pkl_path, 'rb') as f:
				lines = pickle.load(f)

			for line in lines:
				tracklet = lines[line]
				track_id = tracklet['tid']
				if (camera_id, track_id) not in cameraid_trackid_dict:
					cameraid_trackid_dict[(camera_id, track_id)] = tracklet
		return cameraid_trackid_dict

	def get_sim_matrix(self, _cfg, cameraId_tid_dict, cameraId_tids):
		"""
		Compute the similarity matrix between tracklets utilizing visual re-ranking 
		and spatio-temporal (ST) constraints.

		Args:
			_cfg: Configuration object containing MTMC hyperparameters.
			cameraId_tid_dict (dict): Dictionary mapping tracklet IDs to tracklet data.
			cameraId_tids (list): Target (camera_id, track_id) tuples defining matrix axes.

		Returns:
			sim_matrix (np.ndarray): The computed similarity matrix with ST masking applied.
		"""
		count = len(cameraId_tids)
		# print('count: ', count)

		q_arr = np.array([cameraId_tid_dict[cameraId_tids[i]]['mean_feat'] for i in range(count)])
		g_arr = np.array([cameraId_tid_dict[cameraId_tids[i]]['mean_feat'] for i in range(count)])
		q_arr = self.normalize(q_arr, axis=1)
		g_arr = self.normalize(g_arr, axis=1)
		# sim_matrix = np.matmul(q_arr, g_arr.T)

		# st mask
		st_mask = np.ones((count, count), dtype=np.float32)
		st_mask = intracam_ignore(st_mask, cameraId_tids)

		# DEBUG: cho nay la filter khoan cach thoi gian, neu 2 tracklet co khoang cach thoi gian qua nho thi se khong the la cung 1 nguoi duoc, nen se set mask = 0, neu khoang cach thoi gian qua lon thi se khong anh huong den ket qua, nen se set mask = 1
		# FIXME: can chinh lai nhieu cho nay
		# FIXME: 41 or 1 is the camera_id, c041 or C01, tuy thuoc vao dataset
		# st_mask = st_filter(st_mask, cameraId_tids, cameraId_tid_dict)
		# SUGAR: new added
		st_mask = order_filter(st_mask, cameraId_tids, cameraId_tid_dict, _cfg.CAMERA_ORDERS)

		# visual rerank
		visual_sim_matrix = visual_rerank(q_arr, g_arr, cameraId_tids, _cfg)
		visual_sim_matrix = visual_sim_matrix.astype('float32')
		# print(visual_sim_matrix)
		# merge result
		np.set_printoptions(precision=3)
		sim_matrix = visual_sim_matrix * st_mask

		# sim_matrix[sim_matrix < 0] = 0
		np.fill_diagonal(sim_matrix, 0)
		return sim_matrix

	def normalize(self, nparray, axis=0):
		"""L2 normalizes a numpy array along the specified axis."""
		nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
		return nparray

	def get_match(self, cluster_labels):
		"""Group dataset indices together based on their assigned cluster labels."""
		cluster_dict = dict()
		cluster = list()
		for i, l in enumerate(cluster_labels):
			if l in list(cluster_dict.keys()):
				cluster_dict[l].append(i)
			else:
				cluster_dict[l] = [i]
		for idx in cluster_dict:
			cluster.append(cluster_dict[idx])
		return cluster

	def get_cameraid_trackid(self, cluster_labels, cameraId_tids):
		"""Map local cluster indices back to their global (camera_id, track_id) tuples."""
		cluster = list()
		for labels in cluster_labels:
			cameraid_trackid_list = list()
			for label in labels:
				cameraid_trackid_list.append(cameraId_tids[label])
			cluster.append(cameraid_trackid_list)
		return cluster

	def combin_cluster(self, sub_labels, cameraId_tids):
		"""
		Combine overlapping sub-clusters into larger global identities.
		Iterates through tracklets to merge sub-cluster sets sharing matching targets.

		Args:
			sub_labels (dict): Sub-clusters generated at an earlier clustering step.
			cameraId_tids (list): All target (camera_id, track_id) tuples.
		"""
		cluster = list()
		for sub_c_to_c in sub_labels:
			if len(cluster) < 1:
				cluster = sub_labels[sub_c_to_c]
				continue

			for c_ts in sub_labels[sub_c_to_c]:
				is_add = False
				for i_c, c_set in enumerate(cluster):
					if len(set(c_ts) & set(c_set)) > 0:
						new_list     = list(set(c_ts) | set(c_set))
						cluster[i_c] = new_list
						is_add       = True
						break
				if not is_add:
					cluster.append(c_ts)
		labels = list()
		num_tr = 0
		for c_ts in cluster:
			label_list = list()
			for c_t in c_ts:
				label_list.append(cameraId_tids.index(c_t))
				num_tr += 1
			label_list.sort()
			labels.append(label_list)
		print("new tracklets:{}".format(num_tr))
		return labels, cluster

	def combin_feature(self, cameraId_tid_dict, sub_cluster):
		"""
		Average the visual features of tracklets that are grouped within the same sub-cluster.
		Updates the source dictionary's mean feature in-place.
		"""
		for sub_ct in sub_cluster:
			if len(sub_ct) < 2:
				continue

			mean_feat = np.array([cameraId_tid_dict[i]['mean_feat'] for i in sub_ct])
			for i in sub_ct:
				cameraId_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)

		return cameraId_tid_dict

	def get_labels(self, _cfg, cameraId_tid_dict, cameraId_tids, score_thr):
		"""
		Execute a multi-stage Agglomerative Hierarchical Clustering pipeline 
		to generate cross-camera track labels.

		Args:
			_cfg: Configuration object containing thresholds.
			cameraId_tid_dict (dict): Global mapping of tracklet data.
			cameraId_tids (list): Targets to cluster.
			score_thr (float): Unused configuration score threshold placeholder.

		Returns:
			labels (list): A clustered list grouped into global identities.
		"""
		# NOTE: 1st cluster
		sub_cameraId_tids = subcam_list(cameraId_tid_dict, cameraId_tids, cfg.CAMERA_ORDERS)
		sub_labels = dict()
		dis_thrs   = [0.7, 0.5, 0.5, 0.5, 0.5,
					  0.7, 0.5, 0.5, 0.5, 0.5]
		for i, sub_c_to_c in enumerate(sub_cameraId_tids):
			sim_matrix = self.get_sim_matrix(_cfg, cameraId_tid_dict, sub_cameraId_tids[sub_c_to_c])
			cluster_labels = AgglomerativeClustering(n_clusters         = None,
													 distance_threshold = 1 - dis_thrs[i],
													 metric             = 'precomputed',
													 linkage            = 'complete').fit_predict(1 - sim_matrix)
			labels = self.get_match(cluster_labels)
			cluster_cameraId_tids = self.get_cameraid_trackid(labels, sub_cameraId_tids[sub_c_to_c])
			sub_labels[sub_c_to_c] = cluster_cameraId_tids

		print("old tracklets:{}".format(len(cameraId_tids)))
		labels, sub_cluster = self.combin_cluster(sub_labels, cameraId_tids)

		# NOTE: 2nd cluster
		cameraId_tid_dict_new = self.combin_feature(cameraId_tid_dict, sub_cluster)
		sub_cameraId_tids = subcam_list2(cameraId_tid_dict_new, cameraId_tids, cfg.CAMERA_ORDERS)
		sub_labels = dict()
		for i, sub_c_to_c in enumerate(sub_cameraId_tids):
			sim_matrix = self.get_sim_matrix(_cfg, cameraId_tid_dict_new, sub_cameraId_tids[sub_c_to_c])
			cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - 0.1,
													 metric='precomputed',
													 linkage='complete').fit_predict(1 - sim_matrix)
			labels = self.get_match(cluster_labels)
			cluster_cameraId_tids = self.get_cameraid_trackid(labels, sub_cameraId_tids[sub_c_to_c])
			sub_labels[sub_c_to_c] = cluster_cameraId_tids
		print("old tracklets:{}".format(len(cameraId_tids)))
		labels, sub_cluster = self.combin_cluster(sub_labels, cameraId_tids)

		# NOTE: 3rd cluster - luc dau cai nay khong can xai
		# cameraId_tid_dict_new = self.combin_feature(cameraId_tid_dict,sub_cluster)
		# sim_matrix = self.get_sim_matrix(_cfg,cameraId_tid_dict_new, cameraId_tids)
		# cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - 0.2, affinity='precomputed',
		#                                          linkage='complete').fit_predict(1 - sim_matrix)
		# labels = self.get_match(cluster_labels)
		return labels

	# MARK: Write Result

	def writing_result(
			self,
			merge_config,
			video_dir,
			mot_feat_folder,
			output_folder
	):
		"""
		Write the finalized MTMC tracking results to a standardized output text file.
		Adjusts tracklet bounding boxes referencing ROIs for standardized evaluation.

		Args:
			merge_config (dict): Target camera configuration mapping.
			video_dir (str): Base directory storing the original video ROIs.
			mot_feat_folder (str): Directory containing the intermediate track pickles.
			output_folder (str): Path to output the final parsed text results.
		"""
		map_tid     = pickle.load(open(os.path.join(output_folder, 'mtmc_result.pkl'), 'rb'))['cluster']
		output_txt  = os.path.join(output_folder, "mtmc_result.txt")
		self.show_res(map_tid)

		with open(output_txt, 'w') as f_w:
			mot_feat_folder_temp = os.path.dirname(mot_feat_folder)
			for camera_name in tqdm(merge_config['camera'], desc="Writing result: "):
				# cid = int(camera_name[-3:])
				# cid = int(camera_name[1:])
				camera_id = camera_name

				# DEBUG:
				roi = cv2.imread(os.path.join(video_dir, camera_name, f'{camera_name}_roi.jpg'), 0)
				# roi = None

				img_height, img_width = roi.shape
				img_rects = self.parse_pt(os.path.join(mot_feat_folder_temp, camera_name, f'{camera_name}_mot_feat_zone.pkl'))
				for fid in img_rects:
					tid_rects = img_rects[fid]
					fid = int(fid) + 1
					for tid_rect in tid_rects:
						track_id = tid_rect[0]
						rect = tid_rect[1:]
						cx = 0.5 * rect[0] + 0.5 * rect[2]
						cy = 0.5 * rect[1] + 0.5 * rect[3]
						# DEBUG: print sao lai nhan them thong so
						w = abs(rect[2] - rect[0])
						# w = min(w * 1.2, w + 40)
						h = abs(rect[3] - rect[1])
						# h = min(h * 1.2, h + 40)
						rect[2] -= rect[0]
						rect[3] -= rect[1]
						rect[0] = max(0, rect[0])
						rect[1] = max(0, rect[1])
						x1, y1 = max(0, cx - 0.5 * w), max(0, cy - 0.5 * h)
						x2, y2 = min(img_width, cx + 0.5 * w), min(img_height, cy + 0.5 * h)
						w, h = x2 - x1, y2 - y1

						new_rect = list(map(int, [x1, y1, w, h]))
						# new_rect = rect # use original rect
						rect = list(map(int, rect))
						if (camera_id, track_id) in map_tid:
							new_tid = map_tid[(camera_id, track_id)]
							f_w.write(str(camera_id) + ' ' + str(new_tid) + ' ' + str(fid) + ' ' + ' '.join(
								map(str, new_rect)) + ' -1 -1' '\n')

	def parse_pt(self, pt_file):
		"""
		Parse an intermediate pickle file containing raw bounding box locations,
		reorganizing them efficiently grouped by frame ID.
		"""
		with open(pt_file, 'rb') as f:
			lines = pickle.load(f)
		img_rects = dict()
		for line in lines:
			fid = int(lines[line]['frame'])
			tid = lines[line]['id']
			rect = list(map(lambda x: int(float(x)), lines[line]['bbox']))
			if fid not in img_rects:
				img_rects[fid] = list()
			rect.insert(0, tid)
			img_rects[fid].append(rect)
		return img_rects

	def show_res(self, map_tid):
		"""Log the final global ID groupings of the computed clusters directly to standard out."""
		show_dict = dict()
		for cid_tid in map_tid:
			iid = map_tid[cid_tid]
			if iid in show_dict:
				show_dict[iid].append(cid_tid)
			else:
				show_dict[iid] = [cid_tid]
		for i in show_dict:
			print('ID{}:{}'.format(i, show_dict[i]))
