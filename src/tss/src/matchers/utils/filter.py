import os
import sys
from os.path import join as opj
import scipy.io as scio
import cv2
import numpy as np
import pickle
from scipy import spatial
import copy
import multiprocessing
from math import *
from sklearn import preprocessing
from tqdm import tqdm

CAM_DIST = [[0   , 40  , 55, 100, 120 , 145],
			[40  , 0   , 15, 60 , 80  , 105],
			[55  , 15  , 0 , 40 , 65  , 90],
			[100 , 60  , 40, 0  , 20  , 45],
			[120 , 80  , 65, 20 , 0   , 25],
			[145 , 105 , 90, 45 , 25  , 0]]


def intracam_ignore(st_mask, cid_tids):
	count = len(cid_tids)
	for i in range(count):
		for j in range(count):
			if cid_tids[i][0] == cid_tids[j][0]:
				st_mask[i, j] = 0.
	return st_mask


def get_dire(zone_list, cid):
	zs, ze = zone_list[0], zone_list[-1]
	return (zs, ze)


def st_filter(st_mask, cid_tids, cid_tid_dict):
	count = len(cid_tids)
	for i in range(count):
		i_tracklet = cid_tid_dict[cid_tids[i]]
		i_cid = i_tracklet['cam']
		i_dire = get_dire(i_tracklet['zone_list'], i_cid)
		i_iot = i_tracklet['io_time']
		for j in range(count):
			j_tracklet = cid_tid_dict[cid_tids[j]]
			j_cid = j_tracklet['cam']
			j_dire = get_dire(j_tracklet['zone_list'], j_cid)
			j_iot = j_tracklet['io_time']

			match_dire = True

			# DEBUG:
			# FIXME:
			# NOTE: 41 or 1 is the camera_id, c041 or C01, tuy thuoc vao dataset
			# cam_dist = CAM_DIST[i_cid - 41][j_cid - 41]
			cam_dist = CAM_DIST[i_cid - 1][j_cid - 1]

			# If the time overlaps
			if i_iot[0] - cam_dist < j_iot[0] and j_iot[0] < i_iot[1] + cam_dist:
				match_dire = False
			if i_iot[0] - cam_dist < j_iot[1] and j_iot[1] < i_iot[1] + cam_dist:
				match_dire = False

			# No longer matches after going out
			if i_dire[1] in [1, 2]:  # i out
				if i_iot[0] < j_iot[1] + cam_dist:
					match_dire = False

			if i_dire[1] in [1, 2]:
				if i_dire[0] in [3] and i_cid > j_cid:
					match_dire = False
				if i_dire[0] in [4] and i_cid < j_cid:
					match_dire = False

			if i_cid in [41] and i_dire[1] in [4]:
				if i_iot[0] < j_iot[1] + cam_dist:
					match_dire = False
				if i_iot[1] > 199:
					match_dire = False
			if i_cid in [46] and i_dire[1] in [3]:
				if i_iot[0] < j_iot[1] + cam_dist:
					match_dire = False

			# enter to match
			if i_dire[0] in [1, 2]:  # i in
				if i_iot[1] > j_iot[0] - cam_dist:
					match_dire = False

			if i_dire[0] in [1, 2]:
				if i_dire[1] in [3] and i_cid > j_cid:
					match_dire = False
				if i_dire[1] in [4] and i_cid < j_cid:
					match_dire = False

			is_ignore = False
			if ((i_dire[0] == i_dire[1] and i_dire[0] in [3, 4]) or (
					j_dire[0] == j_dire[1] and j_dire[0] in [3, 4])):
				is_ignore = True

			if not is_ignore:
				# direction conflict
				if (i_dire[0] in [3] and j_dire[0] in [4]) or (i_dire[1] in [3] and j_dire[1] in [4]):
					match_dire = False

				# Filter before going to the next scene
				if i_dire[1] in [3] and i_cid < j_cid:
					if i_iot[1] > j_iot[1] - cam_dist:
						match_dire = False
				if i_dire[1] in [4] and i_cid > j_cid:
					if i_iot[1] > j_iot[1] - cam_dist:
						match_dire = False

				if i_dire[0] in [3] and i_cid < j_cid:
					if i_iot[0] < j_iot[0] + cam_dist:
						match_dire = False
				if i_dire[0] in [4] and i_cid > j_cid:
					if i_iot[0] < j_iot[0] + cam_dist:
						match_dire = False
				## ↑ 3-30

				## 4-1
				if i_dire[0] in [3] and i_cid > j_cid:
					if i_iot[1] > j_iot[0] - cam_dist:
						match_dire = False
				if i_dire[0] in [4] and i_cid < j_cid:
					if i_iot[1] > j_iot[0] - cam_dist:
						match_dire = False
				##

				# Filter after going to the next scene
				## 4-7
				if i_dire[1] in [3] and i_cid > j_cid:
					if i_iot[0] < j_iot[1] + cam_dist:
						match_dire = False
				if i_dire[1] in [4] and i_cid < j_cid:
					if i_iot[0] < j_iot[1] + cam_dist:
						match_dire = False
				##
			else:
				if i_iot[1] > 199:
					if i_dire[0] in [3] and i_cid < j_cid:
						if i_iot[0] < j_iot[0] + cam_dist:
							match_dire = False
					if i_dire[0] in [4] and i_cid > j_cid:
						if i_iot[0] < j_iot[0] + cam_dist:
							match_dire = False
					if i_dire[0] in [3] and i_cid > j_cid:
						match_dire = False
					if i_dire[0] in [4] and i_cid < j_cid:
						match_dire = False
				if i_iot[0] < 1:
					if i_dire[1] in [3] and i_cid > j_cid:
						match_dire = False
					if i_dire[1] in [4] and i_cid < j_cid:
						match_dire = False

			if not match_dire:
				st_mask[i, j] = 0.0
				st_mask[j, i] = 0.0
	return st_mask

# SUGAR: new added
def order_filter(st_mask, cid_tids, cid_tid_dict, camera_order):
	"""Keep an adjacent-camera pair only if one tracklet leaves toward the other camera,
	the other enters from it, and the leaving one comes first in time.
	Zone 3 faces the next camera in camera_order, zone 4 faces the previous one."""
	count = len(cid_tids)
	for i in range(count):
		ti = cid_tid_dict[cid_tids[i]]
		for j in range(count):
			tj   = cid_tid_dict[cid_tids[j]]
			step = camera_order.index(tj['cam']) - camera_order.index(ti['cam'])
			if abs(step) != 1:
				continue
			out_zone, in_zone = (3, 4) if step == 1 else (4, 3)
			# i -> j
			if ti['zone_list'][-1] == out_zone and tj['zone_list'][0] == in_zone and \
					ti['io_time'][0] < tj['io_time'][0] and ti['io_time'][1] < tj['io_time'][1]:
				continue
			# j -> i
			if tj['zone_list'][-1] == in_zone and ti['zone_list'][0] == out_zone and \
					tj['io_time'][0] < ti['io_time'][0] and tj['io_time'][1] < ti['io_time'][1]:
				continue
			st_mask[i, j] = 0.
	return st_mask


def subcam_list(cid_tid_dict, cameraId_trackIds, camera_order):
	sub_3_4 = dict()
	sub_4_3 = dict()
	for cameraId_trackId in cameraId_trackIds:
		camera_id, track_id = cameraId_trackId

		idx = camera_order.index(camera_id)
		camera_id_prev = camera_order[idx - 1] if idx - 1 >= 0 else camera_order[-1]
		camera_id_next = camera_order[idx + 1] if idx + 1 < len(camera_order) else camera_order[0]

		tracklet = cid_tid_dict[cameraId_trackId]
		zs, ze   = get_dire(tracklet['zone_list'], camera_id)
		if zs in [3] and camera_id not in [camera_order[-1]]:  # 4 to 3
			if not camera_id_next in sub_4_3:
				sub_4_3[camera_id_next] = []
			sub_4_3[camera_id_next].append(cameraId_trackId)
		if ze in [4] and camera_id not in [camera_order[0]]:  # 4 to 3
			if not camera_id in sub_4_3:
				sub_4_3[camera_id] = []
			sub_4_3[camera_id].append(cameraId_trackId)
		if zs in [4] and camera_id not in [camera_order[0]]:  # 3 to 4
			if not camera_id_prev in sub_3_4:
				sub_3_4[camera_id_prev] = []
			sub_3_4[camera_id_prev].append(cameraId_trackId)
		if ze in [3] and camera_id not in [camera_order[-1]]:  # 3 to 4
			if not camera_id in sub_3_4:
				sub_3_4[camera_id] = []
			sub_3_4[camera_id].append(cameraId_trackId)
	sub_cid_tids = dict()
	for i in sub_3_4:
		idx      = camera_order.index(i)
		next_cam = camera_order[idx + 1] if idx + 1 < len(camera_order) else camera_order[0]
		sub_cid_tids[(i, next_cam)] = sub_3_4[i]
	for i in sub_4_3:
		idx      = camera_order.index(i)
		prev_cam = camera_order[idx - 1] if idx - 1 >= 0 else camera_order[-1]
		sub_cid_tids[(i, prev_cam)] = sub_4_3[i]
	return sub_cid_tids


def subcam_list2(cid_tid_dict, cid_tids, camera_order):
	sub_dict = dict()
	for cameraId_trackId in cid_tids:
		camera_id, track_id = cameraId_trackId
		idx = camera_order.index(camera_id)
		camera_id_prev = camera_order[idx - 1] if idx - 1 >= 0 else camera_order[-1]
		camera_id_next = camera_order[idx + 1] if idx + 1 < len(camera_order) else camera_order[0]

		if camera_id not in [camera_order[0]]:
			if not camera_id in sub_dict:
				sub_dict[camera_id] = []
			sub_dict[camera_id].append(cameraId_trackId)
		if camera_id not in [camera_order[-1]]:
			if not camera_id_next in sub_dict:
				sub_dict[camera_id_next] = []
			sub_dict[camera_id_next].append(cameraId_trackId)
	return sub_dict
