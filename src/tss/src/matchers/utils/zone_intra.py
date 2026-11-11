import os
import sys
from os.path import join as opj
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN

# Minimum fraction of the peak bbox dimension a truncated bbox must reach to be kept.
# Derived empirically: bboxes smaller than ~67 % of the local peak are likely partial cuts.
BBOX_B = 10/15


class ZoneTracker():
	"""Merges duplicate track IDs that occupy nearly the same spatial position.

	Used inside `comb_mot` to reconcile vehicles that were assigned separate IDs
	but are moving through the same zone pixel-region simultaneously.
	`id2id` stores the canonical ID rewrite so later frames stay consistent.
	"""
	def __init__(self):
		# Recent detections in the current zone region.
		self.zone_list = list()
		# Mapping from reassigned ID -> canonical ID.
		self.id2id = dict()

	def update(self,det):
		have_up = False
		det_time = int(det['frame'])

		# Mirror x so that proximity is measured in display-left→right order,
		# matching the physical road layout where lanes run left-to-right.
		det_x = 1280 - (det['bbox'][2]+det['bbox'][0])/2
		det_y = (det['bbox'][3]+det['bbox'][1])/2
		max_sim = 0
		max_zi  = 0
		for zi,zl in enumerate(self.zone_list):
			zl_time = int(zl['frame'])
			if det_time==zl_time:
				continue
			if det_time>zl_time+4:
				self.zone_list[zi] = -1
				continue
			if zl['id']==det['id']:
				self.zone_list[zi] = det
				have_up = True
				break
			if det['id'] in self.id2id:
				if self.id2id[det['id']]==zl['id']:
					if det_time==zl_time:
						have_up = True
						break
					else:
						det['id'] = zl['id']
						self.id2id[det['id']] = zl['id']
						self.zone_list[zi] = det
						have_up = True
						break
			zl_x = 1280 - (zl['bbox'][2] + zl['bbox'][0]) / 2
			zl_y = (zl['bbox'][3] + zl['bbox'][1]) / 2
			if det_x<zl_x+10 and det_y<zl_y+10:
				continue
			if abs(det_x-zl_x)<15 and abs(det_y-zl_y)<15:
				# print("det:{}  zl:{}".format([det['id'],det_x,det_y],[zl['id'],zl_x,zl_y]))
				if det['id'] in self.id2id:
					continue
				det['id'] = zl['id']
				self.id2id[det['id']]=zl['id']
				self.zone_list[zi] = det
				have_up = True
				break
		self.zone_list = [zl for zl in self.zone_list if zl!=-1]
		if not have_up:
			self.zone_list.append(det)
		return det


class ZoneIntra():

	def __init__(self, zone_folder):
		# 1:white, 2:red, 3:green, 4:blue
		# white red non-high speed
		# blue green high speed
		zones = {}
		# NOTE: load the image of zone
		zone_path = f"{zone_folder}/zone"
		for img_name in os.listdir(zone_path):
			# camnum = int(img_name.split('.')[0][-3:])
			# camnum = int(img_name.split('.')[0][1:])
			camnum = os.path.splitext(img_name)[0]
			if "zones" in camnum:
				zone_img = cv2.imread(opj(zone_path, img_name))
				zones[camnum.replace("_zones", "")] = zone_img

		self.zones       = zones
		self.current_cam = 0

	def set_cam(self, cam):
		"""Set current camera's id

		Args:
			cam (int):
				id of current camera
		"""
		self.current_cam = cam

	def get_zone(self, bbox):
		"""From the bounding box, we get the zone by color of the center point

		Args:
			bbox (list):
				xyxy (VOC) format of bounding box
		Returns:
			zone_num (int):
				the zone number where bounding box belong
		"""
		# get the center point of bounding box
		cx = int((bbox[0] + bbox[2]) / 2)
		cy = int((bbox[1] + bbox[3]) / 2)

		# get color of center point
		pix = self.zones[self.current_cam][cy, cx, :]

		# NOTE: color comparison
		# Color thresholding is intentionally simple and robust to small noise.
		zone_num = 0
		if pix[0] > 50 and pix[1] > 50 and pix[2] > 50:  # white
			zone_num = 1
		if pix[0] < 50 and pix[1] < 50 and pix[2] > 50:  # red
			zone_num = 2
		if pix[0] < 50 and pix[1] > 50 and pix[2] < 50:  # green
			zone_num = 3
		if pix[0] > 50 and pix[1] < 50 and pix[2] < 50:  # blue
			zone_num = 4
		return zone_num

	def is_ignore(self, zone_list, frame_list, camera_id, merge_config):
		"""Check is trajectory is good to keep, based on the zone of vehicle each time and the index of frame while vehicle appear
		We can define which tracklet we keep, which one we set for sub__ (e.g., 1) instead of remove (e.g., 2)

		Args:
			zone_list (list):
				Zone of vehicle each time
			frame_list (list):
				Index of frame while vehicle appear
			camera_id (int):
				camera's id
			merge_config (dict)
		Returns:
			(int):
				0: keep
				1: set for sub__
				2: remove
		"""
		# 0 middle of intersection
		# 1 white 2 red 3 green 4 blue
		zone_start , zone_end  = zone_list[0] , zone_list[-1]
		frame_start, frame_end = frame_list[0], frame_list[-1]
		# If always in a region from begin to end, exclude
		# It may happen by wrong detection (e.g., traffic sign,...)
		if zone_start == zone_end:
			# SUGAR: if only stay on 1 zone in camera [merge_config["camera"][0]]
			if camera_id in [merge_config["camera"][0]]:
				if zone_end == 4 and len(set(zone_list)) == 1:
					return 1

			# NOTE: dung yen trong zone 1 va 2 vi zone_start == zone_end in [1, 2]
			if zone_end in [1, 2]:
				return 2
			if zone_start != 0 and 0 in zone_list:
				return 0
			# if frame_end - frame_start > 1500:
			# 	return 2
			if frame_start < 2:
				if camera_id in [45]:
					if zone_end in [3, 4]:
						return 1
					else:
						return 2
			# if frame_end > 1999:
			# 	if camera_id in [merge_config["camera"][0]]:
			# 		if zone_end not in [3]:
			# 			return 2
			# 		else:
			# 			return 0
			# if frame_start < 2 or frame_end > 1999:
			# 	if zone_end in [3, 4]:
			# 		return 0
			if zone_end in [3, 4]:
				return 1
			return 2
		else:  # If the region changes
			# If entering and exiting from branch, exclude
			if camera_id in merge_config["camera"]:
				if zone_start == 1 and zone_end == 2:
					return 2
				if zone_start == 2 and zone_end == 1:
					return 2
			# At camera 41, the vehicle is not in and out of camera 42
			if camera_id in [merge_config["camera"][0]]:
				if (zone_start in [1, 2]) and zone_end == 4:
					return 2
				if zone_start == 4 and (zone_end in [1, 2]):
					return 2
				# SUGAR: neu xe no moi chay vo intersection chu chua lam gi
				# if zone_start == 4 and (zone_end in [0]) and frame_end > 1990:
				# 	return 2
			# On camera 46, the vehicle does not go in and out of camera 45
			if camera_id in [merge_config["camera"][-1]]:
				if (zone_start in [1, 2]) and zone_end == 3:
					return 2
				if zone_start == 3 and (zone_end in [1, 2]):
					return 2
			return 0

	def filter_mot(self, mot_list, cid, merge_config):
		"""Filter trajectory, which tracklet do we keep? which one we remove?

		Args:

			mot_list (dict):
				Store trajectory
			cid (int):
				camera's id
			merge_config (dict):
				configuration for merging trajectory
					- 'filter_sub': if set, we will set some tracklet for sub__ (e.g., 1) instead of remove (e.g., 2)
		Returns:
			new_mot_list(dict):
				The new list of trajectory after filtering
		"""
		new_mot_list = dict()
		sub_mot_list = dict()
		for tracklet in mot_list:
			# if cid == 45 and tracklet==207:
			#     print(tracklet)
			tracklet_dict = mot_list[tracklet]
			frame_list    = list(tracklet_dict.keys())
			frame_list.sort()
			zone_list     = []

			for f in frame_list:
				zone_list.append(tracklet_dict[f]['zone'])

			# DEBUG:
			# if tracklet == 68 and cid == 45:
			# 	zs, ze = zone_list[0], zone_list[-1]
			# 	fs, fe = frame_list[0], frame_list[-1]
			# 	print(zs, ze, fs, fe)
			# 	print(self.is_ignore(zone_list, frame_list, cid))
			# 	print(zone_list)
			# 	sys.exit()

			if self.is_ignore(zone_list, frame_list, cid, merge_config) == 0:
				new_mot_list[tracklet] = tracklet_dict
			# if self.is_ignore(zone_list, frame_list, cid) == 1:
			# 	sub_mot_list[tracklet] = tracklet_dict
		return new_mot_list

	def filter_bbox(self, mot_list, cid):
		"""Remove unnecessary bounding boxes

		Args:
			mot_list (dict):
				List of tracklets
			cid (int):
				camera's id
		Returns:
			new_mot_list (dict):
				List of tracklets after filter
		"""
		new_mot_list = dict()
		yh = self.zones[cid].shape[0]
		for tracklet in mot_list:
			# if tracklet==1812:
			#     print(tracklet)
			tracklet_dict = mot_list[tracklet]
			frame_list    = list(tracklet_dict.keys())
			frame_list.sort()

			bbox_list = []
			for f in frame_list:
				bbox_list.append(tracklet_dict[f]['bbox'])

			bbox_x = [b[0] for b in bbox_list]
			bbox_y = [b[1] for b in bbox_list]
			bbox_w = [b[2] - b[0] for b in bbox_list]
			bbox_h = [b[3] - b[1] for b in bbox_list]

			# DEBUG:
			# zone_list = []
			# if tracklet == 68 and cid == 45:
			# 	for f in frame_list:
			# 		zone_list.append(tracklet_dict[f]['zone'])
			# 	print(zone_list)
			# 	sys.exit()
			# 	pass
			# else:
			# 	continue

			new_frame_list = list()
			if 0 in bbox_x or 0 in bbox_y:
				b0 = [i for i, f in enumerate(frame_list) if bbox_x[i] < 5 or bbox_y[i] + bbox_h[i] > yh - 5]
				# SUGAR:
				if len(b0) == 0:
					continue

				if len(b0) == len(frame_list):
					# if cid in [41, 42, 44, 45, 46]:
					# SUGAR:
					# DEBUG:
					if cid in [41, 42, 44, 46]:
						continue
					max_w = max(bbox_w)
					max_h = max(bbox_h)
					for i, f in enumerate(frame_list):
						if bbox_w[i] > max_w * BBOX_B and bbox_h[i] > max_h * BBOX_B:
							new_frame_list.append(f)
				else:
					l_i, r_i = 0, len(frame_list) - 1
					if b0[0] == 0:
						for i in range(len(b0) - 1):
							if b0[i] + 1 == b0[i + 1]:
								l_i = b0[i + 1]
							else:
								break
					if b0[-1] == len(frame_list) - 1:
						for i in range(len(b0) - 1):
							i = len(b0) - 1 - i
							if b0[i] - 1 == b0[i - 1]:
								r_i = b0[i - 1]
							else:
								break

					max_lw, max_lh = bbox_w[l_i], bbox_h[l_i]
					max_rw, max_rh = bbox_w[r_i], bbox_h[r_i]
					for i, f in enumerate(frame_list):
						if i < l_i:
							if bbox_w[i] > max_lw * BBOX_B and bbox_h[i] > max_lh * BBOX_B:
								new_frame_list.append(f)
						elif i > r_i:
							if bbox_w[i] > max_rw * BBOX_B and bbox_h[i] > max_rh * BBOX_B:
								new_frame_list.append(f)
						else:
							new_frame_list.append(f)
				new_tracklet_dict = dict()
				for f in new_frame_list:
					new_tracklet_dict[f] = tracklet_dict[f]
				new_mot_list[tracklet] = new_tracklet_dict
			else:
				new_mot_list[tracklet] = tracklet_dict

		return new_mot_list

	def break_mot(self, mot_list, cid):
		"""Break one tracklets into many tracklets base on:
			- if the tracklet has the compress zone list which is not unique (e.g., [1, 0, 2, 0, 3, 0, 4])
			- if the tracklet has two adjacent frame index apart more than 100 (e.g., id_pre=50 -> id_post=190)
		Args:
			mot_list (dict):
				Store tracklets
			cid (int):
				camera id
		Returns:
			new_mot_list (dict):
				The result after breaking
		"""
		# if not cid in [41,44,45,46]:
		#     return mot_list
		new_mot_list = dict()
		new_num_tracklets = max(mot_list) + 1
		for tracklet in mot_list:
			tracklet_dict = mot_list[tracklet]
			frame_list    = list(tracklet_dict.keys())
			frame_list.sort()
			zone_list     = []
			back_tracklet = False
			new_zone_f    = 0
			pre_frame     = frame_list[0]
			time_break    = False

			# DEBUG:
			# if tracklet == 118 and cid == 44:
			# 	for f in frame_list:
			# 		zone_list.append(tracklet_dict[f]['zone'])
			# 	print(zone_list)
			# 	sys.exit()

			# NOTE: compress the zone of vehicle
			# (e.g., [1, 1, 0, 0, 0, 2] -> zone_list==[1, 0, 2])
			for f in frame_list:
				# check if 2 adjacent frames index which have vehicle are more than 100.
				if f - pre_frame > 100:
					if cid in [44, 45]:
						time_break = True
						break
				# if cid not in [41, 44, 45, 46]:
				# DEBUG:
				# SUGAR:
				if cid not in [41, 45, 46]:
					break
				pre_frame = f
				new_zone  = tracklet_dict[f]['zone']
				# check if new_zone is the current zone and zone_list is not empty
				if len(zone_list) > 0 and zone_list[-1] == new_zone:
					continue
				if new_zone_f > 1:
					# check if the zone_list mixes with many same zone,
					# (e.g.,  zone_list==[3, 0, 4, 2, 0] new_zone==4)
					if len(zone_list) > 1 and new_zone in zone_list:
						back_tracklet = True
					zone_list.append(new_zone)  # From zone to zone (e.g., zone_list==[3, 0, 4])
					new_zone_f = 0
				else:
					new_zone_f += 1

			# NOTE: check if there is wrong find zone in tracklets
			# if the zone_list mixes with many same zones (e.g.,  [3, 0, 4, 2, 0, 4])
			if back_tracklet:
				new_tracklet_dict = dict()
				pre_bbox = -1
				pre_arrow = 0
				have_break = False
				for f in frame_list:
					now_bbox = tracklet_dict[f]['bbox']
					if pre_bbox == -1:
						pre_bbox = now_bbox
					now_arrow = now_bbox[0] - pre_bbox[0]
					if pre_arrow * now_arrow < 0 and len(new_tracklet_dict) > 15 and not have_break:
						new_mot_list[tracklet] = new_tracklet_dict
						new_tracklet_dict = dict()
						have_break = True
					if have_break:
						tracklet_dict[f]['id'] = new_num_tracklets
					new_tracklet_dict[f] = tracklet_dict[f]
					pre_bbox, pre_arrow = now_bbox, now_arrow
				if have_break:
					new_mot_list[new_num_tracklets] = new_tracklet_dict
					new_num_tracklets += 1
				else:
					new_mot_list[tracklet] = new_tracklet_dict
			# if 2 adjacent frames index which have vehicle are more than 100 (e.g., 57 -> 190)
			elif time_break:
				new_tracklet_dict = dict()
				have_break = False
				pre_frame = frame_list[0]
				for f in frame_list:
					if f - pre_frame > 100:
						new_mot_list[tracklet] = new_tracklet_dict
						new_tracklet_dict = dict()
						have_break = True
					new_tracklet_dict[f] = tracklet_dict[f]
					pre_frame = f
				if have_break:
					new_mot_list[new_num_tracklets] = new_tracklet_dict
					new_num_tracklets += 1
				else:
					new_mot_list[tracklet] = new_tracklet_dict
			else:
				new_mot_list[tracklet] = tracklet_dict

		print("Break from old:{} to new:{}".format(len(mot_list), len(new_mot_list)))

		return new_mot_list

	def comb_mot(self,mot_list,cid):
		time_mot = dict()
		for tid in mot_list:
			tracklet = mot_list[tid]
			for tf in tracklet:
				if tf in time_mot:
					time_mot[tf].append(tracklet[tf])
				else:
					time_mot[tf]=[tracklet[tf]]
		time_list = list(time_mot)
		time_list.sort()
		zone_tack = ZoneTracker()
		new_mot_list = dict()
		for t in time_list:
			det_list = time_mot[t]
			for di,dt in enumerate(det_list):
				if dt['zone'] in [4]:
					dt = zone_tack.update(dt)
				else:
					if dt['id'] in zone_tack.id2id:
						dt['id']=zone_tack.id2id[dt['id']]
				time_mot[t][di] = dt
				if dt['id'] in new_mot_list:
					new_mot_list[dt['id']][int(dt['frame'])]=dt
				else:
					new_mot_list[dt['id']] = {int(dt['frame']): dt}
		print("old:{} new:{}".format(len(mot_list),len(new_mot_list)))
		return new_mot_list
