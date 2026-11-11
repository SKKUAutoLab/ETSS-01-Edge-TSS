#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import colorsys
import os
import threading
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional

import cv2
from src.vision import FrameWriter
from tqdm import tqdm

__all__ = [
	"MTMCDrawer"
]


# MARK: - Drawer For MTMC Track

class MTMCDrawer:

	def __init__(self, *args, **kwargs):
		pass

	# MARK: Method

	def add_panel(self, image, label):
		h, w, _ = image.shape
		font_size = h // 400
		t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)[0]

		# NOTE: frame index
		cv2.rectangle(image, (0, 0), (t_size[0], t_size[1]), (0, 97, 153), -1)
		cv2.putText(image, label, (0, t_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_size, [255, 255, 255], 2)
		return image

	def draw_bbox(self, image, bbox, label, color):
		"""Draw bounding box"""
		# line_thickness = 1
		# tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
		tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
		tf = max(tl - 1, 1)  # font thickness
		x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
		t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, tl / 3, tf)[0]
		cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, tl)
		cv2.rectangle(image, (x_min, int(y_min - t_size[1] - 4)), (int(x_min + t_size[0] + 3), y_min), color, -1)
		cv2.putText(image, label, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, [255, 255, 255], tf)
		return image

	def draw_dets(
			self,
			camera_name,
			data_loader,
			results,
			data_writer_dets_filter_debug
	):
		# NOTE: run
		frame_index = -1
		result_index = 0
		for images, indexes, _, _ in tqdm(data_loader, desc=f"Draw detection {camera_name}: "):
			# reading from frame
			frame_index += 1
			image        = images[0]

			while True:
				if result_index >= len(results):
					break

				result = results[result_index]
				frame, bbox, score, cls = result[0], result[1], result[2], result[3]

				if frame != frame_index:
					break

				color = tuple(create_unique_color_uchar(int(16)))
				# label = f"vehicle - {score:.2f}"
				label = f"{score:.2f}"
				image = self.draw_bbox(image, bbox, label, color)
				result_index += 1

			# add frame index
			image = self.add_panel(image, str(frame_index))
			# writing the extracted images
			data_writer_dets_filter_debug.write_frame(image)


	def draw_mots(
			self,
			camera_name,
			data_loader,
			results,
			data_writer_tracks_debug
	):
		frame_index = -1
		result_index = 0

		# NOTE: draw tracking
		for images, indexes, _, _ in tqdm(data_loader, desc=f"Draw tracking {camera_name}: "):
			frame_index += 1
			image = images[0]

			while True:
				if result_index >= len(results):
					break

				result = results[result_index]
				frame, track_id, x_min, y_min, w, h, score, cls = \
					result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]

				if frame != frame_index:
					break

				x_max = x_min + w
				y_max = y_min + h

				color = tuple(create_unique_color_uchar(int(track_id)))
				# color = tuple(colors[train_ids.index(cls)])
				label = f"ID:{track_id}-cls:{cls}"
				image = self.draw_bbox(image, [x_min, y_min, x_max, y_max], label, color)
				result_index += 1

			# add frame index
			image = self.add_panel(image, str(frame_index))
			data_writer_tracks_debug.write_frame(image)

	def draw_mots_zone(
			self,
			camera_name,
			data_loader,
			results,
			data_writer_tracks_zone
	):
		frame_index = -1
		result_index = 0
		# NOTE: draw tracking
		for images, indexes, _, _ in tqdm(data_loader, desc=f"Draw tracking with zone {camera_name}: "):
			frame_index += 1
			image = images[0]

			while True:
				if result_index >= len(results):
					break

				result = results[result_index]
				frame, track_id, x_min, y_min, w, h, score, cls, zone = \
					result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8]

				if frame != frame_index:
					break

				x_max = x_min + w
				y_max = y_min + h

				color = tuple(create_unique_color_uchar(int(track_id)))
				# color = tuple(colors[train_ids.index(cls)])
				label = f"ID:{track_id}-cls:{cls}-z:{zone}"
				image = self.draw_bbox(image, [x_min, y_min, x_max, y_max], label, color)
				result_index += 1

			# add frame index
			image = self.add_panel(image, str(frame_index))
			data_writer_tracks_zone.write_frame(image)

	def load_mtmc_results_text(self, output_txt_path):
		"""load result from text file"""
		with open(output_txt_path, "r") as f:
			mct_tracks = f.readlines()
		cam_tracks = dict()
		for track_line in mct_tracks:
			c, cid, f, x, y, w, h, _, _ = tuple([int(float(sstr)) if sstr.isdigit() else sstr for sstr in track_line.split(' ')])
			if c in cam_tracks:
				if f in cam_tracks[c]:
					cam_tracks[c][f].append((cid, x, y, w, h))
				else:
					cam_tracks[c][f] = [(cid, x, y, w, h)]
			else:
				cam_tracks[c] = {f: [(cid, x, y, w, h)]}

		return cam_tracks

	def draw_mtmc_thread(self, cam_tracks, camera_id, video_cap, data_writer_mtmc, image_zone_path):
		height_video, width_video, _ = data_writer_mtmc.shape
		frame_count  = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
		pbar = tqdm(total=int(frame_count), desc=f"Drawing MTMC {camera_id}", leave=False, colour="#b89b28")
		# DEBUG:
		# img_roi = cv2.imread(image_zone_path, cv2.IMREAD_COLOR)
		img_roi = None
		roi_opacity = 0.35


		fr_id = 1
		state, img = video_cap.read()
		while state:
			# DEBUG: only for exhibition, draw specific id
			# is_write_frame = False

			if fr_id in cam_tracks[camera_id]:
				for pid, x, y, w, h in cam_tracks[camera_id][fr_id]:

					# DEBUG: only for exhibition, draw specific id
					# if pid != 119:  # 9, 63, 66, 68, 119, 157, 193
					# 	continue
					# is_write_frame = True

					label = f"{pid}"
					color = tuple(create_unique_color_uchar(int(pid)))
					img   = self.draw_bbox(img, [x, y, x + w, y + h], label, color)

			if img_roi is not None:
				img_roi_resized = cv2.resize(img_roi, (img.shape[1], img.shape[0]))
				img = cv2.addWeighted(img, 1.0 - roi_opacity, img_roi_resized, roi_opacity, 0.0)

			# add frame index
			img = self.add_panel(img, f"#{str(fr_id)}")
			img = cv2.resize(img, (width_video, height_video))



			# DEBUG: only for exhibition, write only has the target id
			# if is_write_frame:
			data_writer_mtmc.write_frame(img)

			state, img = video_cap.read()
			fr_id += 1

			pbar.update(1)
		pbar.close()

	def draw_mtmc(
			self,
			video_dir,
			data_writer,
			output_txt_path,
			output_dir
	):
		# NOTE: load results from text file
		cam_tracks = self.load_mtmc_results_text(output_txt_path)

		# NOTE: draw
		data_writer["save_image"] = False
		data_writer["save_video"] = True

		threads = []

		# NOTE: create thread
		for camera_id in tqdm(cam_tracks, desc="Get MTMC result"):
			# DEBUG:
			# FIXME: change it based on dataset
			video_path_in      = os.path.join(video_dir, f"{camera_id}", f"{camera_id}.mp4")
			# video_path_in      = os.path.join(video_dir, f"c0{camera_id}", "vdo.avi")
			# video_path_in      = os.path.join(video_dir, f"C0{camera_id}/out_rgb/video.mp4")
			video_path_out     = os.path.join(output_dir, "mtmc_debug", f"{camera_id}")
			# video_path_out     = os.path.join(output_dir, "mtmc_debug", f"c0{camera_id}")
			# video_path_out     = os.path.join(output_dir, "mtmc_debug", f"C0{camera_id}")
			image_zone_path    = os.path.join(os.path.dirname(video_dir), "zone", f"{camera_id}.png")
			data_writer["dst"] = video_path_out

			video_cap          = cv2.VideoCapture(video_path_in)
			width_video        = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height_video       = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

			data_writer["shape"] = [height_video, width_video, 3]
			data_writer_mtmc     = FrameWriter(**data_writer)
			threads.append(threading.Thread(target=self.draw_mtmc_thread, args=(cam_tracks, camera_id, video_cap, data_writer_mtmc, image_zone_path)))

		# NOTE: start thread
		for thread_tmp in threads:
			thread_tmp.start()

		# NOTE: wait to stop
		for thread_tmp in threads:
			thread_tmp.join()

# MARK: Utils

def create_unique_color_float(tag, hue_step=0.41):
	"""Create a unique RGB color code for a given track id (tag).

	The color code is generated in HSV color space by moving along the
	hue angle and gradually changing the saturation.

	Parameters
	----------
	tag : int
		The unique target identifying tag.
	hue_step : float
		Difference between two neighboring color codes in HSV space (more
		specifically, the distance in hue channel).

	Returns
	-------
	(float, float, float)
		RGB color code in range [0, 1]

	"""
	h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
	r, g, b = colorsys.hsv_to_rgb(h, 1., v)
	return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
	"""Create a unique RGB color code for a given track id (tag).

	The color code is generated in HSV color space by moving along the
	hue angle and gradually changing the saturation.

	Parameters
	----------
	tag : int
		The unique target identifying tag.
	hue_step : float
		Difference between two neighboring color codes in HSV space (more
		specifically, the distance in hue channel).

	Returns
	-------
	(int, int, int)
		RGB color code in range [0, 255]

	"""
	r, g, b = create_unique_color_float(tag, hue_step)
	return (int(255 * r), int(255 * g), int(255 * b))
