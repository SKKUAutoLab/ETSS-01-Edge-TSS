#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pickle
import sys
import os
import glob
from enum import Enum
import shutil
import random
import threading
import multiprocessing
from operator import itemgetter, attrgetter

import multiprocessing
import threading

from tqdm import tqdm
import numpy as np
import cv2

from classes_aic23_track5 import *

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--draw_final", action='store_true', help="Should run detection."
)
parser.add_argument(
	"--draw_pickle", action='store_true', help="Should run detection."
)
parser.add_argument(
	"--path_final",
	default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/outputs/final_result_s1.txt",
	help="Path to pickle folder."
)
parser.add_argument(
	"--path_pickle_in",
	default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/outputs/dets_crop_pkl/yolov8x6/",
	help="Path to pickle folder."
)
parser.add_argument(
	"--path_video_in",
	default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/videos/",
	help="Path to output folder."
)
parser.add_argument(
	"--path_video_out",
	default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/outputs/dets_crop_pkl_debug/",
	help="Path to output folder."
)


# NOTE: SMALL UTILITIES -------------------------------------------------------
def make_dir(path):
	"""Make dir"""
	if not os.path.isdir(path):
		os.makedirs(path)


class AppleRGB(Enum):
	"""Apple's 12 RGB colors."""

	RED    = (255, 59 , 48)
	GREEN  = (52 , 199, 89)
	BLUE   = (0  , 122, 255)
	ORANGE = (255, 149, 5)
	BROWN  = (162, 132, 94)
	PURPLE = (88 , 86 , 214)
	TEAL   = (90 , 200, 250)
	INDIGO = (85 , 190, 240)
	BLACK  = (0  , 0  , 0)
	PINK   = (255, 45 , 85)
	WHITE  = (255, 255, 255)
	GRAY   = (128, 128, 128)
	YELLOW = (255, 204, 0)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
	"""Plots one bounding box on image img"""
	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
	cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# NOTE: VISUALIZE FINAL RESULT ------------------------------------------------
# region VISUALIZE FINAL RESULT

def read_result(gt_path):

	labels = []
	with open(gt_path, 'r') as f_open:
		lines = f_open.readlines()
		for line in lines:
			labels.append([int(word) if index < len(line.replace('\n', '').split(',')) - 1 else float(word) for index, word in enumerate(line.replace('\n', '').split(','))])
	return labels


def draw_final_video(video_path_in, video_path_ou, label_video, colors, labels_name):
	# read images
	images_path      = sorted(glob.glob(os.path.join(video_path_in, "*.jpg")))
	img              = cv2.imread(images_path[0])
	height, width, _ = img.shape
	fps              = 10
	index_frame      = 0

	# generate new video
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video = cv2.VideoWriter(video_path_ou, fourcc, fps, (width, height))

	for image_path in images_path:
		frame = cv2.imread(image_path)
		index_frame = index_frame + 1

		label_image = [label for label in label_video if label[1] == index_frame]
		if len(label_image) > 0:
			for label in label_image:
				cls_index = int(label[6])
				box = [
					label[2],
					label[3],
					label[2] + label[4],
					label[3] + label[5]
				]
				plot_one_box(
					x     = box,
					img   = frame,
					color = colors[cls_index],
					label = f"{labels_name[cls_index - 1]}::{label[7]:.8f}",
				)

		# writing the extracted images
		video.write(frame)

	# Release all space and windows once done
	video.release()


def draw_final_result(args):
	# initiate parameters
	folder_out = args.path_video_out
	folder_in  = args.path_video_in

	# create output folder
	make_dir(folder_out)

	# get result
	result_path = args.path_final
	labels = read_result(result_path)

	# initial color
	colors = []
	for index, color in enumerate(AppleRGB):
		# print(index, color, color.value)
		colors.append(color.value)

	# get list of label
	labels_name = get_list_7_classses()

	# Get list video
	videos_name = sorted(os.listdir(folder_in))

	# draw videos
	for video_name in tqdm(videos_name, desc="Drawing final result"):
		basename_noext = video_name
		video_index    = int(basename_noext)
		video_path_ou  = os.path.join(folder_out, f"{basename_noext}.mp4")
		video_path     = os.path.join(folder_in, video_name)

		# draw one video
		label_video = [label for label in labels if label[0] == video_index]
		draw_final_video(video_path, video_path_ou, label_video, colors, labels_name)

		# DEBUG: run 1 video
		# break

# endregion

# NOTE: VISUALIZE PICKLE RESULT -----------------------------------------------
# region VISUALIZE FINAL RESULT

def draw_pickle_image(video_path_in, video_path_ou, pkl_path, colors, labels_name):
	# read pkl
	dets_pkl = pickle.load(open(pkl_path, 'rb'))

	# read images
	images_path      = sorted(glob.glob(os.path.join(video_path_in, "*.jpg")))
	img              = cv2.imread(images_path[0])
	height, width, _ = img.shape
	fps              = 10
	index_frame      = 0

	# generate new video
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video  = cv2.VideoWriter(video_path_ou, fourcc, fps, (width, height))

	for image_path in images_path:
		frame = cv2.imread(image_path)
		index_frame = index_frame + 1

		label_image = [label for label in dets_pkl if int(label["frame_id"]) == index_frame]

		if len(label_image) >  0:
			for label in label_image:
				box = [
					label["bbox"][0],
					label["bbox"][1],
					label["bbox"][2],
					label["bbox"][3]
				]
				plot_one_box(
					x     = box,
					img   = frame,
					color = colors[int(label["id"])],
					label = f"{labels_name[int(label['class_id'])]}::{label['conf']:.8f}"
				)

		# writing the extracted images
		video.write(frame)

	# Release all space and windows once done
	video.release()


def draw_pickles(args):
	# get the parameters
	path_pickle_in = args.path_pickle_in
	path_video_out = args.path_video_out
	path_video_in  = args.path_video_in

	# create output folder
	make_dir(path_video_out)

	# initial color
	colors = []
	for index, color in enumerate(AppleRGB):
		# print(index, color, color.value)
		colors.append(color.value)

	# get list of label
	labels_name = get_list_7_classses()

	# get list pkl
	pkl_paths = sorted(glob.glob(os.path.join(path_pickle_in, "*.pkl")))

	# draw videos
	for pkl_path in tqdm(pkl_paths, desc="Drawing pickle result"):
		basename       = os.path.basename(pkl_path)
		basename_noext = os.path.splitext(basename)[0]
		video_index    = int(basename_noext)

		video_path_in  = os.path.join(path_video_in   , f"{basename_noext}")
		video_path_ou  = os.path.join(path_video_out  , f"{basename_noext}.mp4")

		# draw one video
		draw_pickle_image(video_path_in, video_path_ou, pkl_path, colors, labels_name)

# endregion

# NOTE: MAIN ------------------------------------------------------------------


def main():
	args = parser.parse_args()
	if args.draw_final:
		draw_final_result(args)
	elif args.draw_pickle:
		draw_pickles(args)


if __name__ == "__main__":
	main()
