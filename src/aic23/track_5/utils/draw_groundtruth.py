#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import sys

import cv2
from tqdm import tqdm

from one.core import AppleRGB


def make_dir(folder_path):
	if not os.path.isdir(folder_path):
		os.makedirs(folder_path)


def extract_image():
	# init parameter
	folder_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/videos/"
	folder_ou = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/images/"

	# NOTE: get list
	list_video = sorted(glob.glob(os.path.join(folder_in, "*.mp4")))

	# NOTE: extraction
	for video_path in tqdm(list_video):
		# get name
		basename       = os.path.basename(video_path)
		basename_noext = os.path.splitext(basename)[0]

		# create folder
		folder_ou_video = os.path.join(folder_ou, basename_noext)
		make_dir(folder_ou_video)

		cam = cv2.VideoCapture(video_path)
		index = 0
		while True:

			# reading from frame
			ret, frame = cam.read()

			if ret:
				# Start point
				index      = index + 1
				name_image = f"{index:08d}"

				# writing the extracted images
				cv2.imwrite(os.path.join(folder_ou_video, f"{name_image}.jpg"), frame)
			else:
				break

		# Release all space and windows once done
		cam.release()

		# DEBUG:
		# print(f"{index:08d}")


def read_groundtruth():
	# NOTE: init parameter
	gth_path = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/gt.txt"

	# NOTE: read file
	# <video_id>, <frame>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
	gth = []
	with open(gth_path, 'r') as f_open:
		lines = f_open.readlines()
		for line in lines:
			words = line.replace('\n', '').split(',')
			gth.append([eval(i) for i in words])

	return gth


# 1, motorbike
# 2, DHelmet
# 3, DNoHelmet
# 4, P1Helmet
# 5, P1NoHelmet
# 6, P2Helmet
# 7, P2NoHelmet
colors = {
	1: {"name": "motorbike" , "color": AppleRGB.WHITE.value},
	2: {"name": "DHelmet"   , "color": AppleRGB.PINK.value},
	3: {"name": "DNoHelmet" , "color": AppleRGB.RED.value},
	4: {"name": "P1Helmet"  , "color": AppleRGB.PURPLE.value},
	5: {"name": "P1NoHelmet", "color": AppleRGB.INDIGO.value},
	6: {"name": "P2Helmet"  , "color": AppleRGB.GREEN.value},
	7: {"name": "P2NoHelmet", "color": AppleRGB.TEAL.value}
}

def draw_one_video(index_video, video_folder_in, video_folder_ou, gth):
	# get list
	image_list = sorted(glob.glob(os.path.join(video_folder_in, f"*.jpg")))

	# filter ground-truth for video
	gth_video = [label for label in gth if label[0] == int(index_video)]

	# NOTE: drawing
	for image_path_in in tqdm(image_list, desc=f"{index_video} draws"):
		# NOTE: get name
		basename       = os.path.basename(image_path_in)
		basename_noext = os.path.splitext(basename)[0]

		# NOTE: filter ground truth for image
		gth_image = [label for label in gth_video if label[1] == int(basename_noext)]

		# NOTE: drawing
		img     = cv2.imread(image_path_in)
		overlay = img.copy()
		output  = img.copy()
		alpha   = 0.5
		for label in gth_image:
			color = colors[label[7]]["color"]

			# <video_id>, <frame>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
			# NOTE: draw boundingox
			pt1, pt2 = (label[3], label[4]), (label[3] + label[5], label[4] + label[6])
			img = cv2.rectangle(img, pt1, pt2, colors[label[7]]["color"], 3)

			# NOTE: add label
			label = f"{colors[label[7]]['name']} - {label[2]}"
			cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

		# write image
		image_path_ou = os.path.join(video_folder_ou, basename)
		cv2.imwrite(image_path_ou, img)



def draw_videos():
	# NOTE: init parameter
	folder_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/images/"
	folder_ou = "/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5/images_draw/"

	# NOTE: get list
	video_list = sorted(os.listdir(folder_in))

	# NOTE: get ground-truth
	gth = read_groundtruth()

	# NOTE: draw videos
	for index_video in tqdm(video_list):
		video_folder_in = os.path.join(folder_in, index_video)
		video_folder_ou = os.path.join(folder_ou, index_video)

		# create folder
		make_dir(video_folder_ou)

		draw_one_video(index_video, video_folder_in, video_folder_ou, gth)



def main():
	# NOTE: we have to extract image first
	# extract_image()

	# NOTE: drawing from extracted image
	draw_videos()


if __name__ == "__main__":
	main()

