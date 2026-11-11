#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter
from timeit import default_timer as timer

import yaml
from src import MTMCTrackingCamera
from src.configuration import data_dir, load_config
from src.utils import console

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# MARK: - Args

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument(
	"--dataset", default="tsd_suwon_19_46_20",
	help="Dataset to run on."
)
# parser.add_argument(
# 	"--subset", default="dataset_a",
# 	help="Subset name. One of: [`dataset_a`, `dataset_b`]."
# )
parser.add_argument(
	"--config", default="default.yaml",
	help="Config file for each camera. Final path to the config file "
		 "is: tss/data/[dataset]/configs/[config]/"
)
parser.add_argument(
	"--config_feat_merge", default="reid_merge.yaml",
	help="Feature-merge config. Final path to the config file "
		 "is: tss/data/[dataset]/configs/[config_feat_merge]"
)
parser.add_argument(
	"--images_extraction", action='store_true', help="Should extract images."
)
parser.add_argument(
	"--detection", action='store_true', help="Should run detection."
)
parser.add_argument(
	"--feature_extraction",  action='store_true', help="Should run feature extraction."
)
parser.add_argument(
	"--feature_merge",  action='store_true', help="Should run Merge all feature."
)
parser.add_argument(
	"--detection_filter",  action='store_true', help="Should run filter detection of all feature."
)
parser.add_argument(
	"--use_dets_feat_filter", action='store_true', help="Should use the detection feature after filtering."
)
parser.add_argument(
	"--tracking", action='store_true', help="Should run Tracking function."
)
parser.add_argument(
	"--tracking_postprocess", action='store_true', help="Should run Post process for tracking."
)
parser.add_argument(
	"--matching_zone", action='store_true', help="Should run Matching trajectory with zoneprocess."
)
parser.add_argument(
	"--use_mots_feat_raw", action='store_true', help="Should run Matching trajectory with zoneprocess."
)
parser.add_argument(
	"--matching_scene", action='store_true', help="Should run Matching trajectory all zone process."
)
parser.add_argument(
	"--write_result", action='store_true', help="Should run Writing result."
)
parser.add_argument(
	"--save_dets_img", action='store_true', help="Should visualize the detection result."
)
parser.add_argument(
	"--save_dets_img_filter", action='store_true', help="Should visualize the detection result after filter."
)
parser.add_argument(
	"--save_tracks_img", action='store_true', help="Should visualize the images."
)
parser.add_argument(
	"--save_tracks_zone", action='store_true', help="Should visualize the images."
)
parser.add_argument(
	"--save_mtmc_result", action='store_true', help="Should visualize final result."
)
parser.add_argument(
	"--verbose", action='store_true', help="Should visualize the images."
)

Camera = MTMCTrackingCamera


# MARK: - Main Function

def main():
	# NOTE: Start timer
	process_start_time = perf_counter()
	camera_start_time  = perf_counter()

	# NOTE: Parse camera config
	args        = parser.parse_args()
	config_path = os.path.join(data_dir, args.dataset, "configs", args.config)
	camera_cfg  = load_config(config_path)

	# DEBUG: print camera config
	# print(camera_cfg)

	# Update value from args
	camera_cfg["dataset"]      = args.dataset
	# camera_cfg["subset"]       = args.subset
	camera_cfg["verbose"]      = args.verbose
	camera_cfg["process"]      = {
		"images_extraction"            : args.images_extraction,  # Extract images from video
		"function_dets"                : args.detection,  # Detection
		"save_dets_crop"               : True,
		"save_dets_pkl"                : True,
		"save_dets_img"                : args.save_dets_img,
		"save_dets_txt"                : True,
		"function_dets_crop_feat"      : args.feature_extraction,  # Feature extraction of Detection
		"function_dets_crop_feat_merge": args.feature_merge,  # Merge Feature extraction of Detection
		"detection_filter"             : args.detection_filter,  # Run filter of all detection result
		"use_dets_feat_filter"         : args.use_dets_feat_filter,  # Whether use detection raw or filter
		"save_dets_img_filter"         : args.save_dets_img_filter,  # Draw detection result after filter
		"function_tracking"            : args.tracking,  # Tracking
		"save_tracks_img"              : args.save_tracks_img,
		"function_tracks_postprocess"  : args.tracking_postprocess,  # Post process of Tracking
		"function_matching_zone"       : args.matching_zone,  # Matching trajectory with zone
		"use_mots_feat_raw"            : args.use_mots_feat_raw,  # Whether use feat_raw of feat with post process
		"save_tracks_zone"             : args.save_tracks_zone,  # Save track with zone
		"function_matching_scene"      : args.matching_scene,  # Matching trajectory with zone
		"function_write_result"        : args.write_result,  # Writing result
		"save_mtmc_result"             : args.save_mtmc_result,  # Draw final result
	}

	# NOTE: add merge feat config if needed
	camera_cfg["merge_feat"] = None
	config_path = os.path.join(data_dir, args.dataset, "configs", args.config_feat_merge)
	with open(config_path) as f:
		camera_cfg["featuremerger"] = yaml.load(f, Loader=yaml.FullLoader)

	# NOTE: Define camera
	camera           = Camera(**camera_cfg)
	camera_init_time = perf_counter() - camera_start_time

	# NOTE: Process
	camera.run()

	# NOTE: End timer
	total_process_time = perf_counter() - process_start_time
	console.log(f"Total processing time: {total_process_time} seconds.")
	console.log(f"Camera init time: {camera_init_time} seconds.")
	console.log(f"Actual processing time: "
				f"{total_process_time - camera_init_time} seconds.")


# MARK: - Entry point

if __name__ == "__main__":
	main()
