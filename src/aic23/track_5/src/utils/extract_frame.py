#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script extracts frames from a video using ffmpeg."""

from __future__ import annotations

import argparse
import glob
import os.path
from pathlib import Path

import click
import ffmpeg

from tqdm import tqdm

# region Function

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument("--source",      default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test/videos/", type=click.Path(exists=True), help="Video filepath or directory.")
parser.add_argument("--destination", default="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/Track_5/aicity2023_track5_test/images/", type=click.Path(exists=False), help="Output video filepath or directory.")
parser.add_argument("--size",        default=None, type=int, nargs="+", help="Output images/video size.")
parser.add_argument("--extension",   default="jpg", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
parser.add_argument("--verbose",     action="store_true")

def make_dir(path):
	"""Make dir"""
	if not os.path.isdir(path):
		os.makedirs(path)

def convert_video(
		source     : Path,
		destination: Path,
		size       : int | list[int],
		extension  : str,
		verbose    : bool
):
	source = sorted(glob.glob(os.path.join(source, "*.mp4")))

	for src in tqdm(source):
		basename       = os.path.basename(src)
		basename_noext = os.path.splitext(basename)[0]
		folder_ou = os.path.join(destination, basename_noext)

		# make directory
		make_dir(folder_ou)

		# extract images
		ffmpeg.input(str(src)).output(os.path.join(folder_ou, f"%08d.{extension}")).run()


# endregion


# region Main

if __name__ == "__main__":
	args = parser.parse_args()
	convert_video(
		source      = args.source,
		destination = args.destination,
		size        = args.size,
		extension   = args.extension,
		verbose     = args.verbose
	)

# endregion
