#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
	def __init__(self):
		super(Exp, self).__init__()
		self.depth = 0.33
		self.width = 0.50
		self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
		self.input_size = (640, 640)  # (height, width)
		self.test_size  = (640, 640)

		# Define yourself dataset path
		self.data_dir  = "/media/vsw/SSD_2/2_dataset/tfe_dataset/45a_SUWON/"
		self.train_ann = "instances_train2017.json"
		self.val_ann   = "instances_val2017.json"

		self.num_classes      = 8

		self.max_epoch        = 300
		self.data_num_workers = 8
		self.eval_interval    = 20

		# ---------- transform config ------------ #
		self.mosaic_prob = 1.0
		self.mixup_prob  = 1.0
		self.hsv_prob    = 1.0
		self.flip_prob   = 0.5
