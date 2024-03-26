

from __future__ import annotations

import abc
import os
import sys
import platform
from pathlib import Path
import pickle
from typing import Optional

import cv2
import numpy as np

from core.factory.builder import HEURISTICS
from core.type.type import Dim3
from core.utils.device import select_device
from core.utils.image import is_channel_first

# NOTE: check system path
is_exist = False
for path in sys.path:
	if "ultralytics" in path:
		is_exist = True
if not is_exist:
	FILE = Path(__file__).resolve()
	ROOT = os.path.join(FILE.parents[0].parents[0])  # YOLOv8 root directory
	if str(ROOT) not in sys.path:
		sys.path.append(str(ROOT))  # add ROOT to PATH
	if platform.system() != 'Windows':
		ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.models.yolo import pose

__all__ = [
	"HeuristicPose"
]


class GetPoseKeypoint:
	"""
	This class is used to define the pose keypoints.
	"""
	NOSE:           int = 0
	LEFT_EYE:       int = 1
	RIGHT_EYE:      int = 2
	LEFT_EAR:       int = 3
	RIGHT_EAR:      int = 4
	LEFT_SHOULDER:  int = 5
	RIGHT_SHOULDER: int = 6
	LEFT_ELBOW:     int = 7
	RIGHT_ELBOW:    int = 8
	LEFT_WRIST:     int = 9
	RIGHT_WRIST:    int = 10
	LEFT_HIP:       int = 11
	RIGHT_HIP:      int = 12
	LEFT_KNEE:      int = 13
	RIGHT_KNEE:     int = 14
	LEFT_ANKLE:     int = 15
	RIGHT_ANKLE:    int = 16


@HEURISTICS.register(name="HeuristicPose")
class HeuristicPose():

	# MARK: Magic Functions

	def __init__(
			self,
			name           : Optional[str]       = "heuristic_pose",
			class_ids      : Optional[list[int]] = [1, 2],
			weights        : Optional[str]       = None,
			shape          : Optional[Dim3]      = None,
			min_confidence : Optional[float]     = 0.5,
			nms_max_overlap: Optional[float]     = 0.4,
			device         : Optional[str]       = None,
			*args, **kwargs
	):
		"""Initialize the heuristic.

		Args:
			name (str, optional):
				Name of the heuristic. Default: "heuristic".
		"""
		self.name            = name
		self.class_ids       = class_ids if class_ids else []
		self.weights         = weights
		self.shape           = shape
		self.min_confidence  = min_confidence
		self.nms_max_overlap = nms_max_overlap
		self.device          = select_device(device= device)
		self.get_keypoint    = GetPoseKeypoint()

		# init model
		self.init_model()

	def init_model(self):
		"""
		Initializes the model for pose prediction.

		This method first determines the image size based on the shape of the input.
		If the input is channel-first, the image size is set to the third dimension of the shape.
		Otherwise, the image size is set to the first dimension of the shape.

		Then, it creates a PosePredictor object with the specified parameters and image size.
		The PosePredictor is initialized with a source image of zeros with the same size as the input image.
		"""
		if is_channel_first(np.ndarray(self.shape)):
			self.img_size = self.shape[2]
		else:
			self.img_size = self.shape[0]

		self.model = pose.PosePredictor(overrides={
			'imgsz'  : self.img_size,
			'conf'   : self.min_confidence,
			'iou'    : self.nms_max_overlap,
			'show'   : False,
			'verbose': False,
			'save'   : False,
			'device' : self.device,
		})
		_ = self.model(source=np.zeros([self.img_size, self.img_size, 3]), model=self.weights)

	# MARK: Abstract Methods

	def run(self, detection_instance, instance_queue, image):
		"""
		Run the heuristic.

		Args:
			detection_instance (Instance):
				The detection of whole motorbike with driver and passengers.
			instance_queue (list[Instance]):
				Instances to process.
			image (np.array):
				Whole image to process
		"""
		instance_list = []
		for instance in instance_queue:
			if instance.class_id in self.class_ids:
				instance_list = instance_list + self.process(detection_instance, instance, image)
			else:
				instance_list.append(instance)
		return instance_list

	def process(self, detection_instance, instance, image):
		"""Process the instance.

		Args:
			detection_instance (Instance):
				The detection of whole motorbike with driver and passengers.
			instance (Instance):
				Instance to process.
			image (np.array):
				Whole image to process
		"""
		# get instance image
		instance_list  = [instance]
		# instance_image = image[instance.bbox[1]:instance.bbox[3], instance.bbox[0]:instance.bbox[2]]
		#
		# # run model
		# results = self.model(source=instance_image)
		#
		# # process
		# for result in results:
		# 	if result is None:
		# 		continue
		#
		# 	# load keypoints
		# 	xys   = result.keypoints.xy.cpu().numpy().astype(int)
		#
		# 	if len(xys) == 0:
		# 		continue
		#
		# 	# load confidence score of keypoint
		# 	if result.keypoints.conf is None:
		# 		continue
		# 	confs = result.keypoints.conf.cpu().numpy().astype(float)
		#
		# 	# TODO: DUONG LAM CHO NAY NHE
		# 	# update instance
		# 	for xy, conf in zip(xys, confs):
		# 		# # example
		# 		# nose     = xy[self.get_keypoint.NOSE]
		# 		# left_eye = xy[self.get_keypoint.LEFT_EYE]
		#
		# 		# DEBUG:
		# 		print("************")
		# 		print(len(xy))
		# 		print(conf)
		# 		print("************")
		return instance_list


if __name__ == "__main__":
	heuristic     = HeuristicPose()
	frame_current = None
	frame_index   = None

	identifications_pickle_loader = pickle.load(
		open(f"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/output_aic24/identifier/yolov8x_320_9cls_track_5_24_crop_train_equal_val_v3/036/identifications_queue.pkl",
			 'rb'))

	for index_frame_ident, _, batch_identifications in identifications_pickle_loader:
		instance_heuristic_queue = []
		for identification_instance in batch_identifications:
			if frame_current is None or frame_index is None or frame_index != index_frame_ident:
				frame_current = cv2.imread(
					os.path.join(
						"/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2024/Track_5/aicity2024_track5_train/images/",
						f"036{index_frame_ident:05d}.jpg")
				)
				frame_index = index_frame_ident
			instance_heuristic_queue.append(identification_instance)
		heuristic.run(instance_heuristic_queue, frame_current)


