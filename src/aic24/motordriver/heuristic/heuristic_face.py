

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
from core.objects.instance import Instance
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

from ultralytics.models.yolo import detect

__all__ = [
	"HeuristicFace"
]


@HEURISTICS.register(name="HeuristicFace")
class HeuristicFace():

	# MARK: Magic Functions

	def __init__(
			self,
			name           : Optional[str]       = "heuristic_face",
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

		# init model
		self.init_model()

	def init_model(self):
		"""
		"""
		if is_channel_first(np.ndarray(self.shape)):
			self.img_size = self.shape[2]
		else:
			self.img_size = self.shape[0]

		self.model = detect.DetectionPredictor(overrides={
			'imgsz'   : self.img_size,
			'conf'    : self.min_confidence,
			'iou'     : self.nms_max_overlap,
			'show'    : False,
			'verbose' : False,
			'save'    : False,
			'device'  : self.device,
			'max_det' : 1600
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
		instance_list  = []
		instance_index = 0
		# get the maximum instance index
		for instance in instance_queue:
			instance_index = max(instance_index, instance.id[2])

		for instance in instance_queue:
			# only process with driver has confidence > 0.1
			if instance.class_id in self.class_ids and instance.confidence > 0.1:
				instance_list_temp, instance_index = self.process(detection_instance, instance, image, instance_index)
				instance_list                      = instance_list + instance_list_temp
			else:
				instance_list.append(instance)
		return instance_list

	def expand_face_to_body(self, face_box, image):
		"""
		Expand from face bounding box to full body box.

		Args:
			face_box (xyxy):
				The bounding box from face detection of P0NoHelmet.
			image (np.array):
				The image to process.
		"""
		img_height, img_width = image.shape[: 2]
		face_width            = face_box[2] - face_box[0]
		face_height           = face_box[3] - face_box[1]
		x1                    = max(0, face_box[0] - face_width)
		y1                    = max(0, face_box[1] - 20)
		x2                    = min(img_width - 1  , face_box[2] + face_width)
		y2                    = min(img_height - 1 , face_box[3] + (face_height * 2))

		return (x1, y1, x2, y2)

	def process(self, detection_instance, instance_driver, image, instance_index):
		"""Process the instance.

		Args:
			detection_instance (Instance):
				The detection of whole motorbike with driver and passengers.
			instance_driver (Instance):
				Instance to process.
			image (np.array):
				Whole image to process
			instance_index (int):
				Current index of instance in the motorbike
		"""

		## Return : input instance + instance_P0
		# get instance image
		instance_list = [instance_driver]
		crop_image    = image[instance_driver.bbox[1]:instance_driver.bbox[3], instance_driver.bbox[0]:instance_driver.bbox[2]]

		# run model
		results = self.model(source=crop_image)

		# DEBUG: img for saving
		# img = crop_image.copy()
		# save = False

		# process
		for idx, result in enumerate(results):
			y_centers = []
			xyxys     = result.boxes.xyxy.cpu().numpy()
			confs     = result.boxes.conf.cpu().numpy()

			if len(xyxys) > 1:
				for xyxy in xyxys:
					y_centers.append((xyxy[1] + xyxy[3]) / 2)

				top_box    = xyxys[np.argmin(y_centers)]  # driver face detection box
				bottom_box = xyxys[np.argmax(y_centers)]  # P0 face detection box
				bot_conf   = confs[np.argmax(y_centers)]

				if top_box[3] < bottom_box[1]:
					delta_h = bottom_box[1] - top_box[3]
					avg_h   = ((top_box[3] - top_box[1]) + (bottom_box[3] - bottom_box[1])) / 2

					if delta_h < (2*avg_h):
						p0_bbox = self.expand_face_to_body(bottom_box, crop_image)

						# DEBUG: draw on save image
						# start_point = (int(p0_bbox[0]), int(p0_bbox[1]))
						# end_point = (int(p0_bbox[2]), int(p0_bbox[3]))
						# color = (128, 0, 128)
						# img = cv2.rectangle(img, start_point, end_point, color)

						# add P0 to instance
						instance_index += 1
						confident       = float(bot_conf)
						class_id        = 8
						instance_id     = detection_instance.id + [instance_index]  # frame_index, bounding_box index, instance_index
						identification_result = {
							'video_name' : detection_instance.video_name,
							'frame_index': detection_instance.frame_index,
							'bbox'       : p0_bbox,
							'class_id'   : class_id,
							'id'         : instance_id,
							'confidence' : (float(detection_instance.confidence) * confident),
							'image_size' : detection_instance.image_size
						}
						instance_list.append(Instance(**identification_result))



		# DEBUG: save to review
		# save_name = folder_output + str(instance_driver.id[0]) + '_' + str(instance_driver.id[1]) + '_' + str(instance_driver.id[2]) + '.jpg'
		# if save:
		# 	print(save_name)
		# 	cv2.imwrite(save_name, img)

		return instance_list, instance_index


if __name__ == "__main__":
	heuristic     = HeuristicFace()
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


