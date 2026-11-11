from __future__ import annotations

import os
import pickle
import platform
import sys
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from src.detectors.basedetector import BaseDetector
from src.factory.builder import DETECTORS
from src.objects.instance import Instance
from src.vision.utils.image import is_channel_first
from torch import Tensor
from tqdm import tqdm
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess

__all__ = [
	"YOLOX_Adapter"
]


# MARK: - YOLOX

@DETECTORS.register(name="yolox")
class YOLOX_Adapter(BaseDetector):

	# MARK: Magic Functions

	def __init__(self,
	             name     : str            = "yolox-x",
	             variant  : Optional[str]  = None,
	             half     : bool           = False,
	             model_cfg: Optional[dict] = None,
	             *args, **kwargs):
		# Set before super().__init__(), which calls init_model(), which reads both
		self.variant = variant  # the YOLOX experiment file, which is YOLOX's model config: model_cfg stays unused
		self.half    = half     # FP16 inference
		super().__init__(name=name, model_cfg=model_cfg, *args, **kwargs)

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		# NOTE: Create model

		# Get image size of detector
		self.exp             = get_exp(exp_file=self.variant)
		self.exp.test_conf   = self.min_confidence
		self.exp.nmsthre     = self.nms_max_overlap
		self.exp.num_classes = self.class_labels.num_classes()

		# Get image size of detector
		if is_channel_first(np.ndarray(self.shape)):
			self.exp.test_size = [self.shape[2], self.shape[1]]
		else:
			self.exp.test_size = [self.shape[1], self.shape[0]]

		self.rgb_means = (0.485, 0.456, 0.406)
		self.std       = (0.229, 0.224, 0.225)

		self.model = self.exp.get_model()
		checkpoint = torch.load(self.weights, map_location=self.device, weights_only=False)
		self.model.load_state_dict(checkpoint["model"])
		self.model.to(self.device)
		self.model.eval()
		if self.half:
			self.model.half()  # the camera YAMLs set half: true. No Conv+BN fusion, as in detect_yield: fusion plus float32 NMS moved 369 of 654 label lines by up to 1 px
		self.ratio = 1.0

	# MARK: Detection

	def detect(self, indexes: np.ndarray, images: np.ndarray) -> list:
		"""Detect objects in the images.

		Args:
			indexes (np.ndarray):
				Image indexes.
			images (np.ndarray):
				Images of shape [B, H, W, C].

		Returns:
			instances (list):
				List of `Instance` objects.
		"""
		# NOTE: Safety check
		if self.model is None:
			logger.error("Model has not been defined yet!")
			raise NotImplementedError

		# NOTE: Preprocess
		input_imgs = self.preprocess(images=images)
		# NOTE: Forward
		preds  = self.forward(input_imgs)
		# NOTE: Postprocess
		instances = self.postprocess(
			indexes=indexes, images=images, input_imgs=input_imgs, pred=preds
		)
		# NOTE: Suppression
		# FIXME: no co van de nghiem trong, chi chay 7cls chu khong phai la 9cls
		# allows_ids trong class_label co van de
		# instances = self.suppress_wrong_labels(instances=instances)

		return instances

	def preprocess(self, images: list) -> Tensor:
		"""Letterbox the frames as ``yolox.data.data_augment.preproc`` does, but in uint8,
		and cast them on the device: a quarter of the bytes to upload.

		Args:
			images (list):
				Frames of shape [H, W, C], BGR uint8.

		Returns:
			input (Tensor):
				[B, C, H, W] on ``self.device``: float16 when ``self.half``, else float32 equal to ``preproc``'s output.
		"""
		height, width = self.exp.test_size  # the (rows, cols) order preproc uses
		batch = np.full((len(images), height, width, 3), 114, dtype=np.uint8)
		for i, image in enumerate(images):
			self.ratio = min(height / image.shape[0], width / image.shape[1])
			resized    = cv2.resize(image, (int(image.shape[1] * self.ratio), int(image.shape[0] * self.ratio)), interpolation=cv2.INTER_LINEAR)
			batch[i, :resized.shape[0], :resized.shape[1]] = resized
		input_imgs = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).contiguous()
		return input_imgs.half() if self.half else input_imgs.float()

	@torch.no_grad()
	def forward(self, input_imgs: Tensor) -> list:
		"""One forward pass and one class-agnostic NMS call for the whole batch.

		Args:
			input_imgs (Tensor):
				Input images of shape [B, C, H, W].

		Returns:
			pred (list):
				One [N, 7] float32 ndarray per image (x1, y1, x2, y2, obj_conf, class_conf, class);
				[0, 7] when nothing is detected (yolox's postprocess gives None there).
		"""
		# Box-corner math and the confidence mask run in the model's dtype (FP16 when self.half), as in detect_yield: don't cast the model output to float32 first
		outputs = postprocess(self.model(input_imgs), self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre, class_agnostic=True)
		return [np.empty((0, 7), dtype=np.float32) if output is None else output.cpu().numpy() for output in outputs]

	def postprocess(
			self,
			indexes   : np.ndarray,
			images    : np.ndarray,
			input_imgs: Tensor,
			pred      : Tensor,
			*args, **kwargs
	) -> list:
		"""Postprocess the prediction.

		Args:
			indexes (np.ndarray): order of images
				Image indexes.
			images (np.ndarray): raw images
				Images of shape [B, H, W, C].
			input_imgs (Tensor): pre_processed images
				Input image of shape [B, C, H, W].
			pred (Tensor): results from model
				Prediction.

		Returns:
			instances (list):
				List of `Instances` objects.
		"""
		# NOTE: Create Detection objects
		instances = []

		for idx, (frame_index, result) in enumerate(zip(indexes, pred)):
			inst  = []
			xyxys = result[: , 0 : 4] / self.ratio
			confs = result[: , 4] * result[: , 5]
			clses = result[: , 6]

			# No second confidence filter: yolox's postprocess already applied test_conf (= min_confidence) in the model's dtype, as detect_yield did

			for bbox_xyxy, conf, cls in zip(xyxys, confs, clses):
				confident   = float(conf)
				class_id    = int(cls)
				class_label = self.class_labels.get_class_label(
					key="train_id", value=class_id
				)
				inst.append(
					Instance(
						frame_index = frame_index,
						bbox        = bbox_xyxy,
						confidence  = confident,
						class_label = class_label,
						label       = class_label,
						class_id    = class_id
					)
				)

			instances.append(inst)
		return instances

	# MARK: Detect a whole source (the GUI's Detection button)

	def detect_source(
			self,
			source     : str,
			name       : str,
			output     : str,
			roi_region : str,
			config_main: dict,
	) -> Iterator[np.ndarray]:
		"""Detect every frame of a video, an image folder or an image, and write what the next stages read.

		Each frame is ANDed with the `roi_region` mask before detection, and the crops are cut from the
		masked frame. With `dst_*` from `config_main["data_writer"]` and `folder_out` from
		`config_main["detector"]`, it writes under `output`:
			<dst_label>/<folder_out>/<name>/<frame:08d>.txt          one YOLO box and its confidence per line
			<dst_crop>/<folder_out>/<name>/<frame:08d>_<det:06d>.jpg  one crop per box
			<dst_debug>/<folder_out>/<name>.mp4                       the annotated frames
			<dst_crop_pkl>/<folder_out>/<name>/<name>_dets_crop.pkl   every box, after the last frame

		Args:
			source (str):
				Video file, image folder or image file.
			name (str):
				Camera name.
			output (str):
				Root of the outputs.
			roi_region (str):
				Mask image; detection ignores its black pixels. No mask if it can't be read.
			config_main (dict):
				The camera's YAML config.

		Yields:
			im0 (np.ndarray):
				Each frame with its boxes drawn, once its outputs are written.
		"""
		data_writer = config_main["data_writer"]
		folder_out  = config_main["detector"]["folder_out"]

		# Get the ignore region
		ignore_region = cv2.imread(roi_region)

		os.makedirs(os.path.join(output, data_writer["dst_debug"], folder_out), exist_ok=True)
		for dst in ("dst_label", "dst_crop", "dst_crop_pkl"):
			os.makedirs(os.path.join(output, data_writer[dst], folder_out, name), exist_ok=True)

		# Determine input type
		image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
		is_video  = True
		files     = []
		pbar      = None
		if os.path.isdir(source):
			files    = sorted(os.path.join(source, f) for f in os.listdir(source) if os.path.splitext(f)[-1].lower() in image_ext)
			is_video = False
		elif os.path.splitext(source)[-1].lower() in image_ext:
			files    = [source]
			is_video = False

		if is_video:
			cap    = cv2.VideoCapture(source)
			total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			pbar = tqdm(total=total_frames, desc=f"Processing video {name}: ")
			width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps    = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
			
		else:
			if len(files) == 0:
				print(f"No images found in {source}.")
				return
			height, width = cv2.imread(files[0]).shape[:2]
			fps = 30
			total_frames = len(files)
			pbar = tqdm(total=total_frames, desc=f"Processing images {name}: ")

		save_path  = os.path.join(output, data_writer["dst_debug"], folder_out, f"{name}.mp4")
		vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

		out_dict = {}
		names    = self.class_labels.names()
		colors   = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

		index_image = 0
		while True:
			if is_video:
				ret, frame = cap.read()
				if not ret:
					break
			else:
				if index_image >= len(files):
					break
				frame = cv2.imread(files[index_image])
				if frame is None:
					index_image += 1
					continue

			im0 = frame.copy()

			# Apply ignored region logic if valid
			if ignore_region is not None:
				if ignore_region.shape[:2] != frame.shape[:2]:
					ignore_region = cv2.resize(ignore_region, (width, height))
				frame = cv2.bitwise_and(frame, ignore_region)

			[instances] = self.detect(indexes=[index_image], images=[frame])

			txt_path = os.path.join(output, data_writer["dst_label"], folder_out, name, f"{index_image:08d}")
			det_path = os.path.join(output, data_writer["dst_crop"] , folder_out, name, f"{index_image:08d}")

			# Details text log, e.g. "19_SUWON 2 buss, 1 truck, "; it also counts the boxes skipped below
			counts  = Counter(inst.class_label["train_id"] for inst in instances)
			str_out = f"{name} " + "".join(f"{n} {names[c]}{'s' * (n > 1)}, " for c, n in sorted(counts.items()))

			det_num = 0
			for inst in instances:
				c     = inst.class_label["train_id"]
				score = inst.confidence

				x1, y1, x2, y2 = map(int, inst.bbox)
				x1 = max(0, x1)
				y1 = max(0, y1)
				x2 = min(width - 1, x2)
				y2 = min(height - 1, y2)

				if x1 >= x2 or y1 >= y2:
					continue

				x_c = (x1 + x2) / 2 / width
				y_c = (y1 + y2) / 2 / height
				w_  = (x2 - x1) / width
				h_  = (y2 - y1) / height
				with open(txt_path + '.txt', 'a') as f:
					f.write(f"{c} {x_c:.6f} {y_c:.6f} {w_:.6f} {h_:.6f} {score:.6f}\n")

				label = f'{names[c]} {score:.2f}'
				color = tuple(int(x) for x in colors[c])
				self.draw_detection_box(im0, (x1, y1, x2, y2), label, color)

				# Crop and save
				det_name     = f"{index_image:08d}_{det_num:06d}"
				det_img_path = f"{det_path}_{det_num:06d}.jpg"

				crop_img = frame[y1:y2, x1:x2]
				if crop_img.size > 0:
					cv2.imwrite(det_img_path, crop_img)

				out_dict[det_name] = {
					'bbox'   : (x1, y1, x2, y2),
					'frame'  : f"{index_image:08d}",
					'id'     : det_num,
					'imgname': f"{det_name}.jpg",
					'class'  : c,
					'conf'   : score
				}
				det_num += 1

			# show the counts of detected objects in the console
			# e.g. "19_SUWON 2 buss, 1 truck,
			# print(f"{str_out} Done.")

			vid_writer.write(im0)
			yield im0

			index_image += 1
			if pbar is not None:
				pbar.update(1)
		if pbar is not None:
			pbar.close()

		if is_video:
			cap.release()
		vid_writer.release()

		pkl_path = os.path.join(output, data_writer["dst_crop_pkl"], folder_out, name, f'{name}_dets_crop.pkl')
		with open(pkl_path, 'wb') as f:
			pickle.dump(out_dict, f)

	@staticmethod
	def draw_detection_box(im0, box, label, color):
		"""Draw a YOLOX detection box with adaptive thickness and a filled label background."""
		x1, y1, x2, y2 = map(int, box)
		tl = max(1, round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1)
		tf = max(tl - 1, 1)
		c1, c2 = (x1, y1), (x2, y2)
		cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
		if label:
			t_size   = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=tl / 3, thickness=tf)[0]
			c2_label = (c1[0] + t_size[0], max(0, c1[1] - t_size[1] - 3))
			cv2.rectangle(im0, c1, c2_label, color, -1, lineType=cv2.LINE_AA)
			cv2.putText(im0, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
