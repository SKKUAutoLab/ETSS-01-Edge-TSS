import argparse
import os
import pickle
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess


def draw_detection_box(im0, box, label, color):
	"""Draw a YOLOX detection box with adaptive thickness and a filled label background."""
	x1, y1, x2, y2 = map(int, box)
	tl = max(1, round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1)
	tf = max(tl - 1, 1)
	c1, c2 = (x1, y1), (x2, y2)
	cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
	if label:
		t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=tl / 3, thickness=tf)[0]
		c2_label = (c1[0] + t_size[0], max(0, c1[1] - t_size[1] - 3))
		cv2.rectangle(im0, c1, c2_label, color, -1, lineType=cv2.LINE_AA)
		cv2.putText(im0, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)


class Predictor(object):
	def __init__(
		self,
		model,
		exp,
		cls_names,
		trt_file  = None,
		decoder   = None,
		device    = "cpu",
		fp16      = False,
		legacy    = False,
		agnostic  = True,
	):
		self.model       = model
		self.cls_names   = cls_names
		self.decoder     = decoder
		self.num_classes = exp.num_classes
		self.confthre    = exp.test_conf
		self.nmsthre     = exp.nmsthre
		self.test_size   = exp.test_size
		self.device      = device
		self.fp16        = fp16
		self.agnostic    = agnostic
		self.preproc     = ValTransform(legacy=legacy)
		if trt_file is not None:
			from torch2trt import TRTModule
			model_trt = TRTModule()
			model_trt.load_state_dict(torch.load(trt_file))

			x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
			self.model(x)
			self.model = model_trt

	def inference(self, img):
		img_info = {"id": 0}
		if isinstance(img, str):
			img_info["file_name"] = os.path.basename(img)
			img = cv2.imread(img)
		else:
			img_info["file_name"] = None

		height, width = img.shape[:2]
		img_info["height"] = height
		img_info["width"] = width
		img_info["raw_img"] = img

		ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
		img_info["ratio"] = ratio

		img, _ = self.preproc(img, None, self.test_size)
		img = torch.from_numpy(img).unsqueeze(0)
		img = img.float()
		if self.device == "gpu":
			img = img.cuda()
			if self.fp16:
				img = img.half()  # to FP16

		with torch.no_grad():
			outputs = self.model(img)
			if self.decoder is not None:
				outputs = self.decoder(outputs, dtype=outputs.type())
			outputs = postprocess(
				outputs, self.num_classes, self.confthre,
				self.nmsthre, class_agnostic=self.agnostic
			)
		return outputs, img_info


def detect(predictor, opt, is_return_img= False):
	out, source, name_video, roi_path = opt.output, opt.source, opt.name, opt.roi_region

	# Get the ignore region
	ignore_region = cv2.imread(roi_path)

	save_dir = Path(out)
	os.makedirs(os.path.join(save_dir, 'dets_debug', 'yolox'), exist_ok=True)
	os.makedirs(os.path.join(save_dir, 'dets_label', 'yolox', name_video), exist_ok=True)
	os.makedirs(os.path.join(save_dir, 'dets_crop', 'yolox', name_video), exist_ok=True)
	os.makedirs(os.path.join(save_dir, 'dets_crop_pkl', 'yolox', name_video), exist_ok=True)

	# Determine input type
	image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
	is_video = True
	files = []
	if os.path.isdir(source):
		files = [os.path.join(source, f) for f in os.listdir(source) if os.path.splitext(f)[-1].lower() in image_ext]
		files.sort()
		is_video = False
	elif os.path.splitext(source)[-1].lower() in image_ext:
		files = [source]
		is_video = False

	if is_video:
		cap = cv2.VideoCapture(source)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
	else:
		if len(files) == 0:
			print(f"No images found in {source}.")
			return
		test_img = cv2.imread(files[0])
		height, width = test_img.shape[:2]
		fps = 30

	save_path = os.path.join(save_dir, 'dets_debug', 'yolox', f"{name_video}.mp4")
	vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

	out_dict = dict()
	names = predictor.cls_names
	colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

	index_image = 0
	t0 = time.time()

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

		t1 = time.time()
		outputs, img_info = predictor.inference(frame)
		t2 = time.time()

		txt_path  = os.path.join(save_dir, 'dets_label', 'yolox', name_video, f"{index_image:08d}")
		det_path  = os.path.join(save_dir, 'dets_crop' , 'yolox', name_video, f"{index_image:08d}")

		str_out = f"{im0.shape[1]}x{im0.shape[0]} "

		if outputs[0] is not None:
			output = outputs[0].cpu()
			bboxes = output[:, 0:4] / img_info["ratio"]
			scores = output[:, 4] * output[:, 5]
			cls = output[:, 6]

			# Filter classes if set
			if opt.classes:
				mask = np.isin(cls, opt.classes)
				bboxes = bboxes[mask]
				scores = scores[mask]
				cls = cls[mask]

			# Append details text log
			for c in np.unique(cls):
				n = (cls == c).sum()
				str_out += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

			det_num = 0
			for i in range(len(bboxes)):
				box = bboxes[i]
				score = scores[i]
				c = cls[i]

				x1, y1, x2, y2 = map(int, box)
				x1 = max(0, x1)
				y1 = max(0, y1)
				x2 = min(width - 1, x2)
				y2 = min(height - 1, y2)

				if x1 >= x2 or y1 >= y2:
					continue

				if opt.save_txt:
					x_c = (x1 + x2) / 2 / width
					y_c = (y1 + y2) / 2 / height
					w_ = (x2 - x1) / width
					h_ = (y2 - y1) / height
					with open(txt_path + '.txt', 'a') as f:
						if opt.save_conf:
							f.write(f"{int(c)} {x_c:.6f} {y_c:.6f} {w_:.6f} {h_:.6f} {score:.6f}\n")
						else:
							f.write(f"{int(c)} {x_c:.6f} {y_c:.6f} {w_:.6f} {h_:.6f}\n")

				label = f'{names[int(c)]} {score:.2f}'
				color = tuple(int(x) for x in colors[int(c)]) if colors is not None else tuple(random.randint(0, 255) for _ in range(3))
				draw_detection_box(im0, (x1, y1, x2, y2), label, color)

				# Crop and save
				det_name     = f"{index_image:08d}_{det_num:06d}"
				det_img_path = f"{det_path}_{det_num:06d}.jpg"
				det_class    = int(c)
				det_conf     = float(score)

				crop_img = frame[y1:y2, x1:x2]
				if crop_img.size > 0:
					cv2.imwrite(det_img_path, crop_img)

				out_dict[det_name] = {
					'bbox'   : (x1, y1, x2, y2),
					'frame'  : f"{index_image:08d}",
					'id'     : det_num,
					'imgname': f"{det_name}.jpg",
					'class'  : det_class,
					'conf'   : det_conf
				}
				det_num += 1

		# Output inference details
		print(f"{str_out}Done. ({t2 - t1:.3f}s)")

		# NOTE: save for GUI
		display_img_path = os.path.join(os.getcwd(), "src/views", f"display_{name_video[-1]}.jpg")
		os.makedirs(os.path.dirname(display_img_path), exist_ok=True)
		cv2.imwrite(display_img_path, im0)

		vid_writer.write(im0)

		# if we want to return the image for GUI display, we can yield it here
		# it run like a generator, and the caller can get the latest image after each detection
		# itterative call to this function will continue the detection and yield new images until the video ends
		# if is_return_img:
		# 	yield im0

		index_image += 1

	if is_video:
		cap.release()
	vid_writer.release()
	cv2.destroyAllWindows()

	pkl_path = os.path.join(save_dir, 'dets_crop_pkl', 'yolox', name_video, f'{name_video}_dets_crop.pkl')
	with open(pkl_path, 'wb') as f:
		pickle.dump(out_dict, f)

	# Output for finish all the detection
	print(f'Done. ({time.time() - t0:.3f}s)')

	return None


def create_predictor(opt):
	if opt.trt:
		opt.device = "gpu"

	# DEBUG:
	# print(f"Configs {opt.__dict__}")
	# import sys
	# sys.exit()

	exp = get_exp(opt.exp_file, None)
	if opt.conf_thres is not None:
		exp.test_conf = opt.conf_thres
	if opt.iou_thres is not None:
		exp.nmsthre   = opt.iou_thres
	if opt.img_size is not None:
		exp.test_size = (opt.img_size, opt.img_size)

	model = exp.get_model()
	if opt.device == "gpu":
		model.cuda()
		if opt.fp16:
			model.half()  # to FP16
	model.eval()

	if not opt.trt:
		ckpt_file = opt.weights
		print(f"Loading checkpoint from: {ckpt_file}")
		ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
		model.load_state_dict(ckpt["model"])
		print("Loaded checkpoint done.")

	if opt.fuse:
		print("\tFusing model...")
		model = fuse_model(model)

	if opt.trt:
		assert not opt.fuse, "TensorRT model does not support model fusing!"
		trt_file = opt.weights
		assert os.path.exists(trt_file), "TensorRT model is not found!"
		model.head.decode_in_inference = False
		decoder = model.head.decode_outputs
		print("Using TensorRT for inference")
	else:
		trt_file = None
		decoder  = None

	predictor = Predictor(
		model, exp, opt.class_names, trt_file, decoder,
		opt.device, opt.fp16, opt.legacy, opt.agnostic_nms
	)

	return predictor


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', '-c', type=str, default='yolox_s.pth', help='model ckpt path')
	parser.add_argument('--exp_file', '-f', type=str, default=None, help='experiment description file')
	parser.add_argument('--name', type=str, default='c041', help='Name of video')  # file/folder, 0 for webcam
	parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
	parser.add_argument('--roi_region', type=str, default='inference/roi.jpg', help='ignore region')  # output folder
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
	parser.add_argument('--device', default='gpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', default=True, help='class-agnostic NMS')
	parser.add_argument('--fp16', action='store_true', help='Adopting mix precision evaluating.')
	parser.add_argument('--fuse', action='store_true', help='Fuse conv and bn for testing.')
	parser.add_argument('--trt', action='store_true', help='Using TensorRT model for testing.')
	parser.add_argument('--legacy', action='store_true', help='To be compatible with older versions')

	opt = parser.parse_args()
	print(opt)

	predictor = create_predictor(opt)

	with torch.no_grad():
		detect(predictor, opt)
