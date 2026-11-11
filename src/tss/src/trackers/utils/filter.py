import colorsys
import os
import sys
import re
import pickle
from operator import itemgetter, attrgetter

from tqdm import tqdm
import numpy as np
import cv2

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class FilterBoxes:

	# MARK: Magic Functions
	def __init__(
			self,
			camera_name: str,
			pickle_path: str,
			*args, **kwargs
	):
		self.camera_name     = camera_name
		self.pickle_path_in  = pickle_path

		self.feat_mode_dict  = None
		self.pickle_path_out = f"{os.path.splitext(self.pickle_path_in)[0]}_filter.pkl"

		self.load_pickle(pickle_path)
		self.add_area_bbox()

	# MARK: Configure

	# MARK: Method

	def add_area_bbox(self):
		"""Calculate area of bounding box"""
		area_max   = 0
		area_min   = sys.maxsize
		width_max  = 0
		height_max = 0

		if self.feat_mode_dict is not None:
			for key, value in self.feat_mode_dict.items():
				bbox      = value['bbox']
				bbox_area = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
				self.feat_mode_dict[key]['bbox_area'] = bbox_area
				area_max       = max(area_max, bbox_area)
				area_min       = min(area_min, bbox_area)
				width_max      = max(width_max, abs(bbox[2] - bbox[0]))
				height_max     = max(height_max, abs(bbox[3] - bbox[1]))

		# DEBUG:
		# print(f"{area_min=}")
		# print(f"{area_max=}")
		# print(f"{width_max=}")
		# print(f"{height_max=}")

	def filter_thres_bbox(self):
		key_list = list(self.feat_mode_dict.keys())
		key_list.sort()
		camera_id = int(re.sub("[^0-9]", "", self.camera_name))

		if camera_id in [41]:
			for key in tqdm(key_list, desc=f"Filter threshold of bounding box {camera_id}"):
				value = self.feat_mode_dict[key]
				area = value['bbox_area']
				bbox = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
				point_c_shapely = Point(point_c[0], point_c[1])

				if float(value['conf']) < 0.2:
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa may cai xe sai o trai duoi
				polygon_check = Polygon([(1, 370), (266, 310), (504, 529), (568, 959), (1, 959)])
				if float(value['conf']) < 0.3 and \
						polygon_check.contains(point_c_shapely):
					self.feat_mode_dict.pop(key, None)
					continue

		if camera_id in [42]:
			for key in tqdm(key_list, desc=f"Filter threshold of bounding box {camera_id}"):
				value = self.feat_mode_dict[key]
				area = value['bbox_area']
				bbox = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))

				if float(value['conf']) < 0.2:
					self.feat_mode_dict.pop(key, None)
					continue

		if camera_id in [43]:
			for key in tqdm(key_list, desc=f"Filter threshold of bounding box {camera_id}"):
				value = self.feat_mode_dict[key]
				area = value['bbox_area']
				bbox = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
				point_c_shapely = Point(point_c[0], point_c[1])

				if float(value['conf']) < 0.15:
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa may cai xe o trai tren
				polygon_check = Polygon([(1, 1), (1, 262), (481, 150)])
				if float(value['conf']) < 0.2 and \
						polygon_check.contains(point_c_shapely):
					self.feat_mode_dict.pop(key, None)
					continue

		if camera_id in [44]:
			conf_thres = 0.14
			for key in tqdm(key_list, desc=f"Filter threshold of bounding box {camera_id}"):
				value = self.feat_mode_dict[key]
				if float(value['conf']) < conf_thres:
					self.feat_mode_dict.pop(key, None)
					continue

		if camera_id in [45]:
			conf_thres = 0.1
			for key in tqdm(key_list, desc=f"Filter threshold of bounding box {camera_id}"):
				value = self.feat_mode_dict[key]
				if float(value['conf']) < conf_thres:
					self.feat_mode_dict.pop(key, None)
					continue

		if camera_id in [46]:
			for key in tqdm(key_list, desc=f"Filter threshold of bounding box {camera_id}"):
				value = self.feat_mode_dict[key]
				area = value['bbox_area']
				bbox = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
				point_c_shapely = Point(point_c[0], point_c[1])

				if float(value['conf']) < 0.2:
					self.feat_mode_dict.pop(key, None)
					continue

	def filter_area_bbox(self):
		key_list = list(self.feat_mode_dict.keys())
		key_list.sort()
		camera_id = int(re.sub("[^0-9]", "", self.camera_name))

		if camera_id in [41]:
			for key in tqdm(key_list, desc=f"Filter area of bounding box {camera_id}"):
				value = self.feat_mode_dict[key]
				area = value['bbox_area']
				bbox = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))

				if area < 550:
					self.feat_mode_dict.pop(key, None)
					continue

	def filter_size_bbox(self):
		key_list = list(self.feat_mode_dict.keys())
		key_list.sort()
		camera_id = int(re.sub("[^0-9]", "", self.camera_name))

		# if camera_id in [43]:
		# 	for key in tqdm(key_list, desc=f"Filter size of bounding box {camera_id}"):
		# 		value = self.feat_mode_dict[key]
		# 		area = value['bbox_area']
		# 		bbox = value['bbox']
		# 		point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
		# 		point_c_shapely = Point(point_c[0], point_c[1])
		# 		bbox_w = abs(bbox[2] - bbox[0])
		# 		bbox_h = abs(bbox[3] - bbox[1])
		#
		# 		# NOTE: xoa may cai den giao thong o tren, dia tren size
		# 		# den #1
		# 		polygon_retaurant = Polygon([(956, 50), (979, 58), (973, 96), (950, 92)])
		# 		if float(bbox_h) / float(bbox_w) > 1.6 and \
		# 				polygon_retaurant.contains(point_c_shapely):
		# 			self.feat_mode_dict.pop(key, None)
		# 			continue
		# 		# den #2
		# 		polygon_retaurant = Polygon([(1058, 78), (1083, 83), (1078, 124), (1049, 116)])
		# 		if float(bbox_h) / float(bbox_w) > 1.6 and \
		# 				polygon_retaurant.contains(point_c_shapely):
		# 			self.feat_mode_dict.pop(key, None)
		# 			continue
		# 		# den #3
		# 		polygon_retaurant = Polygon([(1149, 100), (1175, 108), (1168, 154), (1136, 141)])
		# 		if float(bbox_h) / float(bbox_w) > 1.6 and \
		# 				polygon_retaurant.contains(point_c_shapely):
		# 			self.feat_mode_dict.pop(key, None)
		# 			continue

	def filter_position_bbox(self):
		key_list = list(self.feat_mode_dict.keys())
		key_list.sort()
		camera_id = int(re.sub("[^0-9]", "", self.camera_name))

		if camera_id in [41]:
			for key in tqdm(key_list, desc=f"Filter position of bounding box {camera_id}"):
				value   = self.feat_mode_dict[key]
				area    = value['bbox_area']
				bbox    = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
				point_c_shapely = Point(point_c[0], point_c[1])

				# NOTE: xoa may cai bouding box o cai ven duong phai tren
				polygon_check = Polygon([(1169, 188), (1145, 199), (1141, 209), (1188, 238), (1208, 206)])
				if polygon_check.contains(point_c_shapely):
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa may cai xe ma bounding box chua qua cot dien o trai tren
				if (1 <= point_c[0] <= 35) and \
						(1 <= point_c[1] <= 200):
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa cai nap cong ngay trai giua
				if (225 <= point_c[0] <= 250) and \
						(275 <= point_c[1] <= 290):
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa cai nap cong ngay giua giao lo
				if (720 <= point_c[0] <= 760) and \
						(400 <= point_c[1] <= 430) and \
						area < 3000:
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa may cai xe trong nha hang
				# if (360 <= point_c[0] <= 716) and \
				# 		(5 <= point_c[1] <= 74):
				# 	self.feat_mode_dict.pop(key, None)
				# 	continue
				polygon_check = Polygon([(1, 1), (1, 30), (380, 111), (582, 83), (748, 77), (848, 1)])
				if polygon_check.contains(point_c_shapely):
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa may cai bounding box nho o goc phai duoi
				if (820 <= point_c[0]) and \
						(350 <= point_c[1]) and \
						area < 3000:
					self.feat_mode_dict.pop(key, None)
					continue

		if camera_id in [42]:
			for key in tqdm(key_list, desc=f"Filter position of bounding box {camera_id}"):
				value   = self.feat_mode_dict[key]
				area    = value['bbox_area']
				bbox    = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
				point_c_shapely = Point(point_c[0], point_c[1])

				# NOTE: xoa may cai xe trong nha hang
				polygon_retaurant = Polygon([(1, 1), (1, 142), (203, 141), (422, 109), (878, 60), (878, 1)])
				if polygon_retaurant.contains(point_c_shapely):
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa may cai xe trai tren
				if (141 <= point_c[0] <= 200) and \
						(110 <= point_c[1] <= 140):
					self.feat_mode_dict.pop(key, None)
					continue

		if camera_id in [43]:
			for key in tqdm(key_list, desc=f"Filter position of bounding box {camera_id}"):
				value   = self.feat_mode_dict[key]
				area    = value['bbox_area']
				bbox    = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
				point_c_shapely = Point(point_c[0], point_c[1])

				# NOTE: xoa may cai xe trong nha hang
				polygon_retaurant = Polygon([(63, 3), (63, 25), (277, 71), (392, 84), (392, 84), (1000, 8)])
				if polygon_retaurant.contains(point_c_shapely):
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa cai den giao thong trai tren
				# polygon_retaurant = Polygon([(1151, 114), (1167, 116), (1164, 148), (1145, 141)])
				# if polygon_retaurant.contains(point_c_shapely):
				# 	self.feat_mode_dict.pop(key, None)
				# 	continue

				# NOTE: xoa cai vong xuyen trai giua
				if (3 <= point_c[0] <= 70) and \
						(375 <= point_c[1] <= 420):
					self.feat_mode_dict.pop(key, None)
					continue

		if camera_id in [44]:
			for key in tqdm(key_list, desc=f"Filter position of bounding box {camera_id}"):
				value   = self.feat_mode_dict[key]
				area    = value['bbox_area']
				bbox    = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))

				if bbox[1] < 50:  # y_min < 50
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: remove bottom right bounding box
				if (930 <= point_c[0] <= 1140) and \
						(460 <= point_c[1] <= 600):
					self.feat_mode_dict.pop(key, None)
					continue

				if (460 <= point_c[1] <= 680) and \
						(1200 <= bbox[2] <= 1280):
					self.feat_mode_dict.pop(key, None)
					continue

				if (930 <= point_c[0]) and \
						(460 <= point_c[1]) and \
						(area < 3000):
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: remove traffic sign at top
				if (635 <= point_c[0] <= 680) and \
						(75 <= point_c[1] <= 88):
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa cai phai tren
				if (1235 <= point_c[0] <= 1280) and \
						(242 <= point_c[1] <= 290):
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa cai phai duoi cung
				if (1260 <= bbox[2]) and \
						(920 <= bbox[3]):
					self.feat_mode_dict.pop(key, None)
					continue

		if camera_id in [45]:
			for key in tqdm(key_list, desc=f"Filter position of bounding box {camera_id}"):
				value   = self.feat_mode_dict[key]
				area    = value['bbox_area']
				bbox    = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
				point_c_shapely = Point(point_c[0], point_c[1])

				# NOTE: xoa may cai xe trong bai do xe phai tren
				polygon_retaurant = Polygon([(1050, 12), (1068, 95), (1125, 116), (1280, 140), (1280, 12)])
				if polygon_retaurant.contains(point_c_shapely):
					self.feat_mode_dict.pop(key, None)
					continue

		if camera_id in [46]:
			for key in tqdm(key_list, desc=f"Filter position of bounding box {camera_id}"):
				value   = self.feat_mode_dict[key]
				area    = value['bbox_area']
				bbox    = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
				point_c_shapely = Point(point_c[0], point_c[1])

				# NOTE: xoa cai de chan duong o trai tren
				if (146 <= point_c[0] <= 153) and \
						(143 <= point_c[1] <= 153) and \
						area < 3000:
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa cai detection xai o phai giau
				if (1254 <= point_c[0] <= 1280) and \
						(228 <= point_c[1] <= 356):
					self.feat_mode_dict.pop(key, None)
					continue

				# NOTE: xoa may cai xe o phai tren
				polygon_check = Polygon([(1108, 1), (1108, 100), (1280, 150), (1280, 1)])
				if polygon_check.contains(point_c_shapely):
					self.feat_mode_dict.pop(key, None)
					continue

	def is_bbox_overlap_bbox(self, bbox_big, bbox_small):
		if bbox_big[0] < bbox_small[0] and \
				bbox_big[1] < bbox_small[1] and \
				bbox_big[2] > bbox_small[2] and \
				bbox_big[3] > bbox_small[3]:
				return True
		return False

	def filter_bbox_overlap_bbox(self):
		"""Remove bounding box cover bounding box"""
		key_list = list(self.feat_mode_dict.keys())
		key_list.sort()
		feat_mode_dict_temp = self.feat_mode_dict.copy()
		camera_id = int(re.sub("[^0-9]", "", self.camera_name))

		if camera_id in [43]:
			# NOTE: trai duoi, chi check nhung cai trai duoi
			polygon_check = Polygon([(1, 316), (297, 198), (573, 235), (1277, 402), (1280, 960), (1, 960)])

			for key in key_list:
				value   = self.feat_mode_dict[key]
				area    = value['bbox_area']
				bbox    = value['bbox']
				point_c = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
				point_c_shapely = Point(point_c[0], point_c[1])

				if not polygon_check.contains(point_c_shapely):
					feat_mode_dict_temp.pop(key, None)
					continue

			# Lay nhung cai bounding box con lai
			key_list = list(feat_mode_dict_temp.keys())
			key_list.sort()

			list_key_del = []
			for key in tqdm(key_list, desc=f"Filter overlap of bounding box {camera_id}"):
				value       = self.feat_mode_dict[key]
				bbox        = np.array(value['bbox'])
				point_c     = (bbox[0] + (abs(bbox[2] - bbox[0]) // 2), bbox[1] + (abs(bbox[3] - bbox[1]) // 2))
				frame_index = int(re.sub("[^0-9]", "", self.feat_mode_dict[key]['frame']))

				# NOTE: resize bounding box
				bbox[0] = max(0,    bbox[0] - 20)
				bbox[1] = max(0,    bbox[1] - 20)
				bbox[2] = min(1280, bbox[2] + 20)
				bbox[3] = min(960,  bbox[3] + 20)

				# New da bi del roi thi khoi check
				if key is list_key_del:
					continue

				# NOTE: check whether bbox cover any bbox_temp, Remove the "smaller"
				for key_tmp in key_list:

					if key is key_tmp:
						continue

					if frame_index == int(re.sub("[^0-9]", "", self.feat_mode_dict[key_tmp]['frame'])):
						bbox_temp = self.feat_mode_dict[key_tmp]['bbox']
						if self.is_bbox_overlap_bbox(bbox, bbox_temp):
							if bbox_temp not in list_key_del:
								list_key_del.append(key_tmp)
								break
					elif frame_index < int(re.sub("[^0-9]", "", self.feat_mode_dict[key_tmp]['frame'])):
						break

			print(f"Delete {len(list_key_del)}")
			for key_del in list_key_del:
				self.feat_mode_dict.pop(key_del, None)


	# MARK: Input/Output

	def load_pickle(self, pkl_path):
		"""load feature"""
		self.feat_mode_dict = pickle.load(open(pkl_path, 'rb'))

	def save_pickle(self):
		"""save result"""
		if self.feat_mode_dict is None:
			return
		pickle.dump(self.feat_mode_dict, open(self.pickle_path_out, 'wb'))

	def to_string(self):
		"""print example of pickle"""
		print(f"Length of pickle : {len(self.feat_mode_dict)}")
		print("-------- Example value --------")
		for key, value in self.feat_mode_dict.items():
			print(f"Key : {key} \nValue : {value}")
			break

	def to_string_length_pickle(self):
		print(f"Length of pickle : {len(self.feat_mode_dict)}")

	def get_list_data(self):
		results = []
		for image_name in list(self.feat_mode_dict.keys()):
			feat_dic = self.feat_mode_dict[image_name]
			# sub("[^0-9]", "", "!1a2;b3c?") to remove letter in string
			frame    = int(re.sub("[^0-9]", "", feat_dic['frame']))
			bbox     = np.array(feat_dic['bbox']).astype('int')
			score    = float(feat_dic['conf']) if 'conf' in feat_dic else None
			cls      = int(feat_dic['class']) if 'class' in feat_dic else 2  # 2 is class id of car in COCO dataset
			results.append([frame, bbox, score, cls])

		results = sorted(results, key=itemgetter(0))
		return results

	def draw_bbox(self, image, bbox, label, color):
		"""Draw bounding box"""
		x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
		t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
		cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
		cv2.rectangle(image, (x_min, int(y_min - t_size[1] - 4)), (int(x_min + t_size[0] + 3), y_min), color, -1)
		cv2.putText(image, label, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)
		return image

	def draw_video(self, video_in: str,	video_out: str):
		# NOTE: init in video
		vid_cap = cv2.VideoCapture(video_in)
		frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

		# NOTE: init out video
		fourcc = 'mp4v'  # output video codec
		fps = vid_cap.get(cv2.CAP_PROP_FPS)
		w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		print(f"Video size : {w=}, {h=}")
		vid_writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

		pbar = tqdm(total=frame_count, desc=f"Draw result {self.camera_name}")

		# NOTE: run
		results = self.get_list_data()
		frame_index  = -1
		result_index = 0
		while True:
			# reading from frame
			ret, image = vid_cap.read()
			frame_index = frame_index + 1

			if ret:
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

				# writing the extracted images
				vid_writer.write(image)

			else:
				break
			pbar.update(1)

		pbar.close()
		vid_cap.release()
		vid_writer.release()


# MARK: Utils


def create_unique_color_float(tag, hue_step=0.41):
	h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
	r, g, b = colorsys.hsv_to_rgb(h, 1., v)
	return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
	r, g, b = create_unique_color_float(tag, hue_step)
	return (int(255 * r), int(255 * g), int(255 * b))


if __name__ == "__main__":
	folder_feature = "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/aic22_mtmc/outputs/dets_feat_merge"
	total_crop = 0
	for cam_id in [41, 42, 43, 44, 45, 46]:
		pkl_path = f"{folder_feature}/c0{cam_id}/c0{cam_id}_dets_feat.pkl"
		filter_box = FilterBoxes(camera_name=str(f"c0{cam_id}"), pickle_path=pkl_path)
		# filter_box.pickle_path_out = f"/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/aic22_mtmc/outputs/dets_feat_merge/c0{cam_id}/c0{cam_id}_dets_feat_filter.pkl"
		# filter_box.to_string()
		filter_box.to_string_length_pickle()
		filter_box.filter_thres_bbox()
		filter_box.to_string_length_pickle()
		filter_box.filter_area_bbox()
		filter_box.to_string_length_pickle()
		filter_box.filter_size_bbox()
		filter_box.to_string_length_pickle()
		filter_box.filter_position_bbox()
		filter_box.to_string_length_pickle()
		filter_box.filter_bbox_overlap_bbox()
		filter_box.to_string_length_pickle()
		filter_box.draw_video(
			video_in=f"/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/aic22_mtmc/dataset_a/c0{cam_id}/vdo.avi",
			video_out=f"/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/aic22_mtmc/outputs/dets_debug/c0{cam_id}_filter.mp4"
		)
		filter_box.to_string()
		filter_box.save_pickle()
