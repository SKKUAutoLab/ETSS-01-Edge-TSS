import os
import cv2
import numpy as np
import threading
from tqdm import tqdm

def add_panel(image, label):
	h, w, _ = image.shape
	font_size = h // 400
	t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)[0]

	# NOTE: draw bounding box
	cv2.rectangle(image, (0, 0), (t_size[0], t_size[1]), (0, 97, 153), -1)
	cv2.putText(image, label, (0, t_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_size, [255, 255, 255], 2)
	return image

# INPUT_FILE1 = f'/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/aic22_mtmc/outputs/dets_debug/c044_filter.mp4'  # 2021
# INPUT_FILE2 = f'/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/aic22_mtmc/outputs/mots_debug/fairmot/c044_mots_feat_raw.mp4'  # 2022
# OUTPUT_FILE = f'/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/aic22_mtmc/outputs/dets_debug/c044_dets_mots_compare.mp4'

# ratio_w = 1
# ratio_h = 2


def merge_2_video(INPUT_FILE1, INPUT_FILE2, OUTPUT_FILE, is_horizontal, fps, video_name):
	reader1 = cv2.VideoCapture(INPUT_FILE1)
	reader2 = cv2.VideoCapture(INPUT_FILE2)
	frame_count = min(int(reader1.get(cv2.CAP_PROP_FRAME_COUNT)), int(reader2.get(cv2.CAP_PROP_FRAME_COUNT)))

	if is_horizontal:
		width = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH)) + int(reader2.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT))
	else:
		width = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT)) + int(reader2.get(cv2.CAP_PROP_FRAME_HEIGHT))
	writer = cv2.VideoWriter(OUTPUT_FILE,
							 cv2.VideoWriter_fourcc(*'mp4v'),
							 fps,  # fps
							 (width, height))  # resolution

	print(reader1.isOpened())
	print(reader2.isOpened())
	have_more_frame = True
	index_image = 0
	for _ in tqdm(range(frame_count), desc=f"Concatinate video {video_name}"):
		ret, frame1 = reader1.read()

		if not ret:
			break

		_, frame2 = reader2.read()
		# NOTE: resize
		# frame1 = cv2.resize(frame1, (width // ratio_h, height // ratio_h))
		# frame2 = cv2.resize(frame2, (width // ratio_h, height // ratio_h))

		# NOTE: add panel
		frame1 = add_panel(frame1, str(index_image))
		frame2 = add_panel(frame2, str(index_image))

		if is_horizontal:
			img = np.hstack((frame1, frame2))  # Horizontal concat
		else:
			img = np.vstack((frame1, frame2))  # Vertical concat
		writer.write(img)
		index_image += 1

	writer.release()
	reader1.release()
	reader2.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	# print(os.path.realpath(__file__))
	for video_name in ["c041", "c042", "c043", "c044", "c045", "c046"]:
	# for video_name in ["c041"]:
		is_horizontal = True
		fps = 4

		outputs_folder = "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/aic22_mtmc/outputs"
		INPUT_FILE1 = f'{outputs_folder}/dets_debug/yolov5x/{video_name}.mp4'  # 2021
		INPUT_FILE2 = f'{outputs_folder}/dets_debug/yolov5x/{video_name}_dets_filter.mp4'  # 2022
		OUTPUT_FILE = f'{outputs_folder}/dets_debug/yolov5x/{video_name}_compare.mp4'
		draw_1 = threading.Thread(target=merge_2_video, args=(INPUT_FILE1, INPUT_FILE2, OUTPUT_FILE, is_horizontal, fps, video_name))
		draw_1.start()
		# merge_2_video(
		# 	INPUT_FILE1=INPUT_FILE1,
		# 	INPUT_FILE2=INPUT_FILE2,
		# 	OUTPUT_FILE=OUTPUT_FILE,
		# 	is_horizontal=is_horizontal,
		# 	fps=fps
		# )

		outputs_folder = "/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/MLKit/projects/tss/data/aic22_mtmc/outputs"
		INPUT_FILE1 = f'{outputs_folder}/mots_debug/fairmot/{video_name}_mots_feat_raw.mp4'  # 2021
		INPUT_FILE2 = f'{outputs_folder}/mots_debug/fairmot/{video_name}_mots_feat_zone.mp4'  # 2022
		OUTPUT_FILE = f'{outputs_folder}/mots_debug/fairmot/{video_name}_mots_feat_zone_compare.mp4'
		draw_2 = threading.Thread(target=merge_2_video, args=(INPUT_FILE1, INPUT_FILE2, OUTPUT_FILE, is_horizontal, fps, video_name))
		draw_2.start()
		# merge_2_video(
		# 	INPUT_FILE1=INPUT_FILE1,
		# 	INPUT_FILE2=INPUT_FILE2,
		# 	OUTPUT_FILE=OUTPUT_FILE,
		# 	is_horizontal=is_horizontal,
		# 	fps=fps
		# )
		draw_1.join()
		draw_2.join()
