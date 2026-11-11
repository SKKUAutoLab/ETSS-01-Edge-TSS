from __future__ import annotations

import os
import re
import sys
import time
from collections import deque
from copy import deepcopy
from enum import IntEnum, auto
from functools import partial
from timeit import default_timer as timer
from typing import Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger
from PyQt6 import uic
from PyQt6.QtCore import QProcess, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from src import configuration
from src.data.data_class.class_label import ClassLabels
from src.views.display_workers import DetectionWorker

APP_STYLESHEET = """
QMainWindow {
    background-color: #F5F5F7;
}
QWidget {
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
    color: #1D1D1F;
}
QPushButton {
    background-color: #007AFF;
    color: white;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: bold;
    border: none;
}
QPushButton:hover {
    background-color: #0056b3;
}
QPushButton:pressed {
    background-color: #004085;
}
QLabel {
    background-color: transparent;
}
QLabel#display_frame, QLabel#label_result {
    background-color: #FFFFFF;
    border: 1px solid #D2D2D7;
    border-radius: 8px;
}
QPushButton#button_image_extraction, QPushButton#button_detection, QPushButton#button_reidentification, QPushButton#button_single_camera_tracking, QPushButton#button_inter_camera_tracking, QPushButton#button_show_result_flag, QPushButton#button_back_up_3, QPushButton#button_back_up_4 {
    background: transparent;
    border: none;
    padding: 0px;
}
"""

# region Define videos, Hyperparameters, etc.

# Load class labels from JSON file
class_lables = ClassLabels.create_from_file(os.path.join(configuration.data_dir, 'tsd_suwon_19_46_20', 'class_labels.json'))

# Load yaml configuration for MTMC
config_19_suwon = yaml.safe_load(open(os.path.join(configuration.data_dir, 'tsd_suwon_19_46_20', 'configs', '19_SUWON.yaml')))
config_20_suwon = yaml.safe_load(open(os.path.join(configuration.data_dir, 'tsd_suwon_19_46_20', 'configs', '20_SUWON.yaml')))
config_46_suwon = yaml.safe_load(open(os.path.join(configuration.data_dir, 'tsd_suwon_19_46_20', 'configs', '46_SUWON.yaml')))

# NOTE: DETECTION config
# One YOLOX_Adapter per camera, built from the camera YAML's `detector` section; DetectionWorker runs its detect_source().

def yolox_detector_config(config: dict) -> dict:
	"""Keyword arguments of `YOLOX_Adapter` ("detector") and of its `detect_source` ("source") for one camera."""
	dataset_dir = os.path.join(configuration.data_dir, config["dataset"])
	return {
		"detector": {
			**config["detector"],
			"variant"     : os.path.join(configuration.tss_dir, config["detector"]["variant"]),         # "third_party/YOLOX/exps/example/custom/yolox_x_19_suwon.py"
			"weights"     : os.path.join(configuration.models_zoo_dir, config["detector"]["weights"]),  # "models_zoo/yolox/yolox_x_19_suwon/best_ckpt.pth"
			"class_labels": class_lables,
		},
		"source": {
			"source"     : os.path.join(dataset_dir, "temps", "images", config["name"]),  # "data/tsd_suwon_19_46_20/temps/images/19_SUWON"
			"name"       : config["name"],                                           # "19_SUWON"
			"output"     : os.path.join(dataset_dir, "temps"),                       # "data/tsd_suwon_19_46_20/temps"
			"roi_region" : os.path.join(dataset_dir, config["subset"], config["name"], f"{config['name']}_roi.jpg"),  # "data/tsd_suwon_19_46_20/clearsky_1764390903/19_SUWON/19_SUWON_roi.jpg"
			"config_main": config,
		},
	}


detector_configs = {  # the order of the display panes
	"19_SUWON": yolox_detector_config(config_19_suwon),
	"46_SUWON": yolox_detector_config(config_46_suwon),
	"20_SUWON": yolox_detector_config(config_20_suwon),
}

# NOTE: MTMC information
mtmc_viz = [
	os.path.join(configuration.data_dir, config_19_suwon["dataset"], 'temps', 'mtmc_debug', '19_SUWON.mp4'),
	os.path.join(configuration.data_dir, config_46_suwon["dataset"], 'temps', 'mtmc_debug', '46_SUWON.mp4'),
	os.path.join(configuration.data_dir, config_20_suwon["dataset"], 'temps', 'mtmc_debug', '20_SUWON.mp4'),
]

video_fps = 30

# NOTE: DISPLAY information

images_external_show_result = [None for i in range(len(detector_configs))]

class MTMC_PROCESS(IntEnum):
	"""
	This class represents the different classes of objects that can be detected by the AI system.
	Each class is represented by an integer value.
	"""
	IDLE                   = 0
	IMAGE_EXTRATION        = auto()
	DETECTION              = auto()
	REIDENTIFICATION       = auto()
	SINGLE_CAMERA_TRACKING = auto()
	INTER_CAMERA_TRACKING  = auto()
	SHOW_RESULT            = auto()
	RUN_ALL                = auto()

#endregion


# A regular expression, to extract the % complete.
progress_re = re.compile("Progress complete: (\d+)%")

def simple_percent_parser(output):
	"""
	Matches lines using the progress_re regex,
	returning a single integer for the % progress.
	"""
	m = progress_re.search(output)
	if m:
		pc_complete = m.group(1)
		return int(pc_complete)


class DisplayThread(QThread):
	"""This class represents a thread that continuously updates the display frames with images from the specified paths."""

	# MARK: Magic Functions
	updateFrames = pyqtSignal(dict)

	def __init__(
		self,
		display_width : int | None = 320,
		display_height: int | None = 180,
		display_num   : int | None = 3,
		display_fps   : int | None = 15,
		*args, **kwargs
	):
		super().__init__()
		dir_path = os.path.dirname(os.path.realpath(__file__))

		self.run_flag        = True
		self.display_width   = display_width
		self.display_height  = display_height
		self.display_num     = display_num
		self.display_fps     = display_fps
		self.display_paths   = [os.path.join(dir_path, f"display_{i + 1}.jpg") for i in range(self.display_num)]
		self.capture_videos  = [None for i in range(self.display_num)]
		self.queue           = deque(maxlen=3)

		self.set_external_image_none()

	# MARK: Operation function

	def run(self):
		self.render_displays()

	def stop(self):
		"""Sets run flag to False and waits for thread to finish"""
		self.run_flag = False
		self.wait()

	@staticmethod
	def is_stable_image(path: str) -> bool:
		"""Return True when the Image file appears fully written and readable."""
		try:
			size1 = os.path.getsize(path)
			mtime1 = os.path.getmtime(path)

			time.sleep(0.005)

			size2 = os.path.getsize(path)
			mtime2 = os.path.getmtime(path)

			if size1 != size2 or mtime1 != mtime2:
				return False

			with open(path, 'rb') as f:
				tail = f.read()[-2:]

			return tail == b'\xff\xd9'
		except OSError:
			return False

	@pyqtSlot(int, object)
	def set_external_image(self, index: int, image: np.ndarray):
		"""Set an external image for a specific display"""
		if 0 <= index < self.display_num:
			images_external_show_result[index] = image
		# DEBUG:
		# logger.info(f"Set external image at index {index}.")


	def set_external_image_none(self):
		global images_external_show_result
		images_external_show_result = [None for _ in range(self.display_num)]

	def render_displays(self):
		blank_img    = np.zeros((self.display_height, self.display_width, 3), np.uint8)
		img_list     = [blank_img for _ in range(self.display_num)]
		img_list_pre = deepcopy(img_list)  # all images is black

		# overite all display with the black image
		for img_path, img in zip(self.display_paths, img_list):
			cv2.imwrite(img_path, img)

		is_finish_show_result = None  
		while self.run_flag:
			for i, (img_path, cap) in enumerate(zip(self.display_paths, self.capture_videos)):
				# Init value
				img = None

				# NOTE: Check external image first (highest priority)
				if images_external_show_result[i] is not None:
					img = images_external_show_result[i]

				# NOTE: Then try video
				# read video first
				if cap is not None:
					ret, img = cap.read()
					if not ret:
						img = None
						self.capture_videos[i].release()
						self.capture_videos[i] = None
					else:
						is_finish_show_result = False

				# NOTE: if no image load from video
				# if img is None and os.path.exists(img_path) and self.is_stable_image(img_path):
				if img is None and os.path.exists(img_path):
					img = cv2.imread(img_path, cv2.IMREAD_COLOR)

				# NOTE: if there is an image, update the display, otherwise keep the previous image (which is black at the beginning)
				if img is not None:
					img_list[i] = img.copy()
				else:
					img_list[i] = img_list_pre[i].copy() # if there is no loaded image, we can load all image which are black

			# check finish capture
			if is_finish_show_result is not None:
				if all(cap is None for cap in self.capture_videos):
					is_finish_show_result = True
				else:
					is_finish_show_result = False

			img_list_pre = deepcopy(img_list)
			self.updateFrames.emit({
				"display_imgs"          : img_list,
				"is_finish_show_result" : is_finish_show_result
			})
			# Reset finish show for the next show
			if is_finish_show_result:
				is_finish_show_result = None

			time.sleep(1 / self.display_fps)
			# self.msleep(self.display_fps)


class MTMCMainWindow(QMainWindow):

	# MARK: Magic Functions

	def __init__(
			self,
			ui_path    : Optional[str] = "mtmc.ui",
			script_path: Optional[str] = "runs",
			*args, **kwargs
	):
		super(MTMCMainWindow, self).__init__()
		# Init
		self.run_process           = MTMC_PROCESS.IDLE
		self.thread_displays       = None
		self.thread_video_function = None
		self.qprocess              = None
		self.script_path           = script_path
		self.process_name          = ""

		self.process_current       = None
		self.process_main          = None


		self.process_run_all_start_time = timer()

		uic.loadUi(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.basename(ui_path)), self)
		# uic.loadUi(ui_path, self)
		self.setStyleSheet(APP_STYLESHEET)

		# Build layout
		self.build_layout()
		self.create_display_thread()

		# create thread
		self.detection_thread = None
		self.detection_worker = None

	def build_layout(self):
		# Build Message
		self.text_message.setReadOnly(True)
		self.text_message.setMaximumBlockCount(1000)

		# Build button
		self.button_image_extraction.pressed.connect(partial(self.button_released, MTMC_PROCESS.IMAGE_EXTRATION))
		self.button_detection.pressed.connect(partial(self.button_released, MTMC_PROCESS.DETECTION))
		self.button_reidentification.pressed.connect(partial(self.button_released, MTMC_PROCESS.REIDENTIFICATION))
		self.button_single_camera_tracking.pressed.connect(partial(self.button_released, MTMC_PROCESS.SINGLE_CAMERA_TRACKING))
		self.button_inter_camera_tracking.pressed.connect(partial(self.button_released, MTMC_PROCESS.INTER_CAMERA_TRACKING))
		self.button_show_result.pressed.connect(partial(self.button_released, MTMC_PROCESS.SHOW_RESULT))
		self.button_run_all.pressed.connect(partial(self.button_released, MTMC_PROCESS.RUN_ALL))

	# MARK: Operation handle

	def message(self, msg, color: Optional[QColor] = QColor('black')):
		# List color: https://doc.qt.io/qtforpython-5/PySide2/QtGui/QColorConstants.html
		# Check message:
		# Put near other regex declarations
		error_word_re = re.compile(r"\berror\b", re.IGNORECASE)
		# Remove the word "error" in any letter case.
		msg = error_word_re.sub("", str(msg))

		color_format = self.text_message.currentCharFormat()
		color_format.setForeground(color)
		self.text_message.setCurrentCharFormat(color_format)
		self.text_message.appendPlainText(msg)

	def button_released(self, button_type: MTMC_PROCESS):
		if self.qprocess is not None:
			return

		process_map = {
			MTMC_PROCESS.RUN_ALL: ("Run All Tasks", "run_track_mtmc_cert.sh", QColor('black'), True,),
			MTMC_PROCESS.IMAGE_EXTRATION: ("Image Extraction", "run_track_mtmc_button_image_extraction.sh", QColor('black'), True,),
			MTMC_PROCESS.DETECTION: ("Detection", "run_track_mtmc_button_detection.sh", QColor('black'), True,),
			MTMC_PROCESS.REIDENTIFICATION: ("ReIdentification", "run_track_mtmc_button_reidentification.sh", QColor('black'), True,),
			MTMC_PROCESS.SINGLE_CAMERA_TRACKING: ("Single Camera Tracking", "run_track_mtmc_button_single_camera_tracking.sh", QColor('black'), True,),
			MTMC_PROCESS.INTER_CAMERA_TRACKING: ("Inter Camera Tracking","run_track_mtmc_button_inter_camera_tracking.sh", QColor('black'),	True,),
			MTMC_PROCESS.SHOW_RESULT: ("Show Result", None, QColor('green'), True, ),
		}

		if button_type not in process_map:
			return

		self.process_current = button_type
		if self.process_main is None:
			self.process_main = button_type

		process_name, script_name, color, needs_timer = process_map[button_type]
		self.process_name    = process_name
		if needs_timer:
			self.process_run_all_start_time = timer()

		self.message(f"Executing process - {process_name}", color)

		if button_type is MTMC_PROCESS.RUN_ALL:
			# start from extract image
			self.button_run_all.setEnabled(False)
			self.button_released(MTMC_PROCESS.IMAGE_EXTRATION)
		elif button_type is MTMC_PROCESS.SHOW_RESULT:
			self.qprocess = object()
			self.set_videos_show_result()
		elif button_type is MTMC_PROCESS.DETECTION:
			self.qprocess = object()  # optional running guard, like SHOW_RESULT
			self.run_detection_process()
		else:
			self.create_qprocess("bash", [f"{self.script_path}/{script_name}"])

	def create_qprocess(self, command_script, option_script):
		if self.qprocess is not None:
			return

		# Create process
		self.qprocess = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
		self.qprocess.readyReadStandardOutput.connect(self.handle_stdout)
		self.qprocess.readyReadStandardError.connect(self.handle_stderr)
		self.qprocess.stateChanged.connect(self.handle_state)
		self.qprocess.finished.connect(self.process_finished)  # Clean up once complete.
		self.qprocess.start(command_script, option_script)

	def handle_stderr(self):
		data = self.qprocess.readAllStandardError()
		stderr = bytes(data).decode("utf8")

		# print in "text_message" or terminal
		# self.message(stderr)
		if "%" in stderr:
			print(f"\r{stderr}", end="")
		else:
			print(stderr)

	def handle_stdout(self):
		data = self.qprocess.readAllStandardOutput()
		stdout = bytes(data).decode("utf8")

		# print in "text_message" or terminal
		# self.message(stdout)
		print(stdout)

		# Show progress
		if "PROCESS EXTRACTION" in stdout:
			self.change_button_display(MTMC_PROCESS.IMAGE_EXTRATION)
		elif "PROCESS DETECTION" in stdout:
			self.change_button_display(MTMC_PROCESS.DETECTION)
		elif "PROCESS REIDENTIFICATION" in stdout:
			self.change_button_display(MTMC_PROCESS.REIDENTIFICATION)
		elif "PROCESS SINGLE CAMERA TRACKING" in stdout:
			self.change_button_display(MTMC_PROCESS.SINGLE_CAMERA_TRACKING)
		elif "PROCESS INTER CAMERA TRACKING" in stdout:
			self.change_button_display(MTMC_PROCESS.INTER_CAMERA_TRACKING)

	def handle_state(self, state):
		states = {
			QProcess.ProcessState.NotRunning: 'Not running',
			QProcess.ProcessState.Starting  : 'Starting',
			QProcess.ProcessState.Running   : 'Running',
		}
		state_name = states[state]
		self.message(f"State changed: {state_name} - {self.process_name}", QColor('green'))

	def process_finished(self):
		self.message(f"Process finished - {self.process_name}.", QColor('green'))

		if self.qprocess is not None:
			del self.qprocess
		self.qprocess = None

		# Stop timer
		self.process_run_all_stop_time = timer()
		total_process_time = abs(self.process_run_all_start_time - self.process_run_all_stop_time)
		self.change_button_display(None)

		# Update the state of process
		if self.process_main is not None:
			if self.process_main is not MTMC_PROCESS.RUN_ALL:
				self.process_current = None
				self.process_main    = None
			elif self.process_main is MTMC_PROCESS.RUN_ALL:
				# DEBUG:
				logger.info(f"FINISH {self.process_main=} -- {self.process_current=}")
				if self.process_current is MTMC_PROCESS.SHOW_RESULT:
					self.message(f"<<FINISH>>", QColor('darkblue'))
					# self.message(f"Total run all processing time: {total_process_time} seconds.", QColor('darkblue'))
					self.process_current = None
					self.process_main    = None
					self.button_run_all.setEnabled(True)
					self.button_show_result.setEnabled(True)
				else:
					if self.process_current is MTMC_PROCESS.IMAGE_EXTRATION:
						button_type = MTMC_PROCESS.DETECTION
						self.change_button_display(MTMC_PROCESS.DETECTION)
					elif self.process_current is MTMC_PROCESS.DETECTION:
						button_type = MTMC_PROCESS.REIDENTIFICATION
						self.change_button_display(MTMC_PROCESS.REIDENTIFICATION)
					elif self.process_current is MTMC_PROCESS.REIDENTIFICATION:
						button_type = MTMC_PROCESS.SINGLE_CAMERA_TRACKING
						self.change_button_display(MTMC_PROCESS.SINGLE_CAMERA_TRACKING)
					elif self.process_current is MTMC_PROCESS.SINGLE_CAMERA_TRACKING:
						button_type = MTMC_PROCESS.INTER_CAMERA_TRACKING
						self.change_button_display(MTMC_PROCESS.INTER_CAMERA_TRACKING)
					elif self.process_current is MTMC_PROCESS.INTER_CAMERA_TRACKING:
						button_type = MTMC_PROCESS.SHOW_RESULT
						self.change_button_display(MTMC_PROCESS.SHOW_RESULT)
					else:
						button_type = None
					self.button_released(button_type)

	def change_button_display(self, task: MTMC_PROCESS):
		# Reset all task buttons.
		task_buttons = {
			MTMC_PROCESS.IMAGE_EXTRATION       : self.button_image_extraction,
			MTMC_PROCESS.DETECTION             : self.button_detection,
			MTMC_PROCESS.REIDENTIFICATION      : self.button_reidentification,
			MTMC_PROCESS.SINGLE_CAMERA_TRACKING: self.button_single_camera_tracking,
			MTMC_PROCESS.INTER_CAMERA_TRACKING : self.button_inter_camera_tracking,
			MTMC_PROCESS.SHOW_RESULT           : self.button_show_result_flag,
		}
		for button in task_buttons.values():
			button.setStyleSheet("background-color: none")

		if task is None:
			self.label_process.setText("Processed: 6/6")
			return

		task_progress = {
			MTMC_PROCESS.IMAGE_EXTRATION       : "Processing: 1/6",
			MTMC_PROCESS.DETECTION             : "Processing: 2/6",
			MTMC_PROCESS.REIDENTIFICATION      : "Processing: 3/6",
			MTMC_PROCESS.SINGLE_CAMERA_TRACKING: "Processing: 4/6",
			MTMC_PROCESS.INTER_CAMERA_TRACKING : "Processing: 5/6",
			MTMC_PROCESS.SHOW_RESULT           : "Processing: 6/6",
		}

		# DEBUG:
		logger.info(f"Changing button display for task: {task.name}")

		button = task_buttons.get(task)
		if button is not None:
			button.setStyleSheet("background-color: green")
			self.label_process.setText(task_progress[task])


	# MARK: Display process

	def create_display_thread(self):
		# create the video capture thread
		self.thread_displays = DisplayThread(display_fps=video_fps)
		# connect its signal to the update_image slot
		self.thread_displays.updateFrames.connect(self.update_displays)
		# start thread
		self.start_display_process()

	def start_display_process(self):
		print("Starting...")
		# start the thread
		if self.thread_displays is not None:
			self.thread_displays.start()
		print("Running...")

	def stop_display_process(self):
		# stop the thread
		if self.thread_displays is not None:
			self.thread_displays.stop()
		self.thread_displays = None
		# Give time for the thread to finish
		time.sleep(2)
		print("\nStopped...")

	def convert_cvimg_to_qtimg(self, cv_img, display_frame: QLabel):
		# Guard against zero-sized widgets during startup/layout transitions.
		width  = max(display_frame.width(), 1)
		height = max(display_frame.height(), 1)

		# Skip resize if the frame already matches target size.
		if cv_img.shape[1] != width or cv_img.shape[0] != height:
			cv_img = cv2.resize(cv_img, (width, height), interpolation=cv2.INTER_AREA)

		# Detach from numpy memory so Qt doesn't depend on external buffer lifetime.
		qimg = QImage(
			cv_img.data,
			cv_img.shape[1],
			cv_img.shape[0],
			cv_img.strides[0],
			QImage.Format.Format_BGR888
		).copy()

		return QPixmap.fromImage(qimg)

	def update_displays(self, frames: Union[dict, None] = None):
		# Update displays
		if frames is not None:

			# check finish show result:
			if frames["is_finish_show_result"] is not None:
				if frames["is_finish_show_result"]:
					self.process_finished()

			imgs = frames["display_imgs"]

			# update display_1
			if len(imgs) > 0:
				qt_img = self.convert_cvimg_to_qtimg(imgs[0], self.display_frame_1).copy()
				self.display_frame_1.setPixmap(qt_img)

			# update display_2
			if len(imgs) > 1:
				qt_img = self.convert_cvimg_to_qtimg(imgs[1], self.display_frame_2).copy()
				self.display_frame_2.setPixmap(qt_img)

			# update display_3
			if len(imgs) > 2:
				qt_img = self.convert_cvimg_to_qtimg(imgs[2], self.display_frame_3).copy()
				self.display_frame_3.setPixmap(qt_img)

			# # update display_4
			# if len(imgs) > 3:
			# 	qt_img = self.convert_cvimg_to_qtimg(imgs[3], self.display_frame_4).copy()
			# 	self.display_frame_4.setPixmap(qt_img)
			#
			# # update display_5
			# if len(imgs) > 4:
			# 	qt_img = self.convert_cvimg_to_qtimg(imgs[4], self.display_frame_5).copy()
			# 	self.display_frame_5.setPixmap(qt_img)
			#
			# # update display_6
			# if len(imgs) > 5:
			# 	qt_img = self.convert_cvimg_to_qtimg(imgs[5], self.display_frame_6).copy()
			# 	self.display_frame_6.setPixmap(qt_img)

	def set_videos_show_result(self):
		# Set video to show result
		self.thread_displays.capture_videos = [cv2.VideoCapture(viz) for viz in mtmc_viz]

	# MARK: Process call

	def _on_detection_finished(self):
		self.message("Detection finished.", QColor("green"))

	def _clear_detection_refs(self):
		self.detection_worker = None
		self.detection_thread = None

	def run_detection_process(self):
		if self.detection_thread is not None:
			return  # already running

		self.detection_thread = QThread(self)
		self.detection_worker = DetectionWorker(detector_configs)
		self.detection_worker.moveToThread(self.detection_thread)

		self.detection_thread.started.connect(self.detection_worker.run)
		self.detection_worker.frame_ready.connect(self.thread_displays.set_external_image)
		self.detection_worker.error.connect(lambda msg: self.message(msg, QColor("red")))
		self.detection_worker.finished.connect(self._on_detection_finished)

		# lifecycle cleanup
		self.detection_worker.finished.connect(self.detection_thread.quit)
		self.detection_thread.finished.connect(self.detection_worker.deleteLater)
		self.detection_thread.finished.connect(self.detection_thread.deleteLater)
		self.detection_thread.finished.connect(self._clear_detection_refs)
		self.detection_thread.finished.connect(self.process_finished)

		self.detection_thread.start()


def main():
	app = QApplication(sys.argv)

	w = MTMCMainWindow()
	w.show()

	app.exec()


if __name__ == "__main__":
	main()
