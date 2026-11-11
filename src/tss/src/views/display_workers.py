from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from src.detectors.yolox_adaptor import YOLOX_Adapter

# region Detection process

class DetectionWorker(QObject):
	frame_ready = pyqtSignal(int , object)   # index, np.ndarray
	finished    = pyqtSignal()
	error       = pyqtSignal(str)

	def __init__(self, detector_cfgs: dict):
		super().__init__()
		self.detector_cfgs = detector_cfgs
		self._running = True

	@pyqtSlot()
	def run(self):
		try:
			generators = []
			for _, cfg in self.detector_cfgs.items():
				generators.append(YOLOX_Adapter(**cfg["detector"]).detect_source(**cfg["source"]))

			exhausted = set()

			while self._running and len(exhausted) < len(generators):
				for idx, gen in enumerate(generators):

					if not self._running:
						break
					if idx in exhausted:
						continue

					try:
						img = next(gen)
					except StopIteration:
						exhausted.add(idx)
						continue
					except Exception as e:
						self.error.emit(f"Detection generator {idx} failed: {e}")
						exhausted.add(idx)
						continue

					if img is None:
						exhausted.add(idx)
						continue

					# DEBUG:
					# logger.info(f"Detection generator {idx} produced a frame. Image info: {img.shape if isinstance(img, np.ndarray) else 'not an image'}")

					self.frame_ready.emit(idx, img)


			self.finished.emit()
		except Exception as e:
			self.error.emit(str(e))
			self.finished.emit()

	@pyqtSlot()
	def stop(self):
		self._running = False

#endregion
