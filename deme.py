#coding=utf-8
import cv2
import numpy as np
import time
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from mvnc_ai_kit import AgeNet
from mvnc_ai_kit import GenderNet
from mvnc_ai_kit import TinyYolo

class Camera:
	def __init__(self, width=320, height=240):
		self.cap = cv2.VideoCapture(0)
		self.image= QImage()
		self.width = width
		self.height = height
		ret, frame = self.cap.read()
		frame = cv2.resize(frame, (self.width, self.height))
		self.h, self.w, self.bytesPerComponent = frame.shape
		self.bytesPerLine = self.bytesPerComponent * self.w

	def ReturnOneQPixmap(self):
		# get a frame
		ret, frame = self.cap.read()
		frame = cv2.resize(frame, (self.width, self.height))
		if ret:
			if frame.ndim == 3:
				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			elif frame.ndim == 2:
				rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		self.image = QImage(rgb.data, self.w, self.h, self.bytesPerLine, QImage.Format_RGB888)
		return QPixmap.fromImage(self.image)

	def ReturnOneFrame(self):
		# get a frame
		ret, frame = self.cap.read()
		frame = cv2.resize(frame, (self.width, self.height))
		if ret:
			if frame.ndim == 3:
				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				return rgb
			elif frame.ndim == 2:
				rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
				return rgb

	def DestroyCamera(self):
		cap.release()
		cv2.destroyAllWindows()

class VideoBox(QWidget):
	def __init__(self):
		QWidget.__init__(self)
		self.setWindowFlags(Qt.CustomizeWindowHint)
		# 初始化摄像头
		self.camera = Camera(320, 320)
		# 初始化Age，Gender，TinyYolo
		self.agenet = AgeNet.AgeNet()
		self.gendernet = GenderNet.GenderNet()
		self.tinyyolo = TinyYolo.TinyYolo()
		# 组件展示
		self.pictureLabel = QLabel()
		self.pictureLabel.setFixedSize(320, 240)
		self.pictureLabel.setObjectName("Ai Kit")
		self.init_image = QPixmap("./img/1.png").scaled(320, 240)
		self.pictureLabel.setPixmap(self.init_image)
		self.img = []
		# 选择下拉框
		self.combo = QComboBox()
		self.combo.addItem('Video')
		self.combo.addItem('AgeNet')
		self.combo.addItem('GenderNet')
		self.combo.addItem('TinyYolo')
		self.combo.addItem('Stop')
		self.combo.activated[str].connect(self.onActivatd)
		# 打印输出栏
		self.show_one = QLabel()
		self.show_two = QLabel()
		# 水平布局组件
		control_box = QHBoxLayout()
		control_box.setContentsMargins(0, 0, 0, 0)
		control_box.addWidget(self.combo)
		control_box.addWidget(self.show_one)
		control_box.addWidget(self.show_two)
		# 添加组件
		layout = QVBoxLayout()
		layout.addWidget(self.pictureLabel)
		layout.addLayout(control_box)
		self.setLayout(layout)
		# video timer 设置
		self.video_timer = VideoTimer()
		self.video_timer.timeSignal.signal[str].connect(self.showframe)
		# agenet timer 设置
		self.agenet_timer = VideoTimer(2)
		self.agenet_timer.timeSignal.signal[str].connect(self.runagenet)
		# gender timer 设置
		# self.gendernet_timer = VideoTimer(2)
		# self.gendernet_timer.timeSignal.signal[str].connect(self.rungendernet)

	def onActivatd(self, text):
		self.video_timer.start()
		if text == 'Video':
			print('Video')
		if text == 'AgeNet':
			self.agenet.prepare()
			self.agenet_timer.start()
		if text == 'GenderNet':
			print('GenderNet')		
		if text == 'Stop':
			self.video_timer.stop()
			quit()

	def showframe(self):
		self.pictureLabel.setPixmap(self.camera.ReturnOneQPixmap())

	def runagenet(self):
		res = self.agenet.Run_AgeNet(self.camera.ReturnOneFrame())
		self.show_one.setText(str(res[0]))
		self.show_two.setText(str(res[1]))

class Communicate(QObject):
	signal = pyqtSignal(str)

class VideoTimer(QThread):
	def __init__(self, frequent=20):
		QThread.__init__(self)
		self.stopped = False
		self.frequent = frequent
		self.timeSignal = Communicate()
		self.mutex = QMutex()

	def run(self):
		with QMutexLocker(self.mutex):
			self.stopped = False
		while True:
			if self.stopped:
				return
			self.timeSignal.signal.emit("1")
			time.sleep(1 / self.frequent)

	def stop(self):
		with QMutexLocker(self.mutex):
			self.stopped = True

	def is_stopped(self):
		with QMutexLocker(self.mutex):
			return self.stopped

	def set_fps(self, fps):
		self.frequent = fps

if __name__ == "__main__":
	app = QApplication(sys.argv)
	box = VideoBox()
	# box.showFullScreen()
	box.show()
	sys.exit(app.exec_())
