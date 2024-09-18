import sys
import numpy as np
import pyrealsense2 as rs
import cv2
from PySide6.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer

class Depth:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Depth 카메라 연결 시도
        try:
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline_profile = self.pipeline.start(self.config)
            self.use_depth_camera = True
        except:
            print("Depth 카메라를 사용할 수 없습니다. 웹캠을 사용합니다.")
            self.use_depth_camera = False
            self.cap = cv2.VideoCapture(0)  # 웹캠 사용

    def start_pipeline(self):
        if self.use_depth_camera:
            self.pipeline_profile = self.pipeline.start(self.config)
        else:
            if not self.cap.isOpened():
                raise Exception("웹캠을 열 수 없습니다.")

    def stop_pipeline(self):
        if self.use_depth_camera:
            self.pipeline.stop()
        else:
            if self.cap.isOpened():
                self.cap.release()

    def get_frame(self):
        if self.use_depth_camera:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            return depth_image, color_image
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None, None
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth_image = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            return depth_image, frame


# class Depth(QMainWindow):
#     def __init__(self):
#         super(Depth, self).__init__()

#         self.setWindowTitle("RealSense Depth Camera")
#         self.setGeometry(100, 100, 1280, 480)

#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)

#         self.layout = QVBoxLayout(self.central_widget)

#         self.label = QLabel()
#         self.layout.addWidget(self.label)

#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
#         self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#         self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#         self.pipeline_profile = None

#     def start_pipeline(self):
#         self.pipeline_profile = self.pipeline.start(self.config)

#     def stop_pipeline(self):
#         self.pipeline.stop()

#     def get_frame(self):
#         if self.pipeline_profile is None:
#             self.start_pipeline()
#         frames = self.pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             return None, None

#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())

#         return depth_image, color_image

#     def update_frame(self):
#         try:
#             depth_image, color_image = self.get_frame()
#             if depth_image is None or color_image is None:
#                 return

#             depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#             color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
#             images = np.hstack((color_image_rgb, depth_colormap))

#             height, width, channel = images.shape
#             bytes_per_line = 3 * width
#             qimg = QImage(images.data, width, height, bytes_per_line, QImage.Format_RGB888)
#             pixmap = QPixmap.fromImage(qimg)
#             self.label.setPixmap(pixmap)
#         except RuntimeError as e:
#             print(f"Runtime error: {e}")

#     def closeEvent(self, event):
#         self.stop_pipeline()
#         event.accept()

# def run_gui():
#     app = QApplication(sys.argv)
#     window = Depth()
#     window.start_pipeline()
#     window.show()
#     sys.exit(app.exec())

# if __name__ == "__main__":
#     run_gui()
