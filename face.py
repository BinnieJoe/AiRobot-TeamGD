# import cv2
# import tkinter as tk
# from tkinter import simpledialog
# from PIL import Image, ImageTk, ImageDraw, ImageFont
# import os
# import numpy as np
# import face_recognition
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle
# from concurrent.futures import ThreadPoolExecutor
# import pyrealsense2 as rs

import cv2
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import numpy as np
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from concurrent.futures import ThreadPoolExecutor
import pyttsx3  # 음성 출력 라이브러리 추가
from queue import Queue
import time  # 시간 측정용 라이브러리 추가
from threading import Thread  # threading 모듈에서 Thread 가져오기

class VideoCaptureApp:
    def __init__(self, window, window_title):
        self.window = window  # Tkinter 창 설정
        self.window.title(window_title)  # 창 제목 설정

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0)  # 기본 웹캠을 사용하여 비디오 캡처 객체 생성

        self.canvas = tk.Canvas(window)  # Tkinter 캔버스 생성
        self.canvas.pack(fill=tk.BOTH, expand=True)  # 캔버스를 창에 배치 및 크기 조절 가능하도록 설정

        # DNN 모델 로드 (얼굴 감지용)
        self.net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')  # DNN 모델 로드

        self.delay = 1  # 업데이트 딜레이 설정
        self.frame = None  # 현재 프레임을 저장할 변수
        self.capture_faces = False  # 얼굴 캡처 여부를 저장할 변수
        self.captured_faces = []  # 캡처된 얼굴을 저장할 리스트
        self.embeddings = {}  # 얼굴 임베딩을 저장할 딕셔너리
        self.data_folder = "face_image"  # 얼굴 이미지 저장 폴더
        self.embeddings_file = "embeddings.pkl"  # 얼굴 임베딩 파일
        self.recognized_faces = {}  # 인식된 얼굴과 인식된 시간을 저장하는 딕셔너리
        self.recalculate_embeddings()  # 임베딩 재계산

        # 음성 엔진 초기화
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # 속도 조절

        # 음성 출력 큐
        self.tts_queue = Queue()
        self.tts_thread = Thread(target=self.process_tts_queue)
        self.tts_thread.daemon = True
        self.tts_thread.start()

        self.update()  # 업데이트 함수 호출

        # 스페이스바를 눌렀을 때 start_capture를 호출
        self.window.bind("<space>", self.start_capture)

        # 창의 크기 변경 이벤트에 리사이즈 함수를 바인딩
        self.window.bind("<Configure>", self.resize_image)

        self.window.mainloop()  # Tkinter 이벤트 루프 시작

    def update(self):
        ret, frame = self.cap.read()  # 웹캠에서 프레임 가져오기
        if not ret:  # 프레임을 가져오지 못한 경우
            return

        self.frame = frame  # 현재 프레임 저장
        (h, w) = frame.shape[:2]  # 프레임의 높이와 너비 가져오기
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))  # DNN 입력 블롭 생성

        # DNN 모델을 사용하여 얼굴 감지
        self.net.setInput(blob)  # DNN 모델 입력 설정
        detections = self.net.forward()  # 얼굴 감지 실행

        for i in range(0, detections.shape[2]):  # 감지된 얼굴 수만큼 반복
            confidence = detections[0, 0, i, 2]  # 감지 신뢰도 가져오기

            if confidence > 0.5:  # 신뢰도가 0.5 이상인 경우
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # 얼굴 위치 박스 가져오기
                (startX, startY, endX, endY) = box.astype("int")  # 정수로 변환
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)  # 얼굴 위치에 사각형 그리기

                face = frame[startY:endY, startX:endX]  # 얼굴 이미지 추출
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 얼굴 이미지를 RGB로 변환
                embedding = self.get_face_embedding(face_rgb)  # 얼굴 임베딩 생성
                if embedding is not None:  # 임베딩이 생성된 경우
                    best_match = self.find_best_match(embedding)  # 가장 잘 맞는 얼굴 찾기
                    if best_match:  # 잘 맞는 얼굴이 있는 경우
                        label = os.path.basename(best_match).split('_')[0]  # 얼굴 레이블 추출
                        frame = self.draw_label(frame, label, (startX, startY - 20), (255, 0, 0))  # 레이블 그리기

                        # 현재 시간
                        current_time = time.time()
                        # 이전에 인식된 시간을 확인하고, 5초가 지나지 않았으면 음성 출력을 하지 않음
                        if label not in self.recognized_faces or current_time - self.recognized_faces[label] > 5:  # 5초마다 반복 인식 허용
                            self.recognized_faces[label] = current_time  # 인식된 시간 업데이트
                            # 음성 안내를 큐에 추가
                            message = f"{label}씨가 앞에 있습니다."
                            self.tts_queue.put(message)
                            print(message)  # 터미널에 메시지 출력

                # 얼굴 캡처 중이면 얼굴을 저장
                if self.capture_faces and len(self.captured_faces) < 5:  # 캡처된 얼굴 수가 5 미만인 경우
                    self.captured_faces.append(face)  # 얼굴 리스트에 추가

        # 현재 창 크기를 얻어와서 이미지 리사이즈
        self.resized_image = self.resize_frame(frame)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.resized_image))  # 리사이즈된 프레임을 Tkinter 이미지로 변환
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)  # 캔버스에 이미지 그리기

        # 다음 업데이트를 예약
        self.window.after(self.delay, self.update)  # 일정 시간 후에 다시 update 함수 호출

    def resize_image(self, event):
        # 윈도우 크기가 변경될 때 호출되는 함수
        if self.frame is not None:
            self.resized_image = self.resize_frame(self.frame)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.resized_image))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def resize_frame(self, frame):
        # 현재 창 크기에 맞춰 이미지를 리사이즈
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((canvas_width, canvas_height), Image.LANCZOS)  # ANTIALIAS를 LANCZOS로 변경
        return np.array(image)

    def process_tts_queue(self):
        while True:
            text = self.tts_queue.get()  # 큐에서 텍스트 가져오기
            if text is None:
                break
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            self.tts_queue.task_done()  # 작업 완료 표시

    def speak(self, text):
        """텍스트를 음성으로 출력합니다."""
        self.tts_queue.put(text)

    def start_capture(self, event):
        self.capture_faces = True  # 얼굴 캡처 시작
        self.captured_faces = []  # 캡처된 얼굴 리스트 초기화
        self.window.after(100, self.check_capture_complete)  # 일정 시간 후에 얼굴 캡처 완료 확인

    def check_capture_complete(self):
        if len(self.captured_faces) >= 5:  # 캡처된 얼굴 수가 5 이상인 경우
            self.capture_faces = False  # 얼굴 캡처 중지
            self.prompt_for_file_name()  # 파일 이름 입력 요청
        else:
            self.window.after(100, self.check_capture_complete)  # 일정 시간 후에 다시 얼굴 캡처 완료 확인

    def prompt_for_file_name(self):
        file_name = simpledialog.askstring("Input", "Enter the file name:")  # 파일 이름 입력 대화상자 표시
        if file_name:  # 파일 이름이 입력된 경우
            self.save_faces(file_name)  # 얼굴 저장

    def save_faces(self, file_name):
        folder_path = os.path.join(self.data_folder, file_name)  # 새로운 폴더 경로 생성
        if not os.path.exists(folder_path):  # 새로운 폴더가 없는 경우
            os.makedirs(folder_path)  # 폴더 생성

        new_embeddings = {}  # 새로운 얼굴 임베딩 딕셔너리 생성
        for idx, face in enumerate(self.captured_faces):  # 캡처된 얼굴 수만큼 반복
            face_filename = f"{folder_path}/{file_name}_{idx + 1}.jpg"  # 얼굴 파일 이름 생성
            face_filename = face_filename.encode('utf-8').decode('utf-8')  # 파일 이름 인코딩 처리
            success, encoded_image = cv2.imencode('.jpg', face)  # 얼굴 이미지를 JPEG 형식으로 인코딩
            if success:  # 인코딩 성공한 경우
                with open(face_filename, 'wb') as f:  # 파일 열기
                    f.write(encoded_image.tobytes())  # 인코딩된 이미지 저장
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 얼굴 이미지를 RGB로 변환
                new_embedding = self.get_face_embedding(face_rgb)  # 새로운 얼굴 임베딩 생성
                if new_embedding is not None:  # 임베딩이 생성된 경우
                    new_embeddings[face_filename] = new_embedding  # 새로운 임베딩 딕셔너리에 추가

        self.embeddings.update(new_embeddings)  # 기존 임베딩에 새로운 임베딩 추가
        self.save_embeddings()  # 임베딩 저장
        print("New faces saved and embeddings updated.")  # 로그 출력

    def get_face_embedding(self, image):
        face_locations = face_recognition.face_locations(image)  # 얼굴 위치 찾기
        if len(face_locations) > 0:  # 얼굴이 하나 이상인 경우
            face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)  # 얼굴 임베딩 생성
            if face_encodings:  # 임베딩이 생성된 경우
                return face_encodings[0]  # 첫 번째 임베딩 반환
        return None  # 임베딩이 없는 경우 None 반환

    def find_best_match(self, embedding):
        best_match = None  # 가장 잘 맞는 얼굴 변수 초기화
        best_similarity = 0.0  # 최고 유사도 변수 초기화

        with ThreadPoolExecutor() as executor:  # 스레드 풀 생성
            futures = [executor.submit(self.compare_embedding, embedding, name, emb)  # 임베딩 비교 작업 추가
                       for name, emb in self.embeddings.items()]  # 모든 임베딩에 대해 비교 작업 생성

            for future in futures:  # 각 비교 작업 결과 가져오기
                match, similarity = future.result()  # 비교 결과
                if similarity > best_similarity:  # 유사도가 최고 유사도보다 큰 경우
                    best_similarity = similarity  # 최고 유사도 갱신
                    best_match = match  # 가장 잘 맞는 얼굴 갱신

        if best_similarity > 0.97:  # 최고 유사도가 0.97 이상인 경우
            return best_match  # 가장 잘 맞는 얼굴 반환
        return None  # 그렇지 않은 경우 None 반환

    def compare_embedding(self, embedding, name, compare_embedding):
        similarity = cosine_similarity([embedding], [compare_embedding])[0][0]  # 두 임베딩 간의 유사도 계산
        return name, similarity  # 이름과 유사도 반환

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):  # 임베딩 파일이 존재하는 경우
            with open(self.embeddings_file, 'rb') as f:  # 임베딩 파일 열기
                self.embeddings = pickle.load(f)  # 임베딩 로드
        else:  # 임베딩 파일이 없는 경우
            self.embeddings = self.calculate_embeddings()  # 임베딩 계산
            self.save_embeddings()  # 임베딩 저장
        print("Embeddings loaded.")  # 로그 출력

    def save_embeddings(self):
        with open(self.embeddings_file, 'wb') as f:  # 임베딩 파일 열기
            pickle.dump(self.embeddings, f)  # 임베딩 저장
        print("Embeddings saved.")  # 로그 출력

    def recalculate_embeddings(self):
        self.embeddings = self.calculate_embeddings()  # 임베딩 재계산
        self.save_embeddings()  # 임베딩 저장
        print("Embeddings recalculated and saved.")  # 로그 출력

    def calculate_embeddings(self):
        embeddings = {}  # 임베딩 딕셔너리 초기화
        for root, _, files in os.walk(self.data_folder):  # 얼굴 이미지 폴더 내 모든 파일에 대해 반복
            for file_name in files:  # 각 파일에 대해
                file_path = os.path.join(root, file_name)  # 파일 경로 생성
                if os.path.isfile(file_path):  # 파일인 경우
                    image = face_recognition.load_image_file(file_path)  # 이미지 로드
                    embedding = self.get_face_embedding(image)  # 얼굴 임베딩 생성
                    if embedding is not None:  # 임베딩이 생성된 경우
                        embeddings[file_path] = embedding  # 임베딩 딕셔너리에 추가
        return embeddings  # 임베딩 딕셔너리 반환

    def draw_label(self, image, text, pos, bg_color):
        font_path = "malgun.ttf"  # 글꼴 파일 경로
        try:
            font = ImageFont.truetype(font_path, 20)  # 글꼴 로드
        except IOError:
            font = ImageFont.load_default()  # 기본 글꼴 로드

        img_pil = Image.fromarray(image)  # 이미지 PIL 객체로 변환
        draw = ImageDraw.Draw(img_pil)  # ImageDraw 객체 생성
        text_bbox = draw.textbbox(pos, text, font=font)  # 텍스트 바운딩 박스 계산
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])  # 텍스트 크기 계산
        text_x, text_y = pos  # 텍스트 위치
        draw.rectangle([text_x, text_y, text_x + text_size[0], text_y + text_size[1]], fill=bg_color)  # 배경 사각형 그리기
        draw.text(pos, text, font=font, fill=(255, 255, 255, 0))  # 텍스트 그리기
        return np.array(img_pil)  # 이미지를 numpy 배열로 변환하여 반환

    def on_escape(self, event):
        self.cap.release()  # 웹캠 해제
        self.window.quit()  # Tkinter 창 닫기

    def __del__(self):
        self.cap.release()  # 웹캠 해제

if __name__ == "__main__":
    root = tk.Tk()  # Tkinter 루트 창 생성
    app = VideoCaptureApp(root, "Tkinter and Webcam DNN")  # VideoCaptureApp 객체 생성


# class VideoCaptureApp:
#     def __init__(self, window, window_title):
#         self.window = window  # Tkinter 창 설정
#         self.window.title(window_title)  # 창 제목 설정

#         # RealSense 카메라 초기화
#         self.pipeline = rs.pipeline()  # RealSense 파이프라인 생성
#         self.config = rs.config()  # RealSense 설정 객체 생성
#         self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 깊이 스트림 설정
#         self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 컬러 스트림 설정
#         self.pipeline.start(self.config)  # 설정한 스트림으로 파이프라인 시작

#         self.canvas = tk.Canvas(window, width=640, height=480)  # Tkinter 캔버스 생성
#         self.canvas.pack()  # 캔버스를 창에 배치

#         # DNN 모델 로드 (얼굴 감지용)
#         self.net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')  # DNN 모델 로드

#         self.delay = 1  # 업데이트 딜레이 설정
#         self.frame = None  # 현재 프레임을 저장할 변수
#         self.capture_faces = False  # 얼굴 캡처 여부를 저장할 변수
#         self.captured_faces = []  # 캡처된 얼굴을 저장할 리스트
#         self.embeddings = {}  # 얼굴 임베딩을 저장할 딕셔너리
#         self.data_folder = "face_image"  # 얼굴 이미지 저장 폴더
#         self.embeddings_file = "embeddings.pkl"  # 얼굴 임베딩 파일
#         self.recalculate_embeddings()  # 임베딩 재계산
#         self.update()  # 업데이트 함수 호출

#         # 키 이벤트 바인딩
#         self.window.bind('<Escape>', self.on_escape)  # Escape 키를 누르면 프로그램 종료
#         self.window.bind('<space>', self.start_capture)  # Space 키를 누르면 얼굴 캡처 시작

#         self.window.mainloop()  # Tkinter 이벤트 루프 시작

#     def update(self):
#         # RealSense 카메라에서 프레임 가져오기
#         frames = self.pipeline.wait_for_frames()  # 카메라 프레임 대기
#         color_frame = frames.get_color_frame()  # 컬러 프레임 가져오기
#         if not color_frame:  # 컬러 프레임이 없으면 반환
#             return

#         frame = np.asanyarray(color_frame.get_data())  # 프레임을 numpy 배열로 변환
#         self.frame = frame  # 현재 프레임 저장
#         (h, w) = frame.shape[:2]  # 프레임의 높이와 너비 가져오기
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))  # DNN 입력 블롭 생성

#         # DNN 모델을 사용하여 얼굴 감지
#         self.net.setInput(blob)  # DNN 모델 입력 설정
#         detections = self.net.forward()  # 얼굴 감지 실행

#         for i in range(0, detections.shape[2]):  # 감지된 얼굴 수만큼 반복
#             confidence = detections[0, 0, i, 2]  # 감지 신뢰도 가져오기

#             if confidence > 0.5:  # 신뢰도가 0.5 이상인 경우
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # 얼굴 위치 박스 가져오기
#                 (startX, startY, endX, endY) = box.astype("int")  # 정수로 변환
#                 cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)  # 얼굴 위치에 사각형 그리기

#                 face = frame[startY:endY, startX:endX]  # 얼굴 이미지 추출
#                 face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 얼굴 이미지를 RGB로 변환
#                 embedding = self.get_face_embedding(face_rgb)  # 얼굴 임베딩 생성
#                 if embedding is not None:  # 임베딩이 생성된 경우
#                     best_match = self.find_best_match(embedding)  # 가장 잘 맞는 얼굴 찾기
#                     if best_match:  # 잘 맞는 얼굴이 있는 경우
#                         label = os.path.basename(best_match).split('_')[0]  # 얼굴 레이블 추출
#                         frame = self.draw_label(frame, label, (startX, startY - 20), (255, 0, 0))  # 레이블 그리기

#                 # 얼굴 캡처 중이면 얼굴을 저장
#                 if self.capture_faces and len(self.captured_faces) < 5:  # 캡처된 얼굴 수가 5 미만인 경우
#                     self.captured_faces.append(face)  # 얼굴 리스트에 추가

#         self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))  # 프레임을 Tkinter 이미지로 변환
#         self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)  # 캔버스에 이미지 그리기
        
#         # 다음 업데이트를 예약
#         self.window.after(self.delay, self.update)  # 일정 시간 후에 다시 update 함수 호출

#     def start_capture(self, event):
#         self.capture_faces = True  # 얼굴 캡처 시작
#         self.captured_faces = []  # 캡처된 얼굴 리스트 초기화
#         self.window.after(100, self.check_capture_complete)  # 일정 시간 후에 얼굴 캡처 완료 확인

#     def check_capture_complete(self):
#         if len(self.captured_faces) >= 5:  # 캡처된 얼굴 수가 5 이상인 경우
#             self.capture_faces = False  # 얼굴 캡처 중지
#             self.prompt_for_file_name()  # 파일 이름 입력 요청
#         else:
#             self.window.after(100, self.check_capture_complete)  # 일정 시간 후에 다시 얼굴 캡처 완료 확인

#     def prompt_for_file_name(self):
#         file_name = simpledialog.askstring("Input", "Enter the file name:")  # 파일 이름 입력 대화상자 표시
#         if file_name:  # 파일 이름이 입력된 경우
#             self.save_faces(file_name)  # 얼굴 저 장

#     def save_faces(self, file_name):
#         folder_path = os.path.join(self.data_folder, file_name)  # 새로운 폴더 경로 생성
#         if not os.path.exists(folder_path):  # 새로운 폴더가 없는 경우
#             os.makedirs(folder_path)  # 폴더 생성
        
#         new_embeddings = {}  # 새로운 얼굴 임베딩 딕셔너리 생성
#         for idx, face in enumerate(self.captured_faces):  # 캡처된 얼굴 수만큼 반복
#             face_filename = f"{folder_path}/{file_name}_{idx+1}.jpg"  # 얼굴 파일 이름 생성
#             face_filename = face_filename.encode('utf-8').decode('utf-8')  # 파일 이름 인코딩 처리
#             success, encoded_image = cv2.imencode('.jpg', face)  # 얼굴 이미지를 JPEG 형식으로 인코딩
#             if success:  # 인코딩 성공한 경우
#                 with open(face_filename, 'wb') as f:  # 파일 열기
#                     f.write(encoded_image.tobytes())  # 인코딩된 이미지 저장
#                 face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 얼굴 이미지를 RGB로 변환
#                 new_embedding = self.get_face_embedding(face_rgb)  # 새로운 얼굴 임베딩 생성
#                 if new_embedding is not None:  # 임베딩이 생성된 경우
#                     new_embeddings[face_filename] = new_embedding  # 새로운 임베딩 딕셔너리에 추가

#         self.embeddings.update(new_embeddings)  # 기존 임베딩에 새로운 임베딩 추가
#         self.save_embeddings()  # 임베딩 저장
#         print("New faces saved and embeddings updated.")  # 로그 출력

#     def get_face_embedding(self, image):
#         face_locations = face_recognition.face_locations(image)  # 얼굴 위치 찾기
#         if len(face_locations) > 0:  # 얼굴이 하나 이상인 경우
#             face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)  # 얼굴 임베딩 생성
#             if face_encodings:  # 임베딩이 생성된 경우
#                 return face_encodings[0]  # 첫 번째 임베딩 반환
#         return None  # 임베딩이 없는 경우 None 반환

#     def find_best_match(self, embedding):
#         best_match = None  # 가장 잘 맞는 얼굴 변수 초기화
#         best_similarity = 0.0  # 최고 유사도 변수 초기화

#         with ThreadPoolExecutor() as executor:  # 스레드 풀 생성
#             futures = [executor.submit(self.compare_embedding, embedding, name, emb)  # 임베딩 비교 작업 추가
#                        for name, emb in self.embeddings.items()]  # 모든 임베딩에 대해 비교 작업 생성

#             for future in futures:  # 각 비교 작업 결과 가져오기
#                 match, similarity = future.result()  # 비교 결과
#                 if similarity > best_similarity:  # 유사도가 최고 유사도보다 큰 경우
#                     best_similarity = similarity  # 최고 유사도 갱신
#                     best_match = match  # 가장 잘 맞는 얼굴 갱신

#         if best_similarity > 0.95:  # 최고 유사도가 0.95 이상인 경우
#             return best_match  # 가장 잘 맞는 얼굴 반환
#         return None  # 그렇지 않은 경우 None 반환

#     def compare_embedding(self, embedding, name, compare_embedding):
#         similarity = cosine_similarity([embedding], [compare_embedding])[0][0]  # 두 임베딩 간의 유사도 계산
#         return name, similarity  # 이름과 유사도 반환

#     def load_embeddings(self):
#         if os.path.exists(self.embeddings_file):  # 임베딩 파일이 존재하는 경우
#             with open(self.embeddings_file, 'rb') as f:  # 임베딩 파일 열기
#                 self.embeddings = pickle.load(f)  # 임베딩 로드
#         else:  # 임베딩 파일이 없는 경우
#             self.embeddings = self.calculate_embeddings()  # 임베딩 계산
#             self.save_embeddings()  # 임베딩 저장
#         print("Embeddings loaded.")  # 로그 출력

#     def save_embeddings(self):
#         with open(self.embeddings_file, 'wb') as f:  # 임베딩 파일 열기
#             pickle.dump(self.embeddings, f)  # 임베딩 저장
#         print("Embeddings saved.")  # 로그 출력

#     def recalculate_embeddings(self):
#         self.embeddings = self.calculate_embeddings()  # 임베딩 재계산
#         self.save_embeddings()  # 임베딩 저장
#         print("Embeddings recalculated and saved.")  # 로그 출력

#     def calculate_embeddings(self):
#         embeddings = {}  # 임베딩 딕셔너리 초기화
#         for root, _, files in os.walk(self.data_folder):  # 얼굴 이미지 폴더 내 모든 파일에 대해 반복
#             for file_name in files:  # 각 파일에 대해
#                 file_path = os.path.join(root, file_name)  # 파일 경로 생성
#                 if os.path.isfile(file_path):  # 파일인 경우
#                     image = face_recognition.load_image_file(file_path)  # 이미지 로드
#                     embedding = self.get_face_embedding(image)  # 얼굴 임베딩 생성
#                     if embedding is not None:  # 임베딩이 생성된 경우
#                         embeddings[file_path] = embedding  # 임베딩 딕셔너리에 추가
#         return embeddings  # 임베딩 딕셔너리 반환

#     def draw_label(self, image, text, pos, bg_color):
#         font_path = "malgun.ttf"  # 글꼴 파일 경로
#         try:
#             font = ImageFont.truetype(font_path, 20)  # 글꼴 로드
#         except IOError:
#             font = ImageFont.load_default()  # 기본 글꼴 로드

#         img_pil = Image.fromarray(image)  # 이미지 PIL 객체로 변환
#         draw = ImageDraw.Draw(img_pil)  # ImageDraw 객체 생성
#         text_bbox = draw.textbbox(pos, text, font=font)  # 텍스트 바운딩 박스 계산
#         text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])  # 텍스트 크기 계산
#         text_x, text_y = pos  # 텍스트 위치
#         draw.rectangle([text_x, text_y, text_x + text_size[0], text_y + text_size[1]], fill=bg_color)  # 배경 사각형 그리기
#         draw.text(pos, text, font=font, fill=(255, 255, 255, 0))  # 텍스트 그리기
#         return np.array(img_pil)  # 이미지를 numpy 배열로 변환하여 반환

#     def on_escape(self, event):
#         self.pipeline.stop()  # RealSense 파이프라인 정지
#         self.window.quit()  # Tkinter 창 닫기

#     def __del__(self):
#         self.pipeline.stop()  # RealSense 파이프라인 정지

# if __name__ == "__main__":
#     root = tk.Tk()  # Tkinter 루트 창 생성
#     app = VideoCaptureApp(root, "Tkinter and RealSense DNN")  # VideoCaptureApp 객체 생성
