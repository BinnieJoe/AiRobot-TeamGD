import sys
import cv2
import torch
import time
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer, QThread, Signal, QObject
from threading import Thread, Lock, Event
import numpy as np
from queue import Queue, deque
from Guide_ui import Ui_MainWindow
from cam import Depth
import openai
import asyncio
import concurrent.futures
import speech_recognition as sr
import os
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from concurrent.futures import ThreadPoolExecutor
import pyrealsense2 as rs
from gtts import gTTS
import pyttsx3
import pygame
import json
from pydub import AudioSegment
from pydub.playback import play
from ultralytics import YOLO
import logging

# web.py에서 VideoStreamer 클래스를 임포트합니다.
import web

# ffmpeg 경로 설정
AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"

# YOLO 로깅 비활성화
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# 인식할 클래스 이름들을 정의합니다.
cls_names = ['자전거', '자동차', '의자', '횡단보도', '문', '전동킥보드', '사람', '오토바이', '계단', '녹색 신호', '적색 신호', '나무', '전봇대']

# JSON 파일에서 객체별 멘트를 로드합니다.
def load_object_messages(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

object_messages = load_object_messages('object_messages.json')

class FaceManager:
    def __init__(self):
        self.data_folder = "face_image"  # 얼굴 이미지를 저장할 폴더
        self.embeddings_file = "embeddings.pkl"  # 얼굴 임베딩을 저장할 파일
        self.embeddings = {}  # 얼굴 임베딩을 저장할 딕셔너리
        self.load_embeddings()  # 얼굴 임베딩을 로드
        self.known_files = set(os.listdir(self.data_folder))  # 이미 존재하는 파일 목록 저장

    def save_faces(self, faces, folder_name):
        folder_path = os.path.join(self.data_folder, folder_name)  # 저장할 폴더 경로 생성
        if not os.path.exists(folder_path):  # 폴더가 존재하지 않으면
            os.makedirs(folder_path)  # 폴더 생성

        new_embeddings = {}  # 새로운 얼굴 임베딩을 저장할 딕셔너리
        for idx, face in enumerate(faces):  # 얼굴 리스트를 순회
            face_filename = f"{folder_path}/{folder_name}_{idx+1}.jpg"  # 얼굴 파일 이름 생성
            face_filename = face_filename.encode('utf-8').decode('utf-8')  # 파일 이름을 UTF-8로 인코딩
            success, encoded_image = cv2.imencode('.jpg', face)  # 얼굴 이미지를 JPG로 인코딩
            if success:  # 인코딩 성공 시
                with open(face_filename, 'wb') as f:  # 파일을 바이너리 모드로 열기
                    f.write(encoded_image.tobytes())  # 이미지 데이터를 파일에 쓰기
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # BGR 이미지를 RGB로 변환
                new_embedding = self.get_face_embedding(face_rgb)  # 얼굴 임베딩을 계산
                if new_embedding is not None:  # 임베딩이 존재하면
                    new_embeddings[face_filename] = new_embedding  # 새로운 임베딩을 딕셔너리에 추가

        self.embeddings.update(new_embeddings)  # 기존 임베딩에 새로운 임베딩 추가
        self.save_embeddings()  # 임베딩을 파일에 저장
        print(f"새 얼굴이 저장되고 포함이 업데이트되었습니다. {folder_path}.")  # 얼굴 저장 완료 메시지 출력

    def get_face_embedding(self, image):
        face_locations = face_recognition.face_locations(image)  # 얼굴 위치를 찾기
        if len(face_locations) > 0:  # 얼굴이 하나 이상 감지되면
            face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)  # 얼굴 임베딩 계산
            if face_encodings:  # 얼굴 임베딩이 존재하면
                return face_encodings[0]  # 첫 번째 얼굴 임베딩 반환
        return None  # 얼굴이 없으면 None 반환

    def find_best_match(self, embedding):
        best_match = None  # 가장 유사한 얼굴 초기값
        best_similarity = 0.0  # 최대 유사도 초기값

        with ThreadPoolExecutor() as executor:  # 스레드 풀 실행
            futures = [executor.submit(self.compare_embedding, embedding, name, emb)
                       for name, emb in self.embeddings.items()]  # 모든 임베딩에 대해 유사도 계산

            for future in futures:  # 모든 결과를 순회
                match, similarity = future.result()  # 결과에서 유사도와 이름 추출
                if similarity > best_similarity:  # 현재 유사도가 최대 유사도보다 크면
                    best_similarity = similarity  # 최대 유사도 갱신
                    best_match = match  # 가장 유사한 얼굴 갱신

        if best_similarity > 0.97:  # 유사도가 0.8보다 크면
            return best_match  # 가장 유사한 얼굴 반환
        return None  # 유사한 얼굴이 없으면 None 반환

    def compare_embedding(self, embedding, name, compare_embedding):
        similarity = cosine_similarity([embedding], [compare_embedding])[0][0]  # 두 임베딩 간 유사도 계산
        return name, similarity  # 이름과 유사도 반환

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):  # 임베딩 파일이 존재하면
            try:
                with open(self.embeddings_file, 'rb') as f:  # 파일을 바이너리 모드로 열기
                    self.embeddings = pickle.load(f)  # 임베딩 로드
            except (EOFError, pickle.UnpicklingError):
                print("임베딩 파일을 로드하는 중 오류가 발생했습니다. 새로운 파일을 생성합니다.")
                self.embeddings = {}
        else:  # 임베딩 파일이 없으면
            self.embeddings = self.calculate_embeddings()  # 임베딩 계산
            self.save_embeddings()  # 임베딩을 파일에 저장
        print("Embeddings loaded.")  # 임베딩 로드 완료 메시지 출력

    def save_embeddings(self):
        with open(self.embeddings_file, 'wb') as f:  # 파일을 바이너리 모드로 열기
            pickle.dump(self.embeddings, f)  # 임베딩을 파일에 저장
        print("Embeddings saved.")  # 임베딩 저장 완료 메시지 출력

    def recalculate_embeddings(self):
        self.embeddings = self.calculate_embeddings()  # 임베딩 재계산
        self.save_embeddings()  # 임베딩 저장
        print("Embeddings recalculated and saved.")  # 로그 출력

    def calculate_embeddings(self):
        embeddings = {}  # 임베딩 딕셔너리 초기화
        valid_extensions = ('.jpg', '.jpeg', '.png')  # 유효한 이미지 파일 확장자
        for root, _, files in os.walk(self.data_folder):  # 얼굴 이미지 폴더 내 모든 파일에 대해 반복
            for file_name in files:  # 각 파일에 대해
                if not file_name.lower().endswith(valid_extensions):
                    continue  # 이미지 파일이 아니면 건너뜀
                file_path = os.path.join(root, file_name)  # 파일 경로 생성
                if os.path.isfile(file_path):  # 파일인 경우
                    image = face_recognition.load_image_file(file_path)  # 이미지 로드
                    embedding = self.get_face_embedding(image)  # 얼굴 임베딩 생성
                    if embedding is not None:  # 임베딩이 생성된 경우
                        embeddings[file_path] = embedding  # 임베딩 딕셔너리에 추가
        return embeddings  # 임베딩 딕셔너리 반환

    def monitor_folder(self):
        while True:
            current_files = set(os.listdir(self.data_folder))  # 현재 폴더 내 파일 목록 가져오기
            new_files = current_files - self.known_files  # 새로운 파일 찾기
            if new_files:
                for new_file in new_files:
                    file_path = os.path.join(self.data_folder, new_file)
                    if os.path.isfile(file_path):
                        image = face_recognition.load_image_file(file_path)
                        embedding = self.get_face_embedding(image)
                        if embedding is not None:
                            self.embeddings[file_path] = embedding  # 새로운 임베딩 추가
                            self.save_embeddings()  # 임베딩 저장
                            print(f"New face {new_file} added to embeddings.")
                self.known_files = current_files  # 파일 목록 업데이트
            time.sleep(1)  # 1초 대기

class RecordingThread(QThread):
    recording_done = Signal(str)  # 녹음 완료 시그널

    def __init__(self, folder_path, stop_event):
        super().__init__()
        self.folder_path = folder_path
        self.is_recording = True
        self.stop_event = stop_event
        self.audio_buffer = deque(maxlen=15000)  # 15초 길이의 버퍼 생성
        self.silence_chunk = AudioSegment.silent(duration=1000)  # 1초 길이의 무음 생성

    def run(self):
        r = sr.Recognizer()
        mic = sr.Microphone()

        print("녹음을 시작합니다. 말을 해주세요.")
        while self.is_recording and not self.stop_event.is_set():
            with mic as source:
                r.adjust_for_ambient_noise(source)
                try:
                    audio = r.listen(source, timeout=5)  # 5초 대기 후 다음 루프로 넘어감
                    self.add_to_buffer(audio)
                except sr.WaitTimeoutError:
                    continue

        self.save_last_15_seconds()
        self.recording_done.emit(self.folder_path)

    def stop(self):
        self.is_recording = False

    def add_to_buffer(self, audio):
        temp_wav_path = os.path.join(self.folder_path, "temp_conversation.wav")
        temp_mp3_path = os.path.join(self.folder_path, "conversation.mp3")
        with open(temp_wav_path, "wb") as f:
            f.write(audio.get_wav_data())

        sound = AudioSegment.from_wav(temp_wav_path)
        os.remove(temp_wav_path)  # 임시 WAV 파일 삭제

        self.audio_buffer.append(sound)  # 버퍼에 추가
        combined = sum(self.audio_buffer)  # 버퍼의 모든 오디오를 합침

        combined.export(temp_mp3_path, format="mp3")

    def save_last_15_seconds(self):
        audio_path = os.path.join(self.folder_path, "conversation.mp3")
        if os.path.exists(audio_path):
            combined = sum(self.audio_buffer)  # 버퍼의 모든 오디오를 합침
            if len(combined) > 15000:  # 15초보다 긴 경우
                combined = combined[-15000:]  # 마지막 15초만 저장
            combined.export(audio_path, format="mp3")
            print("녹음된 파일의 마지막 15초가 저장되었습니다.")
        else:
            print("녹음 파일을 찾을 수 없습니다.")

class OpenAi(QObject):
    text_to_voice_done = Signal()  # 음성 출력 완료 시그널

    def __init__(self, api_key):
        super().__init__()
        openai.api_key = api_key  # OpenAI API 키 설정
        self.content = ''  # 대화 내용을 저장하는 변수
        self.is_chatting = False  # 대화 중인지 여부를 나타내는 플래그
        self.volume_lock = Lock()  # 볼륨 조절을 위한 락
        self.is_paused = False  # 대화 일시정지 여부
        self.current_folder = None  # 현재 대화 중인 폴더
        self.original_volume = 1.0  # 원래 볼륨

        # 녹음 관련 변수
        self.is_recording = False
        self.recording_thread = None
        self.stop_recording_event = Event()  # 녹음 중지 이벤트 추가

        # pygame 초기화
        pygame.mixer.init()

        # 녹음 중지 명령을 듣는 스레드 추가
        self.listen_thread = Thread(target=self.listen_for_stop_command)
        self.listen_thread.daemon = True
        self.listen_thread.start()

    # 텍스트를 음성으로 변환하는 함수
    def text_to_voice(self, text, volume=1.0):
        Thread(target=self._text_to_voice_internal, args=(text, volume)).start()

    def _text_to_voice_internal(self, text, volume):
        with self.volume_lock:  # 볼륨 조절을 위해 락을 사용
            tts = gTTS(text=text, lang='ko')  # gTTS를 사용하여 텍스트를 음성으로 변환
            temp_filename = "Guide.mp3"  # 임시 파일 이름 설정
            tts.save(temp_filename)  # 음성을 파일로 저장
            pygame.mixer.music.load(temp_filename)  # pygame을 사용하여 음성 파일 로드
            pygame.mixer.music.set_volume(volume)  # 볼륨 설정
            pygame.mixer.music.play()  # 음성 재생
            while pygame.mixer.music.get_busy():  # 음성이 재생 중일 때 대기
                pygame.time.Clock().tick(10)  # 10ms 대기
            pygame.mixer.music.unload()  # 음성 파일 언로드
            try:
                os.remove(temp_filename)  # 임시 파일 삭제
            except PermissionError:
                print("파일을 삭제할 수 없습니다. 다른 프로세스에서 사용 중입니다.")  # 파일 삭제 실패 시 메시지 출력
        self.text_to_voice_done.emit()  # 음성 출력 완료 시그널 발생

    # 음성을 텍스트로 변환하는 함수
    def voice_to_text(self):
        r = sr.Recognizer()  # 음성 인식기를 초기화
        with sr.Microphone() as source:  # 마이크를 입력 소스로 사용
            print('음성 텍스트를 입력받습니다.')
            r.adjust_for_ambient_noise(source)  # 주변 잡음에 맞춰 음성 인식기 조정
            audio = r.listen(source)
            try:
                if audio.frame_data:  # 오디오 프레임이 있는 경우에만 처리
                    said = r.recognize_google(audio, language="ko-KR")  # Google API를 사용하여 한국어로 음성을 텍스트로 변환
                    print("입력한 음성 텍스트 : ", said)
                    return said
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")  # 요청 실패 시 예외 처리
                return None

    # OpenAI API 호출을 래핑하는 함수
    def fetch_gpt_response(self, messages):
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)  # OpenAI ChatCompletion API 호출
        return response['choices'][0]['message']['content'].strip()  # API 응답에서 필요한 내용을 추출하여 반환

    # GPT-3 응답을 비동기적으로 받기 위한 함수
    async def get_gpt_response(self, messages):
        loop = asyncio.get_running_loop()  # 현재 이벤트 루프를 가져옴
        with concurrent.futures.ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(pool, self.fetch_gpt_response, messages)  # 별도의 스레드에서 OpenAI API 호출
        return response  # 비동기 응답 반환

    # 대화 루프를 실행하는 함수
    async def chat_loop(self):
        self.is_chatting = True
        self.text_to_voice("안녕하세요, 대화를 시작합니다.")  # 초기 인사 메시지를 음성으로 출력

        while self.is_chatting:
            if self.is_paused:
                await asyncio.sleep(1)
                continue

            prompt = self.voice_to_text()  # 사용자의 음성을 텍스트로 변환

            if prompt is None:
                continue

            if "얼굴 저장" in prompt:
                self.pause_chat()
                self.text_to_voice("얼굴을 저장할 사람의 이름을 말해주세요.")
                name = self.voice_to_text()
                if name and len(name) <= 4:
                    self.save_face(name)
                    self.text_to_voice("얼굴 저장이 완료되었습니다.")
                else:
                    self.text_to_voice("이름이 너무 깁니다. 다시 시도해주세요.")
                self.resume_chat()

            elif "정지" in prompt:
                window.pause_object_recognition()
                self.text_to_voice("사물인식을 종료합니다.")

            elif "재생" in prompt:
                window.resume_object_recognition()
                self.text_to_voice("사물인식을 재생합니다.")

            # 녹음 기능 시작
            elif "시작" in prompt:
                if not self.is_recording:
                    self.is_recording = True
                    self.pause_chat()
                    self.text_to_voice("대화 녹음을 시작합니다.")
                    os.makedirs(self.current_folder, exist_ok=True)
                    self.start_recording(self.current_folder)

            if prompt:  # 만약 음성을 인식했다면
                try:
                    messages = [
                        {'role': 'system', 'content': 'You are a helpful assistant who responds in Korean.'},  # 시스템 메시지
                        {'role': 'user', 'content': self.content},  # 이전 대화 내용 추가
                    ]
                    messages.append({'role': 'user', 'content': prompt})  # 사용자의 프롬프트 추가
                except Exception as e:
                    print(f"An error occurred: {e}")  # 예외 발생 시 예외 내용 출력
                    messages = [
                        {'role': 'system', 'content': 'You are a helpful assistant who responds in Korean.'},  # 예외 발생 시 기본 메시지 설정
                        {'role': 'user', 'content': prompt},
                    ]

                # 비동기적으로 GPT-3 응답을 받음
                response = await self.get_gpt_response(messages)  # OpenAI API에 메시지를 보내고 응답을 받음

                print("-------------------------------------------")
                print(response)  # GPT-3의 응답을 콘솔에 출력
                print("-------------------------------------------")

                self.text_to_voice(response, volume=0.5)  # GPT-3의 응답을 음성으로 출력, 볼륨 낮춤

                await asyncio.sleep(0.1)  # 비동기 함수에서 짧은 대기

                self.content += response  # 대화 내용을 업데이트

    def listen_for_stop_command(self):
        r = sr.Recognizer()
        mic = sr.Microphone()
        stop_command_given = False  # 대화 종료 명령 플래그 초기화
        while not stop_command_given:
            with mic as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                try:
                    if audio.frame_data:
                        said = r.recognize_google(audio, language="ko-KR")
                        print("감지된 음성 텍스트:", said)
                        if "대화 끝" in said:
                            self.stop_recording_event.set()
                            self.text_to_voice("대화 종료 명령이 감지되었습니다. 녹음을 중지합니다.")
                            stop_command_given = True  # 대화 종료 명령 플래그 설정
                            break
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    continue

    def save_face(self, name):
        faces = []  # 캡처된 얼굴 리스트를 초기화
        for _ in range(5):
            frames = window.depth_camera.get_frame()
            color_frame = frames[1]
            face_locations = face_recognition.face_locations(color_frame)
            if face_locations:
                for (top, right, bottom, left) in face_locations:
                    face_image = color_frame[top:bottom, left:right]
                    faces.append(face_image)
        if faces:
            window.pause_object_recognition()
            window.pause_speech()
            window.face_manager.save_faces(faces, name)
            self.text_to_voice(f"{name} 얼굴이 저장되었습니다.")
            window.resume_object_recognition()
            window.resume_speech()

    def start_recording(self, folder_path):
        self.recording_thread = RecordingThread(folder_path, self.stop_recording_event)
        self.recording_thread.recording_done.connect(self.on_recording_done)
        self.recording_thread.start()
        self.stop_recording_event.clear()  # 녹음 중지 이벤트 초기화

    def on_recording_done(self, folder_path):
        print(f"녹음이 완료되었습니다: {folder_path}")
        self.text_to_voice("녹음이 완료되었습니다.")
        self.resume_chat()
        window.resume_object_recognition()

    def save_last_15_seconds(self):
        audio_path = os.path.join(self.current_folder, "conversation.mp3")
        if os.path.exists(audio_path):
            combined = sum(self.recording_thread.audio_buffer)  # 버퍼의 모든 오디오를 합침
            if len(combined) > 15000:  # 15초보다 긴 경우
                combined = combined[-15000:]  # 마지막 15초만 저장
            combined.export(audio_path, format="mp3")
            print("녹음된 파일의 마지막 15초가 저장되었습니다.")
            self.text_to_voice("녹음된 파일의 마지막 15초가 저장되었습니다.")
        else:
            print("녹음 파일을 찾을 수 없습니다.")
            self.text_to_voice("녹음 파일을 찾을 수 없습니다.")

    def play_previous_conversation(self, folder_path):
        conversation_path = os.path.join(folder_path, "conversation.mp3")
        if os.path.exists(conversation_path):
            print(f"이전 대화를 재생합니다: {conversation_path}")
            play_thread = Thread(target=self.play_audio, args=(conversation_path,))
            play_thread.start()
        else:
            print("이전 대화 파일을 찾을 수 없습니다.")

    def play_audio(self, audio_path):
        audio = AudioSegment.from_mp3(audio_path)
        play(audio)

    def monitor_recording(self, folder_path):
        while self.is_recording:
            if self.stop_recording_event.is_set():  # 녹음 중지 이벤트가 설정된 경우
                self.recording_thread.stop()  # 녹음 스레드 중지
                self.recording_thread.wait()  # 녹음 스레드 종료 대기
                self.save_last_15_seconds()  # 마지막 15초 저장
                self.text_to_voice("녹음이 완료되었습니다.")
                self.resume_chat()
                window.resume_object_recognition()
                break

    def pause_chat(self):
        self.is_paused = True

    def resume_chat(self):
        self.is_paused = False

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()  # 부모 클래스의 초기화자 호출
        self.setupUi(self)  # UI 초기화
        self.setWindowTitle("시각장애를 위한 객체인식 프로그램")  # 창 제목 설정

        # CUDA가 가능하면 GPU를 사용합니다.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # YOLOv8 모델을 로드합니다.
        self.model = YOLO('./v8n/best.pt')
        # 모델을 설정한 디바이스로 이동합니다.
        self.model.to(device=self.device)

        # Depth 객체 생성 및 파이프라인 시작
        self.depth_camera = Depth()
        self.depth_camera.start_pipeline()

        # 음성 출력을 위한 큐 생성
        self.speech_queue = Queue()

        # gtts 사용을 위한 락 생성
        self.speech_lock = Lock()
        self.speech_event = Event()  # 음성 출력 완료 이벤트 생성

        # pyttsx3 초기화
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        # 음성 출력을 관리하는 스레드 생성 및 시작
        self.speech_thread = Thread(target=self.process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()

        # 프로그램 시작 메시지를 출력
        self.is_speech_paused = False  # 음성 출력 일시정지 플래그 초기화
        self.speak("객체인식 프로그램을 시작합니다.", use_pyttsx3=True)

        self.processing_speech = False  # 음성 출력 중인지 여부를 나타내는 플래그
        self.original_volume = 1.0  # 원래 볼륨

        self.frame_skip = 2  # 매 2 프레임마다 처리를 수행하도록 설정
        self.frame_count = 0  # 프레임 카운터 초기화
        self.results = None  # 인식 결과 초기화
        self.recognized_faces = {}  # 인식된 얼굴과 시간을 저장하는 딕셔너리

        self.detected_objects = {}  # 감지된 객체와 시간을 저장하는 딕셔너리

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

        # OpenAi 인스턴스 생성
        self.openai_api_key = 'sk-None-RTmNg3xO2h2492S67mYsT3BlbkFJvJuStodEdD0TumaCmdZx'  # 자신의 OpenAI API 키로 변경
        self.ai = OpenAi(api_key=self.openai_api_key)

        # 얼굴 인식을 위한 설정
        self.face_manager = FaceManager()

        # 새로운 변수 초기화
        self.delay = 1  # 업데이트 딜레이 설정
        self.frame = None  # 현재 프레임을 저장할 변수
        self.capture_faces = False  # 얼굴 캡처 여부를 저장할 변수
        self.captured_faces = []  # 캡처된 얼굴을 저장할 리스트
        self.embeddings = {}  # 얼굴 임베딩을 저장할 딕셔너리
        self.data_folder = "face_image"  # 얼굴 이미지 저장 폴더
        self.embeddings_file = "embeddings.pkl"  # 얼굴 임베딩 파일
        self.face_manager.recalculate_embeddings()  # 임베딩 재계산
        self.update()  # 업데이트 함수 호출

        # 객체 인식 일시 정지 플래그 초기화
        self.is_object_recognition_paused = False

        # 이벤트 필터 설치
        self.installEventFilter(self)

        # OpenAI 대화 기능을 백그라운드에서 실행
        self.start_openai_chat()

        # face_image 폴더 모니터링 스레드 시작
        self.face_monitor_thread = Thread(target=self.face_manager.monitor_folder)
        self.face_monitor_thread.daemon = True
        self.face_monitor_thread.start()

        # VideoStreamer 인스턴스를 생성하여 웹캠을 위한 스트리밍을 준비합니다.
        web.create_video_streamer(self.depth_camera)
        self.web_thread = Thread(target=self.run_flask_app, daemon=True)
        self.web_thread.start()

    def eventFilter(self, obj, event):
        return super(MainWindow, self).eventFilter(obj, event)

    def start_openai_chat(self):
        if not self.ai.is_chatting:
            Thread(target=self.run_asyncio, daemon=True).start()

    def run_asyncio(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ai.chat_loop())

    def run_flask_app(self):
        web.start_recording()
        web.app.run(host='0.0.0.0', port=5000)

    def pause_object_recognition(self):
        self.is_object_recognition_paused = True

    def resume_object_recognition(self):
        self.is_object_recognition_paused = False

    def pause_speech(self):
        self.is_speech_paused = True

    def resume_speech(self):
        self.is_speech_paused = False

    # 이미지를 QImage로 변환하는 함수
    def convert2QImage(self, img):
        height, width, channel = img.shape  # 이미지의 높이, 너비, 채널 수를 가져옴
        bytes_per_line = width * channel  # 한 줄당 바이트 수 계산
        # QImage 객체 생성 (RGB888 포맷 사용)
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

    # 비디오 프레임을 예측하는 함수
    def video_pred(self):
        if self.is_object_recognition_paused:  # 객체 인식이 일시 정지되었는지 확인
            return

        depth_image, color_image = self.depth_camera.get_frame()  # Depth 카메라에서 프레임 가져오기
        if color_image is None:  # 컬러 이미지가 없는 경우 종료
            return

        else:
            self.frame_count += 1  # 프레임 카운터 증가
            if self.frame_count % self.frame_skip == 0:  # 매 2 프레임마다 다음을 수행
                frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # BGR 이미지를 RGB로 변환
                frame = cv2.resize(frame, (640, 640))  # 이미지 크기를 640x640으로 조정
                start = time.perf_counter()  # 예측 시작 시간 기록
                results = self.model(frame)  # 모델을 사용하여 객체를 감지
                image = results[0].plot()  # 감지 결과를 이미지에 렌더링
                self.update_frame_ui(self.label, image)  # 객체 인식 결과를 label에 표시
                end = time.perf_counter()  # 예측 종료 시간 기록
                detections = results[0].boxes.data.cpu().numpy()  # 감지된 객체를 배열로 변환
                if len(detections) > 0:  # 감지된 객체가 있는 경우
                    sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)[:1]  # 가장 확신이 높은 객체 선택
                    for det in sorted_detections:
                        x1, y1, x2, y2, conf, cls = det  # 감지 결과를 분리
                        if conf < 0.6:  # 인식률이 60% 미만인 경우 무시
                            continue
                        cls = int(cls)
                        class_name = cls_names[cls] if cls < len(cls_names) else f"클래스 {cls}"  # 클래스 이름을 가져옴

                        center_x = int((x1 + x2) / 2)  # 객체의 중심 x좌표 계산
                        center_y = int((y1 + y2) / 2)  # 객체의 중심 y좌표 계산

                        # 중심 좌표가 깊이 이미지의 범위 내에 있는지 확인하고 거리 계산
                        if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                            distance = depth_image[center_y, center_x] * 0.001  # mm를 meter로 변환
                            if distance < 1 or distance > 10:  # 1미터에서 10미터 사이가 아닌 경우 무시
                                continue
                        else:
                            return
                        

                        distance_rounded = round(distance, 2)  # 거리를 반올림

                        # 객체별 멘트를 JSON에서 가져와서 출력
                        message = object_messages.get(class_name, f"{class_name}이(가) 있습니다. 주의하세요.")
                        message = f"{message} {distance_rounded} 미터 앞에 있습니다."

                        # 볼륨 낮추기
                        self.lower_volume()

                        # 객체가 새로운 객체이거나 위치가 이동했는지 확인
                        if class_name not in self.detected_objects or self.has_object_moved(class_name, center_x, center_y):
                            self.detected_objects[class_name] = (center_x, center_y, time.time())
                            # 실시간 객체 안내
                            self.speak(message, priority=True, use_pyttsx3=True)  # 메시지를 음성 출력 큐에 추가

                        # 볼륨 원래대로
                        self.restore_volume()

                # 얼굴 인식 추가
                face_locations = face_recognition.face_locations(color_image)
                if len(face_locations) > 0:
                    face_encodings = face_recognition.face_encodings(color_image, known_face_locations=face_locations)
                    if len(face_encodings) > 0:  # 인코딩이 존재하는지 확인
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            matches = face_recognition.compare_faces(list(self.face_manager.embeddings.values()), face_encoding)
                            face_distances = face_recognition.face_distance(list(self.face_manager.embeddings.values()), face_encoding)
                            if len(face_distances) > 0:  # 얼굴 거리가 존재하는지 확인
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    name = list(self.face_manager.embeddings.keys())[best_match_index]
                                    name = os.path.basename(name).split('_')[0]  # 파일 경로에서 이름 추출
                                    current_time = time.time()  # 현재 시간
                                    if name in self.recognized_faces:
                                        last_recognized_time = self.recognized_faces[name]
                                        if current_time - last_recognized_time < 60:  # 60초 이내에 다시 인식된 경우 무시
                                            continue
                                    self.recognized_faces[name] = current_time  # 현재 시간을 기록
                                    center_x = int((left + right) / 2)
                                    center_y = int((top + bottom) / 2)
                                    if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                                        distance = depth_image[center_y, center_x] * 0.001
                                        if distance < 1 or distance > 10:  # 1미터에서 10미터 사이가 아닌 경우 무시
                                            continue
                                    else:
                                        continue
                                    distance_rounded = round(distance, 2)
                                    message = f"{name}씨가 {distance_rounded} 미터 앞에 있습니다."

                                    if self.ai.is_recording:  # 녹음 중일 경우
                                        self.ai.text_to_voice(message)  # 메시지 전달
                                    else:  # 녹음 중이 아닐 경우
                                        # 볼륨 낮추기
                                        self.lower_volume()

                                        # 실시간 얼굴 인식 안내
                                        self.speak(message, priority=True, use_pyttsx3=True)

                                        # 녹음 시작 전에 음성 안내가 완료되기를 기다림
                                        while self.processing_speech:
                                            time.sleep(0.1)

                                        # 볼륨 원래대로
                                        self.restore_volume()

                                        # 이전 대화 내용 재생
                                        folder_path = os.path.join(self.face_manager.data_folder, name)
                                        self.ai.current_folder = folder_path  # 현재 대화 중인 폴더 설정
                                        self.ai.play_previous_conversation(folder_path)

                                        # 녹음 시작
                                        self.ai.text_to_voice("대화 녹음을 시작할까요?")
                                        response = self.ai.voice_to_text()
                                        if response == "시작":
                                            if not self.ai.is_recording:
                                                self.ai.is_recording = True
                                                self.ai.start_recording(folder_path)
                                                Thread(target=self.ai.monitor_recording, args=(folder_path,)).start()

    def lower_volume(self):
        self.original_volume = pygame.mixer.music.get_volume()
        pygame.mixer.music.set_volume(self.original_volume * 0.2)

    def restore_volume(self):
        pygame.mixer.music.set_volume(self.original_volume)

    def has_object_moved(self, class_name, new_x, new_y, threshold=20):
        """
        객체가 이동했는지 여부를 확인하는 함수
        """
        if class_name not in self.detected_objects:
            return True
        old_x, old_y, _ = self.detected_objects[class_name]
        distance = np.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2)
        return distance > threshold

    def update(self):
        # 업데이트 주기에 맞춰 프레임을 갱신
        if self.frame is not None:
            self.update_frame_ui(self.label, self.frame)
        QTimer.singleShot(self.delay * 1000, self.update)

    # 프레임을 업데이트하는 함수
    def update_frame(self):
        depth_image, color_image = self.depth_camera.get_frame()  # Depth 카메라에서 프레임 가져오기
        if color_image is None:  # 컬러 이미지가 없는 경우 종료
            return

        self.frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # 색상 이미지를 RGB로 변환
        self.update_frame_ui(self.label, self.frame)  # 컬러 이미지를 label에 표시

        # Depth 이미지를 컬러맵으로 변환하여 label_2에 표시
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        self.update_frame_ui(self.label_2, depth_colormap)

        # 객체 감지 수행
        self.video_pred()

    # 큐에 저장된 메시지를 음성 출력하는 함수
    def process_speech_queue(self):
        while True:
            priority, message, use_pyttsx3 = self.speech_queue.get()
            if message is None:
                break
            with self.speech_lock:
                if not self.is_speech_paused:  # 음성 안내가 일시정지되지 않았을 때만 실행
                    self.processing_speech = True
                    print(f"감지된 객체 : {message}")
                    if use_pyttsx3:
                        self.engine.say(message)
                        self.engine.runAndWait()
                    else:
                        self.ai.text_to_voice(message)
                        self.ai.text_to_voice_done.wait()  # 음성 출력 완료 시그널 대기
                    self.processing_speech = False

    # 음성 출력을 위한 함수
    def speak(self, text, priority=False, use_pyttsx3=False):
        self.speech_queue.put((priority, text, use_pyttsx3))

    def closeEvent(self, event):
        self.timer.stop()  # 타이머를 멈춥니다.
        self.depth_camera.stop_pipeline()
        self.speech_queue.put((False, None, False))
        event.accept()

    # label에 이미지를 업데이트하는 함수
    def update_frame_ui(self, label, img):
        qimage = self.convert2QImage(img)  # 이미지를 QImage로 변환
        pixmap = QPixmap.fromImage(qimage)  # QImage를 QPixmap으로 변환
        label.setPixmap(pixmap)  # label에 QPixmap 설정
        label.setScaledContents(True)  # label의 내용이 스케일되도록 설정

if __name__ == "__main__":
    app = QApplication(sys.argv)  # QApplication 객체 생성
    window = MainWindow()  # MainWindow 객체 생성
    window.show()  # 창 표시
    sys.exit(app.exec())  # 애플리케이션 실행 및 종료 시 상태 코드 반환
