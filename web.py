import os
import threading
import time
import cv2
import subprocess
from datetime import datetime
from flask import Flask, render_template, Response, send_from_directory

app = Flask(__name__)

TEMP_DIR = 'temp_outputs'
SAVE_DIR = 'video_outputs'
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
terminate_event = threading.Event()
video_streamer = None  # VideoStreamer 객체를 전역 변수로 선언

class VideoStreamer:
    def __init__(self, depth_camera):
        self.depth_camera = depth_camera
        self.recording = False
        self.lock = threading.Lock()

    def get_next_filename(self, prefix):
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}{current_time}.mp4"

    def save_video(self):
        while not terminate_event.is_set():
            if self.recording:
                temp_video_filename = os.path.join(TEMP_DIR, self.get_next_filename('TO_'))
                reencoded_video_filename = os.path.join(SAVE_DIR, self.get_next_filename(''))

                # Recording settings
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30  # Ensure this matches your camera's frame rate.
                out = cv2.VideoWriter(temp_video_filename, fourcc, fps, (640, 480))

                start_time = time.time()
                frame_count = 0

                while self.recording and (time.time() - start_time) < 120:  # 2 minutes recording
                    with self.lock:
                        depth_image, color_image = self.depth_camera.get_frame()
                        if color_image is not None:
                            out.write(color_image)
                            frame_count += 1
                    elapsed_time = time.time() - start_time
                    expected_frame_count = int(elapsed_time * fps)
                    while frame_count < expected_frame_count:
                        out.write(color_image)
                        frame_count += 1

                out.release()
                print(f"Recorded {frame_count} frames in {temp_video_filename}")

                # Use ffmpeg to re-encode video with the correct settings
                ffmpeg_reencode_command = [
                    'ffmpeg',
                    '-y',
                    '-i', temp_video_filename,
                    '-r', str(fps),  # Set frame rate
                    '-vcodec', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-acodec', 'aac',
                    reencoded_video_filename
                ]

                subprocess.run(ffmpeg_reencode_command)
                print(f"Re-encoded video saved to {reencoded_video_filename}")
                os.remove(temp_video_filename)

    def generate_frames(self):
        while not terminate_event.is_set():
            with self.lock:
                depth_image, color_image = self.depth_camera.get_frame()
                if color_image is None:
                    continue

                ret, buffer = cv2.imencode('.jpg', color_image)
                self.frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(video_streamer.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".mp4")]
    return render_template('index2.html', files=files)

@app.route('/video_outputs/<filename>')
def video(filename):
    return send_from_directory(SAVE_DIR, filename)

def start_recording():
    global video_streamer
    video_streamer.recording = True

def stop_recording():
    global video_streamer
    video_streamer.recording = False

def signal_handler(sig, frame):
    global terminate_event
    terminate_event.set()
    stop_recording()
    time.sleep(1)  # 현재 녹음이 완료될 때까지 잠시 기다려 주세요.

def create_video_streamer(depth_camera):
    global video_streamer
    video_streamer = VideoStreamer(depth_camera)
    threading.Thread(target=video_streamer.save_video, daemon=True).start()

if __name__ == '__main__':
    from cam import Depth
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    depth_camera = Depth()
    depth_camera.start_pipeline()
    create_video_streamer(depth_camera)
    
    app.run(host='0.0.0.0', port=5000)
