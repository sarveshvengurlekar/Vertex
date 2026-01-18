import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

class WebcamManager:
    def __init__(self, model_path="model/20_best.pt"):
        self.model = YOLO(model_path)
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.last_frame = None

    def start(self):
        with self.lock:
            if not self.is_running:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    return False, "Cannot open webcam"
                
                # Request higher resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                self.is_running = True
                return True, "Webcam started"
            return True, "Webcam already running"

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            return True, "Webcam stopped"

    def get_frames(self):
        while True:
            with self.lock:
                if not self.is_running or self.cap is None:
                    break
                
                success, frame = self.cap.read()
                if not success:
                    break

            # YOLO inference
            results = self.model.predict(
                source=frame,
                conf=0.25,
                iou=0.7,
                verbose=False
            )

            # 3. Result frame
            result_frame = results[0].plot(labels=True, conf=True)
            
            # Encode as JPEG
            ret, buffer = cv2.imencode(".jpg", result_frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   frame_bytes + b"\r\n")
            
            # Control frame rate slightly
            time.sleep(0.01)

# Global singleton
webcam_manager = WebcamManager()
