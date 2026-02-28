import cv2
import onnxruntime as ort
import numpy as np
import grpc
import time
import threading
import os
import subprocess
import collections

# --- CONFIGURATION ---
MODEL_PATH = "best.onnx"
SERVER_IP = "172.20.25.43"
CAMERA_ID = "PI_CAM_AMRITA_01"
LAT, LON = 9.0939, 76.4918
FPS = 20.0 

import accident_pb2
import accident_pb2_grpc

BUFFER_DIR = "/dev/shm/kaval_buffer"
if not os.path.exists(BUFFER_DIR): os.makedirs(BUFFER_DIR)

class KavalAgent:
    def __init__(self):
        # AI Setup
        self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Buffer & Shared State
        self.frame_buffer = collections.deque(maxlen=int(FPS * 10))
        self.current_frame = None
        self.latest_bbox = None  # Format: (x1, y1, x2, y2, score)
        self.is_processing_accident = False
        
        # gRPC Setup
        self.channel = grpc.insecure_channel(f'{SERVER_IP}:50051')
        self.stub = accident_pb2_grpc.AccidentReporterStub(self.channel)

    def ai_inference_worker(self):
        """Threaded worker to handle AI without lagging the camera feed"""
        while True:
            if self.current_frame is not None:
                # 1. Take a snapshot to process
                frame_to_proc = self.current_frame.copy()
                h_orig, w_orig = frame_to_proc.shape[:2]
                
                # 2. Preprocess
                blob = cv2.resize(frame_to_proc, (640, 640)).astype(np.float32) / 255.0
                blob = np.transpose(blob, (2, 0, 1))
                blob = np.expand_dims(blob, axis=0)
                
                # 3. Inference
                outputs = self.session.run(None, {self.input_name: blob})[0][0]
                
                # 4. Find Best Detection
                max_score = 0
                best_det = None
                for d in outputs:
                    if d[4] > max_score:
                        max_score = d[4]
                        best_det = d
                
                if max_score > 0.40:
                    # Scaling Logic for Bounding Box
                    xc, yc, w, h = best_det[:4]
                    x1 = int((xc - w/2) * (w_orig / 640))
                    y1 = int((yc - h/2) * (h_orig / 640))
                    x2 = int((xc + w/2) * (w_orig / 640))
                    y2 = int((yc + h/2) * (h_orig / 640))
                    
                    self.latest_bbox = (x1, y1, x2, y2, max_score)
                    
                    # 5. Trigger Network Alerts (Only if not already busy)
                    if not self.is_processing_accident:
                        self.is_processing_accident = True
                        ts = time.strftime('%H:%M:%S')
                        threading.Thread(target=self.send_fast_alert, args=(frame_to_proc, ts)).start()
                        threading.Thread(target=self.send_slow_video, args=(list(self.frame_buffer),)).start()
                else:
                    self.latest_bbox = None
            
            # Throttle to save Pi CPU (Inference 4 times per second)
            time.sleep(0.25)

    def send_fast_alert(self, frame, ts):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            payload = accident_pb2.AccidentData(
                frame=buffer.tobytes(), latitude=LAT, longitude=LON,
                camera_id=CAMERA_ID, timestamp=ts
            )
            self.stub.SendAlert(payload, timeout=2)
            print(" >>> [FAST PATH] Alert Sent.")
        except Exception as e:
            print(f" [!] gRPC Failed: {e}")

    def send_slow_video(self, evidence_frames):
        try:
            ts = int(time.time())
            filename = f"{BUFFER_DIR}/evidence_{ts}.mp4"
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (640, 480))
            for f in evidence_frames:
                out.write(cv2.resize(f, (640, 480)))
            out.release()

            print(f" >>> [SLOW PATH] Pushing SRT...")
            cmd = ['ffmpeg', '-re', '-i', filename, '-c', 'copy', '-f', 'mpegts', f'srt://{SERVER_IP}:9999?mode=caller']
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(filename): os.remove(filename)
        finally:
            self.is_processing_accident = False

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Start AI Worker Thread
        threading.Thread(target=self.ai_inference_worker, daemon=True).start()

        print("Edge Agent Online. Camera Running @ Full Speed.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            self.current_frame = frame
            self.frame_buffer.append(frame.copy())

            # --- DRAWING LOGIC (Main Thread = No Lag) ---
            if self.latest_bbox:
                x1, y1, x2, y2, score = self.latest_bbox
                # Draw red bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # Draw label
                label = f"ACCIDENT: {score:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Kaval Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = KavalAgent()
    agent.run()