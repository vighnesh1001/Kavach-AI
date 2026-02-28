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
SERVER_IP = "10.112.88.233"
CAMERA_ID = "PI_CAM_AMRITA_01"
LAT, LON = 9.0939, 76.4918
FPS = 20.0 

import accident_pb2
import accident_pb2_grpc

BUFFER_DIR = "/dev/shm/kaval_buffer"
if not os.path.exists(BUFFER_DIR): os.makedirs(BUFFER_DIR)

class KavalAgent:
    def __init__(self):
        self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        # Buffer stores 10 seconds total to ensure we have enough room for the full clip
        self.frame_buffer = collections.deque(maxlen=int(FPS * 11)) 
        self.is_processing_accident = False
        self.latest_bbox = None
        
        channel = grpc.insecure_channel(f'{SERVER_IP}:50051')
        self.stub = accident_pb2_grpc.AccidentReporterStub(channel)

    def send_fast_alert(self, frame, ts):
        """gRPC: Immediate Path"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            payload = accident_pb2.AccidentData(
                frame=buffer.tobytes(), latitude=LAT, longitude=LON,
                camera_id=CAMERA_ID, timestamp=ts
            )
            self.stub.SendAlert(payload, timeout=2)
            print(" >>> [FAST PATH] gRPC Alert Sent.")
        except Exception as e:
            print(f" [!] gRPC Failed: {e}")

    def capture_aftermath_and_send(self):
        """Wait 5 seconds, then grab the 10s window from the buffer"""
        print(" >>> [SLOW PATH] Waiting 5s to capture aftermath...")
        time.sleep(5) # Wait for the 'future' 5 seconds to be recorded into the buffer
        
        # Now the buffer contains [5s before, 1s detection, 5s after]
        evidence_frames = list(self.frame_buffer)
        
        ts = int(time.time())
        filename = f"{BUFFER_DIR}/evidence_{ts}.mp4"
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (640, 480))
        
        # Take the last 10 seconds of frames from the buffer
        start_idx = max(0, len(evidence_frames) - int(FPS * 10))
        for i in range(start_idx, len(evidence_frames)):
            out.write(cv2.resize(evidence_frames[i], (640, 480)))
        out.release()

        print(f" >>> [SLOW PATH] Pushing 10s SRT Clip...")
        cmd = ['ffmpeg', '-re', '-i', filename, '-c', 'copy', '-f', 'mpegts', f'srt://{SERVER_IP}:9999?mode=caller']
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(filename): os.remove(filename)
        self.is_processing_accident = False

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Edge Agent Online. Monitoring...")
        last_ai_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            h_orig, w_orig = frame.shape[:2]
            # Capture the raw frame into the buffer
            self.frame_buffer.append(frame.copy())

            current_time = time.time()
            if current_time - last_ai_time >= 0.5: 
                last_ai_time = current_time
                
                # Inference
                blob = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
                blob = np.transpose(blob, (2, 0, 1))
                blob = np.expand_dims(blob, axis=0)
                outputs = self.session.run(None, {self.input_name: blob})[0][0]
                
                max_score = 0
                best_det = None
                for d in outputs:
                    if d[4] > max_score:
                        max_score = d[4]
                        best_det = d
                
                if max_score > 0.40:
                    xc, yc, w, h = best_det[:4]
                    x1, y1 = int((xc - w/2) * (w_orig / 640)), int((yc - h/2) * (h_orig / 640))
                    x2, y2 = int((xc + w/2) * (w_orig / 640)), int((yc + h/2) * (h_orig / 640))
                    self.latest_bbox = (x1, y1, x2, y2, max_score)

                    if not self.is_processing_accident:
                        self.is_processing_accident = True
                        print(f"!!! ACCIDENT DETECTED ({max_score:.2f}) !!!")
                        
                        # 1. FAST PATH: Immediate Snapshot
                        alert_frame = frame.copy()
                        cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        threading.Thread(target=self.send_fast_alert, args=(alert_frame, time.strftime('%H:%M:%S'))).start()
                        
                        # 2. SLOW PATH: Wait 5s then send 10s video
                        threading.Thread(target=self.capture_aftermath_and_send).start()
                else:
                    self.latest_bbox = None

            # Drawing Logic for webcam feed
            if self.latest_bbox:
                x1, y1, x2, y2, score = self.latest_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"ACCIDENT: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Kaval Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = KavalAgent()
    agent.run()