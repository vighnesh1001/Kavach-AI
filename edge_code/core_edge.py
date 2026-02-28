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
        # AI Setup
        self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Buffer Setup
        self.frame_buffer = collections.deque(maxlen=int(FPS * 11)) 
        self.is_processing_accident = False
        self.latest_bbox = None
        
        # gRPC Setup
        channel = grpc.insecure_channel(f'{SERVER_IP}:50051')
        self.stub = accident_pb2_grpc.AccidentReporterStub(channel)

    def send_fast_alert(self, frame, ts):
        try:
            # Optimize: compress slightly to ensure it fits gRPC deadline
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            payload = accident_pb2.AccidentData(
                frame=buffer.tobytes(), latitude=LAT, longitude=LON,
                camera_id=CAMERA_ID, timestamp=ts
            )
            self.stub.SendAlert(payload, timeout=5) # Increased timeout
            print(" >>> [FAST PATH] gRPC Alert Sent.")
        except Exception as e:
            print(f" [!] gRPC Failed: {e}")

    def capture_aftermath_and_send(self):
        print(" >>> [SLOW PATH] Waiting 5s for aftermath...")
        time.sleep(5)
        
        evidence_frames = list(self.frame_buffer)
        ts = int(time.time())
        filename = f"{BUFFER_DIR}/evidence_{ts}.mp4"
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (640, 480))
        
        start_idx = max(0, len(evidence_frames) - int(FPS * 10))
        for i in range(start_idx, len(evidence_frames)):
            out.write(cv2.resize(evidence_frames[i], (640, 480)))
        out.release()

        print(f" >>> [SLOW PATH] Pushing SRT Clip...")
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
            self.frame_buffer.append(frame.copy())

            current_time = time.time()
            # Run AI every 0.3 seconds instead of 0.5 for better responsiveness
            if current_time - last_ai_time >= 0.3: 
                last_ai_time = current_time
                
                # Inference Preprocessing
                blob = cv2.resize(frame, (640, 640))
                blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB) # YOLO expects RGB
                blob = blob.astype(np.float32) / 255.0
                blob = np.transpose(blob, (2, 0, 1))
                blob = np.expand_dims(blob, axis=0)
                
                outputs = self.session.run(None, {self.input_name: blob})[0][0]
                
                # YOLOv8 Transpose logic (84 x 8400 -> 8400 x 84)
                if outputs.shape[0] < outputs.shape[1]:
                    outputs = outputs.T

                # Find detection with highest confidence
                # Index 4 is usually the confidence score in YOLOv8
                scores = outputs[:, 4]
                max_idx = np.argmax(scores)
                max_score = scores[max_idx]
                
                if max_score > 0.40:
                    # Get BBox [cx, cy, w, h]
                    det = outputs[max_idx]
                    xc, yc, w, h = det[:4]
                    
                    # Scale to original resolution
                    x1 = int((xc - w/2) * (w_orig / 640))
                    y1 = int((yc - h/2) * (h_orig / 640))
                    x2 = int((xc + w/2) * (w_orig / 640))
                    y2 = int((yc + h/2) * (h_orig / 640))
                    
                    self.latest_bbox = (x1, y1, x2, y2, max_score)

                    if not self.is_processing_accident:
                        self.is_processing_accident = True
                        print(f"!!! ACCIDENT DETECTED ({max_score:.2f}) !!!")
                        
                        # Prepare snapshot with box drawn
                        alert_frame = frame.copy()
                        cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        threading.Thread(target=self.send_fast_alert, args=(alert_frame, time.strftime('%H:%M:%S'))).start()
                        threading.Thread(target=self.capture_aftermath_and_send).start()
                else:
                    # Optional: Keep the box for a few frames even if detection drops
                    self.latest_bbox = None

            # --- DRAWING (Always runs at 20 FPS) ---
            if self.latest_bbox:
                x1, y1, x2, y2, score = self.latest_bbox
                # Thick red box for visibility
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                # Label with background for readability
                label = f"ACCIDENT: {int(score*100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Kaval Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = KavalAgent()
    agent.run()