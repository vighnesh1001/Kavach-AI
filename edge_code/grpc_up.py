import cv2
import onnxruntime as ort
import numpy as np
import grpc
import time
import threading
import os

# --- GENERATED STUBS ---
import accident_pb2
import accident_pb2_grpc

# --- CONFIGURATION ---
MODEL_PATH = "best.onnx"
SERVER_IP = "10.84.200.233"  # Updated based on your logs
CAMERA_ID = "PI_CAM_AMRITA_01"
LAT, LON = 9.0939, 76.4918 

class KavalAgent:
    def __init__(self):
        # AI Init
        self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Shared Variables
        self.current_frame = None
        self.active_bbox = None  # [x1, y1, x2, y2, score]
        self.is_sending = False
        
        # gRPC Setup
        channel = grpc.insecure_channel(f'{SERVER_IP}:50051')
        self.stub = accident_pb2_grpc.AccidentReporterStub(channel)

    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    def send_alert_worker(self, frame, score, ts):
        """FIRE-AND-FORGET: Sends data without blocking the camera"""
        try:
            # OPTIMIZATION: Reduce quality to 60% for fast transmission
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            payload = accident_pb2.AccidentData(
                frame=buffer.tobytes(), latitude=LAT, longitude=LON,
                camera_id=CAMERA_ID, timestamp=ts
            )
            # Increased timeout to 20s
            self.stub.SendAlert(payload, timeout=20)
            print(f" >>> [FAST PATH] Alert Sent Successfully (Score: {score:.2f})")
        except Exception as e:
            print(f" [!] gRPC Fail: {e}")
        finally:
            self.is_sending = False

    def ai_worker(self):
        """Thread: Independent AI processing"""
        while True:
            if self.current_frame is not None:
                h_orig, w_orig = self.current_frame.shape[:2]
                blob = self.preprocess(self.current_frame)
                
                outputs = self.session.run(None, {self.input_name: blob})[0][0]
                
                max_s = 0
                best_det = None
                for d in outputs:
                    if d[4] > max_s:
                        max_s = d[4]
                        best_det = d
                
                if max_s > 0.35:
                    # Scaling Logic
                    xc, yc, w, h = best_det[:4]
                    x1 = int((xc - w/2) * (w_orig / 640))
                    y1 = int((yc - h/2) * (h_orig / 640))
                    x2 = int((xc + w/2) * (w_orig / 640))
                    y2 = int((yc + h/2) * (h_orig / 640))
                    
                    self.active_bbox = [x1, y1, x2, y2, max_s]
                    
                    # Trigger alert in a NEW thread so AI loop continues
                    if not self.is_sending:
                        self.is_sending = True
                        ts = time.strftime('%H:%M:%S')
                        threading.Thread(target=self.send_alert_worker, 
                                         args=(self.current_frame.copy(), max_s, ts)).start()
                else:
                    self.active_bbox = None
            
            time.sleep(0.1) # Throttle AI to ~10 FPS to save Pi CPU

    def run(self):
        # To fix the 'wayland' and 'fonts' errors in terminal
        os.environ["QT_QPA_PLATFORM"] = "xcb" 
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        threading.Thread(target=self.ai_worker, daemon=True).start()
        print(f"Kaval Edge Online. Target: {SERVER_IP}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            self.current_frame = frame

            # --- DRAWING (Runs at full speed, no lag) ---
            if self.active_bbox:
                x1, y1, x2, y2, score = self.active_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                
                # Professional label
                label = f"ACCIDENT {int(score*100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Kaval Feed - Zero Lag", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = KavalAgent()
    agent.run()