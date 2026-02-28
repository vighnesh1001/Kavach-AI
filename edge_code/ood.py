import cv2
import onnxruntime as ort
import numpy as np
import grpc
import time
import threading
import os
import accident_pb2
import accident_pb2_grpc

# --- CONFIGURATION ---
SERVER_IP = "10.84.200.233" # Check your Laptop IP again!
CAMERA_ID = "PI_CAM_AMRITA_01"
LAT, LON = 9.0939, 76.4918 

class KavalAgent:
    def __init__(self):
        # Force CPU to avoid Conv node errors
        self.session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        self.current_frame = None
        self.active_bbox = None 
        self.running = True

        # Initial gRPC connection
        self.connect_grpc()

    def connect_grpc(self):
        """Creates a fresh channel and stub"""
        self.channel = grpc.insecure_channel(f'{SERVER_IP}:50051')
        self.stub = accident_pb2_grpc.AccidentReporterStub(self.channel)

    def send_alert_worker(self, frame, score, ts):
        """Background thread that won't crash the main app"""
        try:
            # Low quality to ensure it sends even on bad Wi-Fi
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            payload = accident_pb2.AccidentData(
                frame=buffer.tobytes(), latitude=LAT, longitude=LON,
                camera_id=CAMERA_ID, timestamp=ts
            )
            # Short timeout so it doesn't hang
            self.stub.SendAlert(payload, timeout=3)
            print(f" >>> [SUCCESS] Alert Sent (Score: {score:.2f})")
        except grpc.RpcError as e:
            # If unavailable, don't crash, just log and move on
            print(f" [!] Server Offline: {e.code()}")
        except Exception as e:
            print(f" [!] Alert Thread Error: {e}")

    def ai_worker(self):
        while self.running:
            if self.current_frame is not None:
                try:
                    # AI Processing logic remains here...
                    # (Resizing, Inference, etc.)
                    # If accident detected:
                    # threading.Thread(target=self.send_alert_worker, ...).start()
                    pass 
                except Exception:
                    pass
            time.sleep(0.1)

    def run(self):
        # Environment fix for Qt errors
        os.environ["QT_QPA_PLATFORM"] = "xcb" 
        cap = cv2.VideoCapture(0)
        
        # ... Rest of your run loop ...