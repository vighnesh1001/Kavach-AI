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
MODEL_PATH = "best.onnx"
SERVER_IP = "10.84.200.233" 
CAMERA_ID = "PI_CAM_AMRITA_01"
LAT, LON = 9.0939, 76.4918 

class KavalAgent:
    def __init__(self):
        # AI Engine - Force CPU and disable some optimizations that cause the "Conv" error
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        self.current_frame = None
        self.active_bbox = None 
        self.running = True

        # PERSISTENT CHANNEL: Open once and keep open
        self.channel = grpc.insecure_channel(f'{SERVER_IP}:50051')
        self.stub = accident_pb2_grpc.AccidentReporterStub(self.channel)

    def send_alert_safe(self, frame, score, ts):
        """Thread-safe network sender"""
        try:
            # Drop quality to 50% to ensure it fits in the network buffer quickly
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            payload = accident_pb2.AccidentData(
                frame=buffer.tobytes(), latitude=LAT, longitude=LON,
                camera_id=CAMERA_ID, timestamp=ts
            )
            # Use a longer timeout for the first handshake
            self.stub.SendAlert(payload, timeout=10)
            print(f" >>> [FAST PATH] Alert Delivered (Score: {score:.2f})")
        except grpc.RpcError as e:
            print(f" [!] gRPC Network Busy: {e.code()}")
        except Exception as e:
            print(f" [!] Alert Thread Error: {e}")

    def ai_worker(self):
        while self.running:
            if self.current_frame is not None:
                try:
                    frame_to_proc = self.current_frame.copy()
                    h_orig, w_orig = frame_to_proc.shape[:2]
                    
                    # Preprocess
                    blob = cv2.resize(frame_to_proc, (640, 640))
                    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
                    blob = blob.astype(np.float32) / 255.0
                    blob = np.transpose(blob, (2, 0, 1))
                    blob = np.expand_dims(blob, axis=0)
                    
                    # Inference
                    outputs = self.session.run(None, {self.input_name: blob})[0][0]
                    
                    # Post-process
                    max_s = 0
                    best_det = None
                    for d in outputs:
                        if d[4] > max_s:
                            max_s = d[4]; best_det = d
                    
                    if max_s > 0.35:
                        xc, yc, w, h = best_det[:4]
                        x1 = int((xc - w/2) * (w_orig / 640))
                        y1 = int((yc - h/2) * (h_orig / 640))
                        x2 = int((xc + w/2) * (w_orig / 640))
                        y2 = int((yc + h/2) * (h_orig / 640))
                        self.active_bbox = [x1, y1, x2, y2, max_s]
                        
                        # Trigger alert in background
                        ts = time.strftime('%H:%M:%S')
                        threading.Thread(target=self.send_alert_safe, args=(frame_to_proc, max_s, ts)).start()
                    else:
                        self.active_bbox = None
                except Exception as e:
                    print(f"AI Loop Error: {e}")
            
            time.sleep(0.1)

    def run(self):
        # Force X11 to avoid Wayland crashes
        os.environ["QT_QPA_PLATFORM"] = "xcb" 
        cap = cv2.VideoCapture(0)
        
        threading.Thread(target=self.ai_worker, daemon=True).start()
        print(f"--- System Monitoring: {SERVER_IP} ---")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            self.current_frame = frame

            if self.active_bbox:
                x1, y1, x2, y2, score = self.active_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"ACCIDENT {int(score*100)}%", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Kaval Edge", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.channel.close()

if __name__ == "__main__":
    agent = KavalAgent()
    agent.run()