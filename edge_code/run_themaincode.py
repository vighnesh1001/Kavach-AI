import cv2
import onnxruntime as ort
import numpy as np
import grpc
import time
import threading
import os

# --- GENERATED STUBS (Ensure these files are in the same folder) ---
import accident_pb2
import accident_pb2_grpc

# --- CONFIGURATION ---
MODEL_PATH = "best.onnx"
SERVER_IP = "10.84.200.233"  # <--- DOUBLE CHECK THIS IP
CAMERA_ID = "PI_CAM_AMRITA_01"
LAT, LON = 9.0939, 76.4918 

class KavalAgent:
    def __init__(self):
        # AI Engine Initialization
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Shared State
        self.current_frame = None
        self.active_bbox = None  # Format: [x1, y1, x2, y2, score]
        self.is_sending = False
        self.running = True

        # Persistent gRPC Channel
        self.channel = grpc.insecure_channel(f'{SERVER_IP}:50051')
        self.stub = accident_pb2_grpc.AccidentReporterStub(self.channel)

    def send_alert_async(self, frame, score, ts):
        """BACKGROUND THREAD: Sends byte stream without stopping the camera"""
        try:
            # COMPRESSION: High compression (40) ensures the byte stream is small
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            
            payload = accident_pb2.AccidentData(
                frame=buffer.tobytes(), 
                latitude=LAT, longitude=LON,
                camera_id=CAMERA_ID, timestamp=ts
            )
            # Long timeout (15s) to prevent Deadline Exceeded
            self.stub.SendAlert(payload, timeout=15)
            print(f" >>> [FAST PATH] Alert Sent (Score: {score:.2f})")
        except grpc.RpcError as e:
            print(f" [!] Network Busy: {e.code()}")
        except Exception as e:
            print(f" [!] Alert Thread Error: {e}")
        finally:
            self.is_sending = False

    def ai_worker_thread(self):
        """BACKGROUND THREAD: Performs AI inference"""
        while self.running:
            if self.current_frame is not None:
                try:
                    work_frame = self.current_frame.copy()
                    h_orig, w_orig = work_frame.shape[:2]
                    
                    # Preprocess
                    blob = cv2.resize(work_frame, (640, 640))
                    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
                    blob = blob.astype(np.float32) / 255.0
                    blob = np.transpose(blob, (2, 0, 1))
                    blob = np.expand_dims(blob, axis=0)
                    
                    # Inference
                    outputs = self.session.run(None, {self.input_name: blob})[0][0]
                    
                    # Find highest score
                    max_s = 0
                    best_det = None
                    for d in outputs:
                        if d[4] > max_s:
                            max_s = d[4]; best_det = d
                    
                    if max_s > 0.35:
                        # SCALE COORDINATES: 640 -> Original
                        xc, yc, w, h = best_det[:4]
                        x1 = int((xc - w/2) * (w_orig / 640))
                        y1 = int((yc - h/2) * (h_orig / 640))
                        x2 = int((xc + w/2) * (w_orig / 640))
                        y2 = int((yc + h/2) * (h_orig / 640))
                        
                        self.active_bbox = [x1, y1, x2, y2, max_s]
                        
                        # Trigger Network if not currently busy
                        if not self.is_sending:
                            self.is_sending = True
                            ts = time.strftime('%H:%M:%S')
                            threading.Thread(target=self.send_alert_async, 
                                             args=(work_frame, max_s, ts)).start()
                    else:
                        self.active_bbox = None
                except Exception as e:
                    print(f"AI Worker Error: {e}")
            
            time.sleep(0.1) # Throttle AI to ~10 FPS to save Pi CPU

    def run(self):
        # Environment fix for Wayland/Qt
        os.environ["QT_QPA_PLATFORM"] = "xcb" 
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start background AI
        threading.Thread(target=self.ai_worker_thread, daemon=True).start()
        print(f"--- Kaval System Online: Targeting {SERVER_IP} ---")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            self.current_frame = frame

            # --- DRAWING (Full FPS, Zero Lag) ---
            if self.active_bbox:
                x1, y1, x2, y2, score = self.active_bbox
                # Thick Red Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                # Label with background
                label = f"ACCIDENT {int(score*100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Kaval Feed - RealTime", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.channel.close()

if __name__ == "__main__":
    agent = KavalAgent()
    agent.run()