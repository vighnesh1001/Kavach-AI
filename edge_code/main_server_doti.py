import cv2
import onnxruntime as ort
import numpy as np
import grpc
import time
import threading
import accident_pb2
import accident_pb2_grpc

# --- CONFIGURATION ---
MODEL_PATH = "best.onnx"
SERVER_IP = "10.84.206.233" 
CAMERA_ID = "PI_CAM_AMRITA_01"
LAT, LON = 9.0939, 76.4918 

class KavalAgent:
    def __init__(self):
        # Initialize AI
        self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Thread-safe variables
        self.latest_frame = None
        self.latest_bbox = None  # Stores [x1, y1, x2, y2, score]
        self.running = True
        
        # gRPC Setup
        self.channel = grpc.insecure_channel(f'{SERVER_IP}:50051')
        self.stub = accident_pb2_grpc.AccidentReporterStub(self.channel)

    def preprocess(self, frame):
        # Use INTER_NEAREST for faster resizing on Pi
        img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    def send_alert_async(self, frame, score):
        """Background thread for network calls"""
        try:
            ts = time.strftime('%H:%M:%S')
            # Compress image slightly to speed up transmission
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            payload = accident_pb2.AccidentData(
                frame=buffer.tobytes(), latitude=LAT, longitude=LON,
                camera_id=CAMERA_ID, timestamp=ts
            )
            self.stub.SendAlert(payload, timeout=5)
            print(f" >>> [FAST PATH] Alert Sent (Score: {score:.2f})")
        except Exception as e:
            print(f" [!] gRPC Fail: {e}")

    def inference_worker(self):
        """Background thread for AI logic"""
        while self.running:
            if self.latest_frame is not None:
                frame_to_proc = self.latest_frame.copy()
                h_orig, w_orig = frame_to_proc.shape[:2]
                
                # 1. Run Model
                blob = self.preprocess(frame_to_proc)
                outputs = self.session.run(None, {self.input_name: blob})[0][0]
                
                # 2. Parse Results
                max_score = 0
                best_det = None
                for d in outputs:
                    if d[4] > max_score:
                        max_score = d[4]
                        best_det = d
                
                if max_score > 0.25:
                    # Scaling Logic
                    coords = best_det[:4]
                    if np.max(coords) <= 1.01:
                        x1, y1, x2, y2 = int(coords[0]*w_orig), int(coords[1]*h_orig), int(coords[2]*w_orig), int(coords[3]*h_orig)
                    else:
                        x1, y1, x2, y2 = int(coords[0]*w_orig/640), int(coords[1]*h_orig/640), int(coords[2]*w_orig/640), int(coords[3]*h_orig/640)
                    
                    self.latest_bbox = [x1, y1, x2, y2, max_score]
                    
                    # 3. Trigger Async Alert
                    threading.Thread(target=self.send_alert_async, args=(frame_to_proc, max_score)).start()
                else:
                    self.latest_bbox = None
                
                # Throttle AI slightly so the Pi CPU doesn't choke (approx 5-10 times per sec)
                time.sleep(0.1)

    def run(self):
        cap = cv2.VideoCapture(0)
        # Force a specific resolution to save bandwidth
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start AI thread
        threading.Thread(target=self.inference_worker, daemon=True).start()
        
        print(f"Edge Agent Online. Targeting {SERVER_IP}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Store the current frame for the AI thread
            self.latest_frame = frame.copy()

            # --- DRAWING (Runs at full webcam speed) ---
            if self.latest_bbox:
                x1, y1, x2, y2, score = self.latest_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, f"ACCIDENT {score:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.imshow("Kaval Feed - Zero Lag", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = KavalAgent()
    agent.run()