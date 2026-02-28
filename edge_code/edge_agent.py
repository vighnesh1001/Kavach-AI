import cv2
import onnxruntime as ort
import numpy as np
import grpc
import time

# --- CONFIGURATION ---
MODEL_PATH = "best.onnx"
SERVER_IP = "10.84.200.233" # <--- MUST CHANGE THIS to your Laptop IP (e.g., 192.168.1.45)
CAMERA_ID = "PI_CAM_AMRITA_01"
LAT, LON = 9.0939, 76.4918 

session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def send_fast_path_alert(frame, timestamp):
    if "x.x" in SERVER_IP:
        print(" [SKIP] gRPC not sent: You haven't set the Laptop IP yet!")
        return
    try:
        with grpc.insecure_channel(f'{SERVER_IP}:50051') as channel:
            stub = accident_pb2_grpc.AccidentReporterStub(channel)
            _, buffer = cv2.imencode('.jpg', frame)
            payload = accident_pb2.AccidentData(
                frame=buffer.tobytes(), latitude=LAT, longitude=LON,
                camera_id=CAMERA_ID, timestamp=timestamp
            )
            stub.SendAlert(payload, timeout=15)
            print(f" >>> [FAST PATH] Payload sent to {SERVER_IP}")
    except Exception as e:
        print(f" [ERROR] gRPC Failed: {e}")

def run_edge_agent():
    cap = cv2.VideoCapture(0)
    last_process_time = 0
    
    print(f"Edge Agent Online. Targeting Server: {SERVER_IP}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()
        
        if current_time - last_process_time >= 1.0:
            last_process_time = current_time
            
            # 1. Preprocessing
            img_input = cv2.resize(frame, (640, 640))
            img_blob = img_input.astype(np.float32) / 255.0
            img_blob = np.transpose(img_blob, (2, 0, 1))
            blob = np.expand_dims(img_blob, axis=0)

            # 2. Inference
            outputs = session.run(None, {input_name: blob})
            output = outputs[0][0] # (300, 6)

            accident_detected = False
            h_orig, w_orig = frame.shape[:2]
            max_score = 0

            for detection in output:
                score = detection[4]
                if score > max_score: max_score = score

                if score > 0.25: # Set to 0.25 to catch that 0.30 detection you had
                    accident_detected = True
                    
                    # 3. ROBUST COORDINATE SCALING
                    # If coordinates are < 1.0, they are normalized. Multiply by full width/height.
                    # If coordinates are > 1.0, they are pixel values based on 640.
                    coords = detection[:4]
                    if np.max(coords) <= 1.01: 
                        # Normalized format
                        x1 = int(coords[0] * w_orig)
                        y1 = int(coords[1] * h_orig)
                        x2 = int(coords[2] * w_orig)
                        y2 = int(coords[3] * h_orig)
                    else:
                        # Pixel format (640 scale)
                        x1 = int(coords[0] * w_orig / 640)
                        y1 = int(coords[1] * h_orig / 640)
                        x2 = int(coords[2] * w_orig / 640)
                        y2 = int(coords[3] * h_orig / 640)

                    # Draw thick box and text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cv2.putText(frame, f"ACCIDENT {score:.2f}", (x1, y1-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            print(f"Max Score: {max_score:.4f} | Detected: {accident_detected}")

            if accident_detected:
                ts = time.strftime('%H:%M:%S', time.localtime())
                send_fast_path_alert(frame, ts)

        cv2.imshow("Kaval Agent - Edge Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import accident_pb2, accident_pb2_grpc # Ensure these are imported
    run_edge_agent()