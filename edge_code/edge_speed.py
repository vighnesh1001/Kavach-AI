import cv2
import onnxruntime as ort
import numpy as np
import grpc
import time
import threading
import queue
import os

# --- 1. CONIGURATION ---
SERVER_IP = "172.20.25.43"  # <--- Change this to your Laptop's Hotspot IP
MODEL_PATH = "best.onnx"
CAMERA_ID = "PI_CAM_AMRITA_01"
LAT, LON = 9.0939, 76.4918

# Import gRPC stubs
import accident_pb2
import accident_pb2_grpc

# --- 2. MULTI-THREADING SETUP ---
# This queue will hold the latest frame for the AI to process
processing_queue = queue.Queue(maxsize=1) 
latest_max_score = 0.0

# --- 3. NETWORK LOGIC (Fast Path) ---
def send_fast_path_alert(frame, timestamp):
    """Network worker: Sends bytes without blocking the main loop."""
    try:
        with grpc.insecure_channel(f'{SERVER_IP}:50051') as channel:
            stub = accident_pb2_grpc.AccidentReporterStub(channel)
            
            # Encode to JPEG (reduced quality to 70% for faster transmission)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            
            payload = accident_pb2.AccidentData(
                frame=buffer.tobytes(),
                latitude=LAT,
                longitude=LON,
                camera_id=CAMERA_ID,
                timestamp=timestamp
            )
            
            # Long timeout to handle potential Wi-Fi jitter
            response = stub.SendAlert(payload, timeout=10)
            print(f" >>> [NETWORK SUCCESS] Alert sent at {timestamp}")
    except Exception as e:
        print(f" [!] Network Error: {e}")

# --- 4. AI INTERFACING THREAD ---
def ai_worker():
    """Background thread for YOLO Inference."""
    global latest_max_score
    
    print("AI Worker Thread Initializing...")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    while True:
        # Get the latest frame from the queue (blocks until a frame is available)
        frame = processing_queue.get()
        if frame is None: break
        
        h_orig, w_orig = frame.shape[:2]
        
        # Preprocessing (640x640)
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        blob = np.expand_dims(img, axis=0)

        # Run Inference
        outputs = session.run(None, {input_name: blob})
        output = outputs[0][0] # Expected shape (300, 6) or (8400, 6)

        # Process detections
        accident_found = False
        max_s = 0
        
        # Check if output needs transpose (handles both raw and NMS-integrated models)
        if output.shape[0] < output.shape[1]: output = output.T

        for det in output:
            score = det[4]
            if score > max_s: max_s = score
            
            if score > 0.35: # Trigger threshold
                accident_found = True
        
        latest_max_score = max_s

        if accident_found:
            ts = time.strftime('%H:%M:%S', time.localtime())
            print(f"*** ACCIDENT DETECTED ({max_s:.2f}) ***")
            # Start a one-off thread to send the data so the AI keeps moving
            threading.Thread(target=send_fast_path_alert, args=(frame.copy(), ts)).start()
        
        # Yield to prevent CPU overheating; processes at roughly 1-2 FPS
        time.sleep(0.5)

# --- 5. MAIN CAMERA LOOP ---
def run_main():
    global latest_max_score
    cap = cv2.VideoCapture(0)
    
    # Performance hack: Capture at lower res to reduce Pi memory load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Start the AI thread
    threading.Thread(target=ai_worker, daemon=True).start()
    
    print("System Online. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Try to put frame in queue; if full, replace old frame with newest
        if processing_queue.full():
            try: processing_queue.get_nowait()
            except queue.Empty: pass
        processing_queue.put(frame)

        # Visual Feedback
        status_color = (0, 0, 255) if latest_max_score > 0.35 else (0, 255, 0)
        cv2.putText(frame, f"System Active | Max Score: {latest_max_score:.2f}", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow("Kaval Surveillance - Edge Device", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    processing_queue.put(None) # Signal AI thread to stop
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()