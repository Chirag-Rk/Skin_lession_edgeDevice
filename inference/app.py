# inference/app.py - Flask web application for NVIDIA Jetson Nano edge interface

import os
import sys
import time
import json
import threading
import subprocess
import traceback
from PIL import Image
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CFG, DEVICE, label_map
from inference.predict import SkinClassifier, LABEL_NAMES
from inference.grad_cam import GradCAM

# Setup directories
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)

# Global variables
camera = None
camera_mode = "usb"  # "usb", "csi", or "mock"
camera_index = 0
led_pin = 12         # Board Pin 12 default
led_enabled = False
fl_process = None
fl_logs = []
fl_active = False
current_classifier = None

# Fallback/Mock classes for GPIO and system diagnostics if not on Jetson
try:
    import Jetson.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[GPIO] Jetson.GPIO not available. Using mock GPIO mode.")

class MockGPIO:
    BOARD = "BOARD"
    OUT = "OUT"
    HIGH = "HIGH"
    LOW = "LOW"
    def setmode(self, mode): print(f"[Mock GPIO] Set mode to {mode}")
    def setup(self, pin, mode): print(f"[Mock GPIO] Setup pin {pin} as {mode}")
    def output(self, pin, val): print(f"[Mock GPIO] Output pin {pin} -> {val}")
    def cleanup(self): print("[Mock GPIO] Cleanup")

if not GPIO_AVAILABLE:
    GPIO = MockGPIO()

# Initialize GPIO Pin
def init_gpio():
    try:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(led_pin, GPIO.OUT)
        GPIO.output(led_pin, GPIO.LOW)
    except Exception as e:
        print(f"[GPIO] Setup error: {e}")

init_gpio()

# System Metrics Function
def get_system_metrics():
    metrics = {
        "cpu_usage": 0.0,
        "ram_usage": 0.0,
        "gpu_usage": 0.0,
        "temperature": 0.0,
        "jetson_detected": False
    }
    
    # 1. CPU and RAM usage via psutil if available, otherwise mock/read proc
    try:
        import psutil
        metrics["cpu_usage"] = psutil.cpu_percent()
        metrics["ram_usage"] = psutil.virtual_memory().percent
    except ImportError:
        # Simple fallback for cpu/ram
        try:
            with open("/proc/loadavg", "r") as f:
                metrics["cpu_usage"] = float(f.read().split()[0]) * 100 / os.cpu_count()
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
                total = int(lines[0].split()[1])
                free = int(lines[1].split()[1])
                metrics["ram_usage"] = ((total - free) / total) * 100
        except Exception:
            metrics["cpu_usage"] = 15.0
            metrics["ram_usage"] = 45.0

    # 2. Jetson Nano Temperature
    temp_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(temp_path):
        try:
            with open(temp_path, "r") as f:
                metrics["temperature"] = float(f.read().strip()) / 1000.0
                metrics["jetson_detected"] = True
        except Exception:
            metrics["temperature"] = 42.5
    else:
        metrics["temperature"] = 38.0

    # 3. Jetson Nano GPU load
    gpu_path = "/sys/devices/gpu.0/load"
    if os.path.exists(gpu_path):
        try:
            with open(gpu_path, "r") as f:
                metrics["gpu_usage"] = float(f.read().strip()) / 10.0  # value is out of 1000
                metrics["jetson_detected"] = True
        except Exception:
            metrics["gpu_usage"] = 0.0
    else:
        # Check if CUDA model is loaded and run a mock number or cuda check
        if DEVICE.type == "cuda":
            metrics["gpu_usage"] = 25.0
        else:
            metrics["gpu_usage"] = 0.0
            
    return metrics

# Initialize Classifier
def get_classifier():
    global current_classifier
    if current_classifier is None:
        # Instantiate classifier
        current_classifier = SkinClassifier(device=DEVICE)
    return current_classifier

# GStreamer pipeline for Jetson CSI camera
def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, display_width=640, display_height=480, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

# Camera Manager
def get_camera():
    global camera, camera_mode, camera_index
    if camera is None:
        if camera_mode == "csi":
            pipeline = gstreamer_pipeline(sensor_id=camera_index)
            print(f"[Camera] Opening CSI Camera with pipeline: {pipeline}")
            camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        elif camera_mode == "usb":
            print(f"[Camera] Opening USB Camera index {camera_index}")
            camera = cv2.VideoCapture(camera_index)
            
        # Verify if camera is successfully opened
        if camera is not None and camera.isOpened():
            # Set resolution for USB camera
            if camera_mode == "usb":
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            print("[Camera] Failed to open physical camera. Falling back to Mock mode.")
            camera_mode = "mock"
            camera = None
            
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        print("[Camera] Physical camera released.")

# Helper to generate synthetic lesion image for Mock Camera
def generate_mock_image(time_offset=0):
    # Create blank image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw background (skin colored)
    img[:, :] = [180, 210, 235]  # BGR skin tone
    
    # Draw lesion (dark brown blob)
    center_x = int(320 + np.sin(time_offset) * 10)
    center_y = int(240 + np.cos(time_offset) * 10)
    
    # Main lesion body
    cv2.circle(img, (center_x, center_y), 60, (50, 70, 90), -1)
    
    # Irregular borders
    cv2.circle(img, (center_x - 30, center_y - 20), 45, (40, 60, 80), -1)
    cv2.circle(img, (center_x + 30, center_y + 25), 50, (60, 80, 100), -1)
    cv2.circle(img, (center_x - 10, center_y + 40), 40, (30, 50, 70), -1)
    
    # Blur to make it look realistic
    img = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Add noise / texture
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add text overlay
    cv2.putText(img, "Dermoscope Live Preview (MOCK MODE)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
    
    # Check if LED Ring Light is "ON"
    global led_enabled
    if led_enabled:
        # Illuminate center
        overlay = img.copy()
        cv2.circle(overlay, (320, 240), 180, (255, 255, 255), -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        cv2.putText(img, "LED RING LIGHT ACTIVE", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
                    
    return img

def get_real_sample_image():
    """Load a real image from HAM10000 directory if available, for realistic mock."""
    try:
        from src.config import DATA_DIR
        img_dirs = [
            os.path.join(DATA_DIR, "HAM10000_images_part_1"),
            os.path.join(DATA_DIR, "HAM10000_images_part_2")
        ]
        for d in img_dirs:
            if os.path.exists(d):
                files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg")]
                if files:
                    import random
                    path = random.choice(files)
                    return Image.open(path)
    except Exception:
        pass
    return None

# Video streaming generator
def gen_frames():
    t = 0
    while True:
        t += 0.1
        global camera_mode
        if camera_mode == "mock":
            frame = generate_mock_image(t)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
        else:
            cam = get_camera()
            if cam is None or not cam.isOpened():
                camera_mode = "mock"
                continue
            ret, frame = cam.read()
            if not ret:
                print("[Camera] Error reading frame.")
                continue
                
            # If LED enabled, we can add a visual indicator text or let the hardware handle it
            if led_enabled:
                cv2.putText(frame, "LED LIGHT ON", (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)  # ~25 FPS

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle_led', methods=['POST'])
def toggle_led():
    global led_enabled
    data = request.json or {}
    state = data.get("state", not led_enabled)
    
    led_enabled = state
    try:
        val = GPIO.HIGH if led_enabled else GPIO.LOW
        GPIO.output(led_pin, val)
        status = "ON" if led_enabled else "OFF"
        print(f"[GPIO] LED Ring Light turned {status} on pin {led_pin}")
    except Exception as e:
        print(f"[GPIO] Error toggling LED pin: {e}")
        
    return jsonify({"success": True, "led_enabled": led_enabled})

@app.route('/api/test_led', methods=['POST'])
def test_led():
    # Flashes the LED twice
    def flash():
        try:
            print("[GPIO] Flashing LED for test...")
            for _ in range(2):
                GPIO.output(led_pin, GPIO.HIGH)
                time.sleep(0.15)
                GPIO.output(led_pin, GPIO.LOW)
                time.sleep(0.15)
        except Exception as e:
            print(f"[GPIO] Test flash error: {e}")
            
    threading.Thread(target=flash).start()
    return jsonify({"success": True})

@app.route('/api/settings', methods=['POST', 'GET'])
def settings():
    global camera_mode, camera_index, led_pin
    if request.method == 'POST':
        data = request.json or {}
        new_mode = data.get("camera_mode", camera_mode)
        new_index = int(data.get("camera_index", camera_index))
        new_pin = int(data.get("led_pin", led_pin))
        
        # Apply camera change if changed
        if new_mode != camera_mode or new_index != camera_index:
            release_camera()
            camera_mode = new_mode
            camera_index = new_index
            get_camera()  # try initializing
            
        # Apply LED pin change if changed
        if new_pin != led_pin:
            try:
                GPIO.cleanup()
            except Exception:
                pass
            led_pin = new_pin
            init_gpio()
            
        return jsonify({
            "success": True, 
            "camera_mode": camera_mode, 
            "camera_index": camera_index, 
            "led_pin": led_pin
        })
    else:
        return jsonify({
            "camera_mode": camera_mode,
            "camera_index": camera_index,
            "led_pin": led_pin,
            "device": DEVICE.type
        })

@app.route('/api/system_stats')
def system_stats():
    return jsonify(get_system_metrics())

@app.route('/api/capture', methods=['POST'])
def capture_and_classify():
    global led_enabled, camera_mode
    
    # 1. LED Flash Sequence
    try:
        # Turn LED ON
        GPIO.output(led_pin, GPIO.HIGH)
        led_enabled = True
        time.sleep(0.2)  # Wait for illumination to settle
    except Exception as e:
        print(f"[GPIO] Capture LED ON error: {e}")

    # 2. Capture Frame
    pil_image = None
    try:
        if camera_mode == "mock":
            # Attempt to get a real sample image from HAM10000, fallback to synthetic
            pil_image = get_real_sample_image()
            if pil_image is None:
                # Generate synthetic frame and convert to PIL
                cv_img = generate_mock_image()
                cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_img_rgb)
        else:
            cam = get_camera()
            if cam is not None and cam.isOpened():
                # Flush buffer
                for _ in range(5):
                    cam.read()
                ret, frame = cam.read()
                if ret:
                    cv_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(cv_img_rgb)
                    
            if pil_image is None:
                # If frame capture failed, load fallback synthetic
                cv_img = generate_mock_image()
                cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_img_rgb)
    except Exception as e:
        print(f"[Capture] Capture frame error: {e}")
        # Final fallback
        pil_image = Image.new("RGB", (CFG["image_size"], CFG["image_size"]), (128, 128, 128))
    
    # 3. Turn LED OFF
    try:
        GPIO.output(led_pin, GPIO.LOW)
        led_enabled = False
    except Exception as e:
        print(f"[GPIO] Capture LED OFF error: {e}")

    # 4. Process and Run Inference
    try:
        # Save captured original image
        captured_path = os.path.join(OUTPUT_DIR, "captured.png")
        pil_image.save(captured_path)
        
        # Run classification
        classifier = get_classifier()
        pred_res = classifier.predict(pil_image)
        
        # Run Grad-CAM
        gradcam_engine = GradCAM(classifier.model)
        _, heatmap_img, overlay_img = gradcam_engine.generate(
            pil_image, pred_res["class_idx"], classifier.device
        )
        
        # Save visualizations
        heatmap_path = os.path.join(OUTPUT_DIR, "heatmap.png")
        overlay_path = os.path.join(OUTPUT_DIR, "overlay.png")
        heatmap_img.save(heatmap_path)
        overlay_img.save(overlay_path)
        
        # Append image URIs
        pred_res["captured_url"] = "/static/outputs/captured.png?t=" + str(time.time())
        pred_res["heatmap_url"] = "/static/outputs/heatmap.png?t=" + str(time.time())
        pred_res["overlay_url"] = "/static/outputs/overlay.png?t=" + str(time.time())
        pred_res["success"] = True
        
        return jsonify(pred_res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

# Federated Learning Integration Functions
def fl_thread_worker(client_id):
    global fl_active, fl_logs
    fl_logs.append(f"[{time.strftime('%H:%M:%S')}] >>> INITIALIZING FL CLIENT {client_id} ...\n")
    
    # We will trigger the fl.client_app or fl.client (depending on what's available)
    # Check what exists. We have fl/client_app.py!
    cmd = [sys.executable, "fl/client_app.py", str(client_id)]
    
    # We must run it with proper environment
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        global fl_process
        fl_process = process
        
        # Read stdout line by line
        for line in iter(process.stdout.readline, ""):
            fl_logs.append(line)
            # Limit logs size in memory
            if len(fl_logs) > 1000:
                fl_logs.pop(0)
                
        process.wait()
        fl_logs.append(f"\n[{time.strftime('%H:%M:%S')}] >>> FL CLIENT PROCESS EXITED WITH CODE: {process.returncode}\n")
        
        # If client successfully completed and saved weights, reload classifier model
        if process.returncode == 0:
            fl_logs.append(f"[{time.strftime('%H:%M:%S')}] >>> PARTICIPATION COMPLETE. Reloading global weights...\n")
            global current_classifier
            current_classifier = None # trigger reload
            
    except Exception as e:
        fl_logs.append(f"\n[{time.strftime('%H:%M:%S')}] ERROR EXECUTING FL CLIENT: {str(e)}\n")
    finally:
        fl_active = False

@app.route('/api/fl/start', methods=['POST'])
def start_fl_client():
    global fl_active, fl_logs
    if fl_active:
        return jsonify({"success": False, "message": "Federated Learning client already running."})
        
    data = request.json or {}
    client_id = data.get("client_id", 0)
    
    fl_logs = []
    fl_active = True
    
    # Launch FL client in separate background thread
    threading.Thread(target=fl_thread_worker, args=(client_id,)).start()
    
    return jsonify({"success": True, "message": "Federated Learning client started."})

@app.route('/api/fl/status')
def fl_status():
    global fl_active, fl_logs
    return jsonify({
        "active": fl_active,
        "logs": "".join(fl_logs)
    })

@app.route('/api/fl/stop', methods=['POST'])
def stop_fl_client():
    global fl_process, fl_active
    if fl_process and fl_active:
        try:
            fl_process.terminate()
            fl_active = False
            return jsonify({"success": True, "message": "Client terminated."})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    return jsonify({"success": False, "message": "No active client running."})

# Static serve helper for outputs directory
@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    # Initial setup
    get_camera()
    print("---------------------------------------------")
    print("Skin Cancer Detection Web GUI Server running.")
    print("Access locally: http://localhost:5000")
    print("---------------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
