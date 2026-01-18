from flask import Flask, render_template, send_from_directory, request, Response, jsonify, abort
import os
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import io
import uuid
from werkzeug.utils import secure_filename

# Import detection modules
from inference.video import run_video_detection
from inference.image import run_image_detection, image_validator
from inference.webcam import webcam_manager

app = Flask(__name__,
            static_folder='inference/static',
            template_folder='inference/templates')

# Helper for absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure upload folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, "inference", "uploads")
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, "processed")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1500 * 1024 * 1024  # 1.5 GB for videos

# Load YOLO model once at startup
model = YOLO("model/20_best.pt")

# ========== Helper Functions ==========

def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

# ========== Main Routes ==========

@app.route('/')
@app.route('/home')
def home():
    """Main entry point - serves the main index.html page"""
    return render_template('index.html')

# ========== Template Fragment Routes ==========
# These routes serve the HTML partials that Vue.js loads dynamically

@app.route('/templates/<path:filename>')
def serve_template_fragments(filename):
    """Serve HTML partials from inference/templates/ folder"""
    return send_from_directory('inference/templates', filename)

@app.route('/inference/<path:filename>')
def serve_inference_pages(filename):
    """Serve inference-related HTML fragments"""
    # Check if it's in the inference folder or templates subfolder
    inference_path = os.path.join('inference', filename)
    if os.path.exists(inference_path):
        return send_from_directory('inference', filename)
    else:
        # Try templates subfolder
        return send_from_directory('inference/templates', filename)

# ========== Image Inference Routes ==========

@app.route('/image-inference', methods=['GET'])
def image_inference():
    """Image Inference page"""
    return render_template('image_inference.html')

@app.route('/image-upload', methods=['POST'])
def image_upload():
    """Process image upload and return detection results as JSON"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    # Use the image detection module
    original, result, detections_list, error, status_code = run_image_detection(file)
    
    if error:
        return jsonify({"success": False, "error": error}), status_code
        
    original_b64 = to_base64(original) if original is not None else None
    result_b64 = to_base64(result) if result is not None else None
    
    # Format detections for display
    detections = [f"{d['label']:12} : {d['confidence']:.3f}" for d in detections_list]
    
    return jsonify({
        "success": True,
        "original": original_b64,
        "result": result_b64,
        "detections": detections
    }), 200

# ========== Video Inference Routes ==========

@app.route('/video-inference')
def video_inference():
    """Video Inference page"""
    return render_template('video_inference.html')

@app.route('/upload', methods=['POST'])
def upload_and_process():
    """Upload and process video with YOLO using the video detection module"""
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Use the video detection module
    original_filename, processed_filename, error, status_code = run_video_detection(
        file_storage=file,
        upload_dir=app.config['UPLOAD_FOLDER'],
        processed_dir=app.config['PROCESSED_FOLDER'],
        conf=0.25
    )
    
    # Handle errors
    if error:
        return jsonify({"error": error}), status_code
    
    # Return links to both original and processed videos
    return jsonify({
        "success": True,
        "message": "Processing finished",
        "original_video_url": f"/uploads/{original_filename}",
        "processed_video_url": f"/processed/{processed_filename}"
    }), status_code

@app.route("/processed/<filename>")
def serve_processed(filename):
    """Serve processed video files"""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# ========== Webcam Inference Routes ==========

@app.route('/webcam-inference')
def webcam_inference():
    """Webcam Inference page"""
    return render_template('webcam_inference.html')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start the webcam stream"""
    success, message = webcam_manager.start()
    return jsonify({"success": success, "message": message})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop the webcam stream"""
    success, message = webcam_manager.stop()
    return jsonify({"success": success, "message": message})

@app.route('/video_feed')
def video_feed():
    """Video streaming route for webcam"""
    return Response(
        webcam_manager.get_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ========== Other Page Routes (for direct access) ==========

@app.route('/model-inference')
def model_inference():
    """Model Inference page"""
    return render_template('inference.html')

@app.route('/dataset')
def dataset():
    """Dataset Overview page"""
    return render_template('dataset.html')

@app.route('/evaluation')
def evaluation():
    """Model Evaluation page"""
    return render_template('evaluation.html')

@app.route('/docs')
def docs():
    """Documentation page"""
    return render_template('docs.html')

@app.route('/dev')
def dev():
    """Developer info page"""
    return render_template('dev.html')

@app.route('/test')
def test():
    """Test page"""
    return render_template('test.html')

# ========== Static File Routes ==========

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, images)"""
    return send_from_directory('inference/static', filename)

# Optional: Serve uploads and other media
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """Serve uploaded original video files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/inference/uploads/<path:filename>')
def serve_inference_uploads(filename):
    """Serve inference uploaded files"""
    return send_from_directory('inference/uploads', filename)


