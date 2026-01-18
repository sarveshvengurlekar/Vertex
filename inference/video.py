import os
import uuid
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Load model once
model = YOLO("model/20_best.pt")  # change path if needed

def run_video_detection(file_storage, upload_dir, processed_dir, conf=0.25):
    """
    Process video with YOLO detection and return both original and processed filenames.
    
    Args:
        file_storage: Flask FileStorage object containing uploaded video
        upload_dir: Directory to save original uploaded video
        processed_dir: Directory to save processed video
        conf: Confidence threshold for YOLO detection
    
    Returns:
        tuple: (original_filename, processed_filename, error_message, status_code)
    """
    import os, cv2, uuid
    from werkzeug.utils import secure_filename

    cap = None
    out = None

    if not file_storage or file_storage.filename == "":
        return None, None, "No file selected", 400

    if not file_storage.filename.lower().endswith(".mp4"):
        return None, None, "Only .mp4 files allowed", 400

    unique_id = uuid.uuid4().hex[:10]
    original_name = secure_filename(file_storage.filename)

    # Use "original_" prefix instead of "input_" to match app.py naming
    original_filename = f"original_{unique_id}_{original_name}"
    processed_filename = f"processed_{unique_id}.mp4"

    original_path = os.path.join(upload_dir, original_filename)
    processed_path = os.path.join(processed_dir, processed_filename)

    try:
        # Save the original uploaded file
        file_storage.save(original_path)

        # Open video for processing
        cap = cv2.VideoCapture(original_path)
        if not cap.isOpened():
            return None, None, "Cannot open video", 400

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer for processed output
        # Use avc1 (H.264) codec for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(processed_path, fourcc, fps, (w, h))

        if not out.isOpened():
            return None, None, "VideoWriter failed to open (codec issue)", 500

        # Process each frame with YOLO
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=conf, verbose=True)
            annotated = results[0].plot()
            out.write(annotated)

        # Return both filenames for serving to frontend
        return original_filename, processed_filename, None, 200

    except Exception as e:
        # Cleanup on error
        for path in [original_path, processed_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        return None, None, str(e), 500

    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
