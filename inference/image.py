import cv2
import numpy as np
import base64
import io
from PIL import Image
from ultralytics import YOLO

# Load model once (module-level, efficient)
model = YOLO("model/20_best.pt")


def image_validator(file_stream):
    """
    Validate JPEG/JPG/PNG image under 15MB (in-memory)
    """
    if not file_stream:
        return False, "No file provided"

    try:
        file_bytes = file_stream.read()
    except Exception as e:
        return False, f"Cannot read file: {str(e)}"

    if len(file_bytes) == 0:
        return False, "Empty file"

    if len(file_bytes) > 15_000_000:
        return False, "File size exceeds 15MB"

    try:
        img = Image.open(io.BytesIO(file_bytes))
        if img.format not in ("JPEG", "JPG", "PNG"):
            return False, f"This is a {img.format} image, not JPEG/JPG/PNG"
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def run_image_detection(file_storage):
    """
    Core image detection function

    Args:
        file_storage: Flask FileStorage object

    Returns:
        original_img (np.ndarray | None)
        result_img   (np.ndarray | None)
        detections   (list)
        error        (str | None)
        status_code  (int)
    """

    original_img = None
    result_img = None
    detections = []
    error = None
    status_code = 200

    if file_storage.filename == "":
        return None, None, [], "No file selected", 400

    # ---- Validate ----
    is_valid, msg = image_validator(file_storage)
    if not is_valid:
        return None, None, [], msg, 400

    # Reset pointer after validation
    file_storage.seek(0)

    try:
        # ---- Decode Image ----
        file_bytes = np.frombuffer(file_storage.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return None, None, [], "Failed to decode image", 400

        original_img = img.copy()

        # ---- YOLO Inference ----
        results = model.predict(
            source=img,
            conf=0.25,
            iou=0.45,
            verbose=False
        )

        result = results[0]

        # ---- Detections ----
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3)
                })
        else:
            detections.append({"label": "No objects detected", "confidence": 0.0})

        # ---- Annotated Image ----
        result_img = result.plot()

        return original_img, result_img, detections, None, 200

    except Exception as e:
        return None, None, [], f"Processing error: {str(e)}", 500
