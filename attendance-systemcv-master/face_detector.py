"""
Face Detector Module
Uses YOLOv8-face for accurate, real-time face detection.
Downloads model automatically if not present.

CPU-tuned: imgsz=320, half=False for maximum compatibility.
"""
import os
import urllib.request
from ultralytics import YOLO
from config import (
    YOLO_FACE_MODEL, YOLO_MODEL_URL, MODELS_DIR,
    YOLO_CONFIDENCE_THRESHOLD, YOLO_IMG_SIZE, YOLO_HALF
)


class FaceDetector:
    """YOLOv8-face based face detector — CPU-optimised."""

    _model = None  # Class-level shared model instance

    def __init__(self):
        """Initialize the face detector, downloading model if needed."""
        if FaceDetector._model is None:
            self._ensure_model()
            FaceDetector._model = YOLO(YOLO_FACE_MODEL)
            print("[FaceDetector] YOLOv8-face model loaded successfully.")

    def _ensure_model(self):
        """Download model file if it doesn't exist."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        if not os.path.exists(YOLO_FACE_MODEL):
            print("[FaceDetector] Downloading YOLOv8-face model (~6 MB)...")
            urllib.request.urlretrieve(YOLO_MODEL_URL, YOLO_FACE_MODEL)
            print("[FaceDetector] Model downloaded.")

    def detect(self, frame):
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR OpenCV frame

        Returns:
            list of (x, y, w, h) tuples for each detected face
        """
        results = FaceDetector._model.predict(
            frame,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            imgsz=YOLO_IMG_SIZE,   # Smaller input → faster CPU inference
            half=YOLO_HALF,        # False on CPU
            verbose=False,
        )

        faces = []
        h_frame, w_frame = frame.shape[:2]
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Clamp to frame boundaries
                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(w_frame, x2);  y2 = min(h_frame, y2)

                face_w = x2 - x1
                face_h = y2 - y1

                if face_w > 30 and face_h > 30:   # Minimum useful face size
                    faces.append((x1, y1, face_w, face_h))

        return faces

    def detect_with_confidence(self, frame):
        """
        Detect faces with confidence scores.

        Args:
            frame: BGR OpenCV frame

        Returns:
            list of (x, y, w, h, confidence) tuples
        """
        results = FaceDetector._model.predict(
            frame,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            imgsz=YOLO_IMG_SIZE,
            half=YOLO_HALF,
            verbose=False,
        )

        faces = []
        h_frame, w_frame = frame.shape[:2]
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])

                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(w_frame, x2);  y2 = min(h_frame, y2)

                face_w = x2 - x1
                face_h = y2 - y1

                if face_w > 30 and face_h > 30:
                    faces.append((x1, y1, face_w, face_h, confidence))

        return faces
