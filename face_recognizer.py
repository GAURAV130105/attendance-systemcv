"""
Face Recognizer Module (OpenCV LBPH version)
Real-time face recognition using OpenCV's LBPH Face Recognizer.
No dlib or C++ compilation required.

Key fixes vs previous version:
  - MediaPipe FaceLandmarker is lazily initialised (no blocking on startup)
  - Face mesh overlay is throttled to every FACE_MESH_EVERY_N_FRAMES frames
  - FPS is measured per-frame and drawn on the video feed when SHOW_FPS_COUNTER=True
"""
import os
import time
import threading
import urllib.request
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import face_landmarker as fl

from face_encoder import load_all_encodings
from face_detector import FaceDetector
from config import (
    LBPH_MODEL_FILE, LBPH_THRESHOLD, MODELS_DIR,
    FACE_MESH_EVERY_N_FRAMES, SHOW_FPS_COUNTER,
    VOTE_FRAMES, VOTE_THRESHOLD
)

_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_LANDMARKER_PATH = None   # set in __init__


class FaceRecognizer:
    """Handles real-time face recognition from camera frames."""

    def __init__(self):
        """Load the LBPH model and label map. MediaPipe is loaded lazily."""
        global _LANDMARKER_PATH
        _LANDMARKER_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

        self.detector = FaceDetector()
        self.recognizer = None
        self.label_map  = {}
        self.reverse_label_map = {}  # label_id → {name, roll_no}

        # ── CLAHE — same params as face_encoder training pipeline ────────────
        # Normalises each live face ROI so brightness/contrast differences
        # between enrolment session and live recognition are eliminated.
        # Must match the clipLimit/tileGridSize used in _train_lbph_model.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ── Vote buffer: tracks last N raw predictions per face slot ────
        # One deque per simultaneously visible face; reset when face count changes.
        # A name wins once it holds >= VOTE_THRESHOLD of the rolling window.
        self._vote_buffers: list[deque] = []

        # ── MediaPipe lazy-init state ───────────────────────────────────
        self._face_landmarker = None          # created on first mesh draw
        self._landmarker_lock = threading.Lock()
        self._landmarker_ready = False
        # Start background download/init immediately so it's ready as soon
        # as the first mesh draw is needed — but never blocks the GUI.
        threading.Thread(target=self._init_landmarker, daemon=True).start()

        # ── Throttle: draw mesh every Nth worker frame ──────────────────
        self._frame_counter = 0
        self._mesh_every_n  = FACE_MESH_EVERY_N_FRAMES

        # ── FPS measurement ─────────────────────────────────────────────
        self._fps_t0   = time.perf_counter()
        self._fps_count = 0
        self._fps_last  = 0.0   # Most recently computed FPS

        self.reload_model()

    # ──────────────────────────────────────────────────────────────────────
    # Model management
    # ──────────────────────────────────────────────────────────────────────

    def reload_model(self):
        """Reload the LBPH model and label map from disk."""
        self.label_map = load_all_encodings()

        self.reverse_label_map = {}
        for name, data in self.label_map.items():
            self.reverse_label_map[data["label_id"]] = {
                "name": name,
                "roll_no": data["roll_no"]
            }

        if os.path.exists(LBPH_MODEL_FILE):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(LBPH_MODEL_FILE)
            self.recognizer.setThreshold(LBPH_THRESHOLD)
            print(
                f"[FaceRecognizer] LBPH model loaded. "
                f"{len(self.label_map)} student(s) enrolled."
            )
        else:
            self.recognizer = None
            print("[FaceRecognizer] No LBPH model found. Enroll students first.")

    # ──────────────────────────────────────────────────────────────────────
    # MediaPipe — lazy background initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _init_landmarker(self):
        """Download (if needed) and init the FaceLandmarker — runs in BG thread."""
        try:
            if not os.path.exists(_LANDMARKER_PATH):
                print("[FaceRecognizer] Downloading face landmarker model (~10 MB)…")
                urllib.request.urlretrieve(_LANDMARKER_URL, _LANDMARKER_PATH)
                print("[FaceRecognizer] Face landmarker downloaded.")

            base_options = mp.tasks.BaseOptions(model_asset_path=_LANDMARKER_PATH)
            options = fl.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=5,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
            landmarker = fl.FaceLandmarker.create_from_options(options)
            with self._landmarker_lock:
                self._face_landmarker = landmarker
                self._landmarker_ready = True
            print("[FaceRecognizer] FaceLandmarker ready.")
        except Exception as e:
            print(f"[FaceRecognizer] FaceLandmarker init failed (mesh will be skipped): {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Core recognition
    # ──────────────────────────────────────────────────────────────────────

    def recognize_frame(self, frame):
        """
        Detect and recognise faces in a single frame.

        Recognition is **vote-smoothed**: raw LBPH predictions are buffered over
        VOTE_FRAMES consecutive frames per face slot.  A name is only reported
        as confirmed once it wins ≥ VOTE_THRESHOLD fraction of the window.
        This eliminates the flickering Unknown → Name → Unknown cycle that made
        attendance marking feel unreliable.

        Args:
            frame: BGR OpenCV frame

        Returns:
            list of dicts:
                name        — student name or 'Unknown'
                roll_no     — roll number or ''
                location    — (x, y, w, h) bounding box
                confidence  — float 0-1 (1 = perfect match)
                confirmed   — bool: True when vote threshold is met
        """
        faces = self.detector.detect(frame)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Grow / shrink vote buffer slots to match detected face count
        while len(self._vote_buffers) < len(faces):
            self._vote_buffers.append(deque(maxlen=VOTE_FRAMES))
        # Trim excess slots (faces left the frame)
        self._vote_buffers = self._vote_buffers[:len(faces)]

        results = []
        for slot_idx, (x, y, w, h) in enumerate(faces):
            raw_name   = "Unknown"
            raw_roll   = ""
            raw_conf   = 0.0

            if self.recognizer is not None:
                face_roi     = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))
                # Apply CLAHE — must match the normalisation applied during
                # training in face_encoder._train_lbph_model so that distances
                # are meaningful across different lighting sessions.
                face_resized = self._clahe.apply(face_resized)
                try:
                    label_id, dist = self.recognizer.predict(face_resized)
                    if dist < LBPH_THRESHOLD and label_id in self.reverse_label_map:
                        student  = self.reverse_label_map[label_id]
                        raw_name = student["name"]
                        raw_roll = student["roll_no"]
                        raw_conf = max(0.0, 1.0 - (dist / LBPH_THRESHOLD))
                except cv2.error:
                    pass

            # Push raw prediction into the vote buffer for this slot
            buf = self._vote_buffers[slot_idx]
            buf.append(raw_name)

            # Tally votes
            counts: dict[str, int] = {}
            for n in buf:
                counts[n] = counts.get(n, 0) + 1
            best_name  = max(counts, key=counts.__getitem__)
            best_votes = counts[best_name]
            confirmed  = (
                best_name != "Unknown"
                and best_votes / len(buf) >= VOTE_THRESHOLD
            )

            # Use raw values for conf/roll when confirmed, else Unknown
            name       = raw_name if confirmed else ("Unknown" if best_name == "Unknown" else "SCANNING...")
            roll_no    = raw_roll if confirmed else ""
            confidence = raw_conf if confirmed else 0.0

            results.append({
                "name":       name,
                "roll_no":    roll_no,
                "location":   (x, y, w, h),
                "confidence": round(confidence, 2),
                "confirmed":  confirmed,
            })

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ──────────────────────────────────────────────────────────────────────

    def _draw_corner_brackets(self, frame, x, y, w, h, color,
                               thickness=2, length_ratio=0.25):
        """Futuristic corner-bracket bounding box."""
        corner_len = int(min(w, h) * length_ratio)
        x2, y2 = x + w, y + h

        for (cx, cy, sx, sy) in [
            (x,  y,   1,  1),
            (x2, y,  -1,  1),
            (x,  y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + sx * corner_len, cy), color, thickness)
            cv2.line(frame, (cx, cy), (cx, cy + sy * corner_len), color, thickness)

    def _draw_scan_line(self, frame, x, y, w, h, color):
        """Animated horizontal scan line sweeping through the face bbox."""
        elapsed  = time.time() - self._fps_t0
        cycle    = elapsed % 2.0
        progress = cycle / 2.0 if cycle < 1.0 else (2.0 - cycle) / 1.0
        scan_y   = int(y + h * progress)

        overlay = frame.copy()
        cv2.line(overlay, (x, scan_y), (x + w, scan_y), color, 1)
        glow_h = 8
        cv2.rectangle(overlay,
                      (x, max(y, scan_y - glow_h)),
                      (x + w, min(y + h, scan_y + glow_h)),
                      color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.line(frame, (x + 2, scan_y), (x + w - 2, scan_y), color, 1)

    def _draw_confidence_bar(self, frame, x, y, w, confidence, color):
        """Thin confidence bar above the face box."""
        bar_y    = y - 12
        bar_h    = 4
        filled_w = int(w * confidence)

        cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + bar_h),
                      (40, 40, 40), -1)
        if filled_w > 0:
            cv2.rectangle(frame, (x, bar_y), (x + filled_w, bar_y + bar_h),
                          color, -1)
        cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + bar_h), color, 1)

    def _draw_hud_label(self, frame, x, y, w, h, name, confidence, color):
        """Semi-transparent HUD label below the face box."""
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1

        if name in ("Unknown", "SCANNING..."):
            label     = "SCANNING..."
            sub_label = "IDENTIFYING"
        else:
            label     = name.upper()
            sub_label = f"MATCH: {confidence:.0%}  ✓ PRESENT"

        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        sub_size   = cv2.getTextSize(sub_label, font, font_scale * 0.8, thickness)[0]

        label_y = y + h + 20
        pad     = 8
        bg_w    = max(label_size[0], sub_size[0]) + pad * 2
        bg_h    = label_size[1] + sub_size[1] + pad * 3

        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x, y + h + 4),
                      (x + bg_w, y + h + 4 + bg_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.line(frame, (x, y + h + 4), (x + bg_w, y + h + 4), color, 2)
        cv2.putText(frame, label,
                    (x + pad, label_y + label_size[1]),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, sub_label,
                    (x + pad, label_y + label_size[1] + sub_size[1] + pad),
                    font, font_scale * 0.8, (180, 180, 180), thickness, cv2.LINE_AA)

    def _draw_face_mesh(self, frame, recognized=False):
        """
        Draw face contour wireframe (called at most every FACE_MESH_EVERY_N_FRAMES
        frames).  No-ops silently if the landmarker is not yet ready.
        """
        with self._landmarker_lock:
            if not self._landmarker_ready or self._face_landmarker is None:
                return
            landmarker = self._face_landmarker

        h, w = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            mp_results = landmarker.detect(mp_image)
        except Exception as e:
            print(f"[FaceRecognizer] MediaPipe face mesh error: {e}")
            return

        if not mp_results.face_landmarks:
            return

        mesh_color = (0, 200, 80) if recognized else (60, 60, 180)
        overlay    = frame.copy()

        contour_sets = [
            fl.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
            fl.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
            fl.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
            fl.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
            fl.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
            fl.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
            fl.FaceLandmarksConnections.FACE_LANDMARKS_NOSE,
        ]

        for face_landmarks in mp_results.face_landmarks:
            for contour in contour_sets:
                for connection in contour:
                    s = face_landmarks[connection.start]
                    e = face_landmarks[connection.end]
                    cv2.line(overlay,
                             (int(s.x * w), int(s.y * h)),
                             (int(e.x * w), int(e.y * h)),
                             mesh_color, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    def _draw_fps(self, frame):
        """Overlay the live FPS reading in the top-right corner."""
        # Update FPS measurement
        self._fps_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self._fps_last  = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_t0    = now

        fps_text = f"FPS: {self._fps_last:.1f}"
        font     = cv2.FONT_HERSHEY_SIMPLEX
        scale    = 0.6
        thick    = 1
        (tw, th), _ = cv2.getTextSize(fps_text, font, scale, thick)

        h_frame, w_frame = frame.shape[:2]
        margin = 10
        x_pos  = w_frame - tw - margin - 4
        y_pos  = margin + th + 4

        # Dark pill background
        cv2.rectangle(frame,
                      (x_pos - 4, margin),
                      (x_pos + tw + 4, margin + th + 8),
                      (20, 20, 20), -1)
        cv2.rectangle(frame,
                      (x_pos - 4, margin),
                      (x_pos + tw + 4, margin + th + 8),
                      (0, 180, 100), 1)

        # Choose colour: green ≥25 fps, amber 15-24, red <15
        if self._fps_last >= 25:
            fps_color = (0, 220, 80)
        elif self._fps_last >= 15:
            fps_color = (0, 165, 220)
        else:
            fps_color = (0, 60, 220)

        cv2.putText(frame, fps_text,
                    (x_pos, y_pos + 2),
                    font, scale, fps_color, thick, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────────
    # Main draw entry point (called by worker thread)
    # ──────────────────────────────────────────────────────────────────────

    def draw_results(self, frame, results):
        """
        Draw futuristic HUD-style bounding boxes + labels on the frame.

        Args:
            frame:   BGR OpenCV frame (modified in-place)
            results: Output from recognize_frame()

        Returns:
            Annotated frame
        """
        self._frame_counter += 1

        for result in results:
            x, y, w, h = result["location"]
            name        = result["name"]
            confidence  = result["confidence"]
            confirmed   = result.get("confirmed", False)

            if not confirmed:
                color      = (0, 0, 200)
                glow_color = (0, 0, 120)
            else:
                color      = (0, 230, 0)
                glow_color = (0, 150, 0)

            # Outer glow brackets
            self._draw_corner_brackets(frame, x - 3, y - 3, w + 6, h + 6,
                                       glow_color, thickness=1, length_ratio=0.2)
            # Main corner brackets
            self._draw_corner_brackets(frame, x, y, w, h,
                                       color, thickness=2, length_ratio=0.25)
            # Animated scan line
            self._draw_scan_line(frame, x, y, w, h, color)
            # Confidence bar (confirmed faces only)
            if confirmed:
                self._draw_confidence_bar(frame, x, y, w, confidence, color)
            # HUD label
            self._draw_hud_label(frame, x, y, w, h, name, confidence, color)

        # Face mesh — throttled AND only for frames with at least one confirmed face.
        # Skipping mesh for Unknown saves ~30 ms per call on CPU.
        any_confirmed = any(r.get("confirmed", False) for r in results)
        if any_confirmed and (self._frame_counter % self._mesh_every_n == 0):
            self._draw_face_mesh(frame, recognized=True)

        # Live FPS counter
        if SHOW_FPS_COUNTER:
            self._draw_fps(frame)

        return frame

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def num_enrolled(self):
        """Number of enrolled students."""
        return len(self.label_map)
