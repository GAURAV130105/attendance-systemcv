"""
Configuration for the Attendance Management System.
Uses YOLOv8-face detector + LBPH face recognizer (no dlib/C++ compilation needed).
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory paths
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ENCODINGS_DIR   = os.path.join(BASE_DIR, "encodings")
ATTENDANCE_DIR  = os.path.join(BASE_DIR, "attendance_records")
MODELS_DIR      = os.path.join(BASE_DIR, "models")

# Ensure directories exist
for d in [KNOWN_FACES_DIR, ENCODINGS_DIR, ATTENDANCE_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── YOLOv8-face detection settings ─────────────────────────────
YOLO_CONFIDENCE_THRESHOLD = 0.50   # Minimum detection confidence
YOLO_FACE_MODEL  = os.path.join(MODELS_DIR, "yolov8n-face.pt")
YOLO_IMG_SIZE    = 320             # Inference resolution — smaller = faster on CPU
YOLO_HALF        = False           # Half-precision (GPU only — keep False for CPU)

# YOLO model download URL
YOLO_MODEL_URL = (
    "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)

# ── Face recognition settings (LBPH) ───────────────────────────
# Raised from 75 → 90 so recognition still works under different
# lighting conditions (e.g. morning enrolled, evening attending).
# If you get too many false matches, lower back towards 75.
LBPH_THRESHOLD        = 90   # Lower = stricter (50-100 range for ≤20 students)
NUM_ENROLLMENT_PHOTOS = 5    # Raw captures per student
# Each raw capture is augmented to N training samples automatically
# Raised from 12 → 20 to include gamma/contrast variants that improve
# recognition across different lighting hours (morning vs. evening).
AUGMENT_SAMPLES_PER_PHOTO = 20   # flip + brightness + gamma + contrast variants

# ── Recognition voting (smooths per-frame jitter) ───────────────
# A name is only "confirmed" once it wins VOTE_THRESHOLD fraction
# of the last VOTE_FRAMES worker frames for that face position.
# VOTE_FRAMES=3 → green box appears faster (locks in within ~0.1 s).
# VOTE_THRESHOLD=0.55 → tolerates 1 bad frame in 3 without dropping lock.
VOTE_FRAMES    = 3    # Rolling window size (was 5)
VOTE_THRESHOLD = 0.55 # 55 % of frames must agree (was 0.60)

# ── Camera settings ────────────────────────────────────────────
CAMERA_INDEX = 0   # Default webcam

# ── File paths ─────────────────────────────────────────────────
LBPH_MODEL_FILE  = os.path.join(ENCODINGS_DIR, "lbph_model.yml")
LABEL_MAP_FILE   = os.path.join(ENCODINGS_DIR, "label_map.pkl")
# Single master attendance workbook — all sessions in one file
ATTENDANCE_FILE  = os.path.join(ATTENDANCE_DIR, "attendance_records.xlsx")

# ── Performance / MediaPipe throttle ───────────────────────────
# Run the heavy MediaPipe face-mesh overlay only every Nth worker frame.
# Setting to 1 draws on every frame (slow); 10 gives a smooth ~3 fps mesh update.
FACE_MESH_EVERY_N_FRAMES = 10

# ── GUI settings ───────────────────────────────────────────────
WINDOW_TITLE  = "Smart Attendance System"
WINDOW_WIDTH  = 1200
WINDOW_HEIGHT = 750

SHOW_FPS_COUNTER = True   # Overlay live FPS on camera feed

# ── Colors (BGR for OpenCV, hex for tkinter) ───────────────────
COLOR_RECOGNIZED_BGR = (0, 200, 0)
COLOR_UNKNOWN_BGR    = (0, 0, 220)
COLOR_RECOGNIZED_HEX = "#00C853"
COLOR_UNKNOWN_HEX    = "#FF1744"
BG_COLOR      = "#1a1a2e"
FG_COLOR      = "#e0e0e0"
ACCENT_COLOR  = "#0f3460"
BUTTON_COLOR  = "#16213e"
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#FF6D00"
