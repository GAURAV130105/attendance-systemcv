"""
Face Encoder Module (YOLOv8-face + LBPH version)
Handles student enrollment: capturing face photos and training the LBPH recognizer.
No dlib or C++ compilation required.

Augmentation: each captured photo is flipped + brightness-varied to produce
AUGMENT_SAMPLES_PER_PHOTO training samples, dramatically improving LBPH accuracy
for fewer-than-20-student use cases.
"""
import os
import pickle
import re
import shutil
import cv2
import numpy as np
from config import (
    KNOWN_FACES_DIR, ENCODINGS_DIR, LBPH_MODEL_FILE,
    LABEL_MAP_FILE, NUM_ENROLLMENT_PHOTOS, AUGMENT_SAMPLES_PER_PHOTO
)
from face_detector import FaceDetector

# Module-level FaceDetector singleton — avoids re-constructing on every capture
_detector = None


def _sanitize_path_component(value: str) -> str:
    """Strip characters unsafe for filesystem directory names."""
    return re.sub(r'[^\w\-. ]', '_', value).strip()


def _augment_face(face_gray: np.ndarray) -> list:
    """
    Generate augmented variants of a grayscale face crop.

    Variants include brightness shifts, contrast scaling, and gamma
    correction so the LBPH model generalises across different lighting
    conditions (e.g. morning enrollment vs. evening attendance).

    Returns a list containing the original + augmented images,
    capped at AUGMENT_SAMPLES_PER_PHOTO.
    """
    samples = [face_gray]

    # Horizontal flip — handles slight head orientation changes
    flipped = cv2.flip(face_gray, 1)
    samples.append(flipped)

    # Brightness additive variants: ±20, ±40 for original and flipped
    for delta in (-40, -20, 20, 40):
        for src in (face_gray, flipped):
            adjusted = cv2.convertScaleAbs(src, alpha=1.0, beta=delta)
            samples.append(adjusted)

    # Contrast scaling (alpha): simulates harsh vs. soft lighting
    for alpha in (0.75, 1.3):
        samples.append(cv2.convertScaleAbs(face_gray, alpha=alpha, beta=0))
        samples.append(cv2.convertScaleAbs(flipped,    alpha=alpha, beta=0))

    # Gamma correction: simulates different ambient light hours
    # gamma < 1  = brighter (overcast / indoor light)
    # gamma > 1  = darker   (evening / shadowed)
    def _apply_gamma(img, gamma):
        table = np.array([
            min(255, int((i / 255.0) ** gamma * 255))
            for i in range(256)
        ], dtype=np.uint8)
        return cv2.LUT(img, table)

    for gamma in (0.6, 0.8, 1.4, 1.8):
        samples.append(_apply_gamma(face_gray, gamma))
        samples.append(_apply_gamma(flipped,   gamma))

    # Slight Gaussian blur (focus softness)
    samples.append(cv2.GaussianBlur(face_gray, (3, 3), 0))
    samples.append(cv2.GaussianBlur(flipped,   (3, 3), 0))

    # Return at most AUGMENT_SAMPLES_PER_PHOTO to keep things consistent
    return samples[:AUGMENT_SAMPLES_PER_PHOTO]


def enroll_from_gui(name, roll_no, frame_bgr):
    """
    Detect a single face in a BGR frame for enrollment.

    Args:
        name: Student name
        roll_no: Student roll number
        frame_bgr: BGR frame from camera

    Returns:
        tuple: (gray_face_image, face_bbox) or (None, None) if no single face found
    """
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    faces = _detector.detect(frame_bgr)

    if len(faces) == 1:
        x, y, w, h = faces[0]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        face_gray = gray[y:y+h, x:x+w]
        # Resize to standard size for LBPH consistency
        face_resized = cv2.resize(face_gray, (200, 200))
        # ── CLAHE: normalise contrast so training images are lighting-agnostic ─
        # This ensures the training distribution matches the CLAHE-normalised
        # frames used during recognition, making the model robust across
        # different lighting hours (morning enrol / evening recognise).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_resized = clahe.apply(face_resized)
        return face_resized, (x, y, w, h)

    return None, None


def save_enrollment(name, roll_no, face_images, frames=None,
                    progress_callback=None):
    """
    Save enrollment photos and retrain the LBPH model.

    Args:
        name: Student name
        roll_no: Roll number
        face_images: List of grayscale face images (200×200)
        frames: Optional list of full BGR frames to save as reference photos
        progress_callback: Optional callable(msg: str) for UI status updates
    """
    def _report(msg):
        if progress_callback:
            progress_callback(msg)

    # Save reference photos — sanitize inputs before use as path components
    student_dir = os.path.join(
        KNOWN_FACES_DIR,
        f"{_sanitize_path_component(roll_no)}_{_sanitize_path_component(name)}"
    )
    os.makedirs(student_dir, exist_ok=True)

    if frames:
        for i, frame in enumerate(frames):
            photo_path = os.path.join(student_dir, f"photo_{i + 1}.jpg")
            cv2.imwrite(photo_path, frame)

    # Save original + augmented face images
    _report("Generating augmented training samples…")
    aug_idx = 0
    for i, face_img in enumerate(face_images):
        variants = _augment_face(face_img)
        for v_img in variants:
            face_path = os.path.join(student_dir, f"face_{aug_idx:04d}.jpg")
            cv2.imwrite(face_path, v_img)
            aug_idx += 1

    _report(f"Saved {aug_idx} training images for {name}. Retraining model…")

    # Update label map
    label_map = _load_label_map()
    if name not in label_map:
        existing_ids = [v["label_id"] for v in label_map.values()]
        new_id = max(existing_ids) + 1 if existing_ids else 0
        label_map[name] = {"label_id": new_id, "roll_no": roll_no}
    _save_label_map(label_map)

    # Retrain LBPH model with ALL enrolled faces
    _train_lbph_model(progress_callback=progress_callback)
    _report("Enrollment complete ✅")


def _train_lbph_model(progress_callback=None):
    """Train/retrain the LBPH model from all saved face images."""
    def _report(msg):
        if progress_callback:
            progress_callback(msg)

    label_map = _load_label_map()
    faces  = []
    labels = []

    # Shared CLAHE instance — same parameters used in recognition inference
    # so training and prediction operate on the same normalised distribution.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for name, data in label_map.items():
        label_id = data["label_id"]
        roll_no  = data["roll_no"]
        student_dir = os.path.join(
            KNOWN_FACES_DIR,
            f"{_sanitize_path_component(roll_no)}_{_sanitize_path_component(name)}"
        )
        if not os.path.exists(student_dir):
            continue

        student_count = 0
        for filename in sorted(os.listdir(student_dir)):
            if filename.startswith("face_") and filename.endswith(".jpg"):
                filepath = os.path.join(student_dir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (200, 200))
                    # Apply CLAHE — normalises brightness/contrast so the model
                    # is invariant to different lighting at recognition time.
                    img = clahe.apply(img)
                    faces.append(img)
                    labels.append(label_id)
                    student_count += 1

        _report(f"Loaded {student_count} samples for {name}…")

    if len(faces) == 0:
        print("[FaceEncoder] No face images found for training!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces, np.array(labels))
    recognizer.save(LBPH_MODEL_FILE)
    n_students = len(label_map)
    print(
        f"[FaceEncoder] LBPH model trained with {len(faces)} images "
        f"from {n_students} student(s)."
    )
    _report(f"Model trained — {len(faces)} images, {n_students} student(s)")


def _load_label_map():
    """Load the name → label_id mapping from disk."""
    if os.path.exists(LABEL_MAP_FILE):
        with open(LABEL_MAP_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_label_map(label_map):
    """Save the label map to disk."""
    with open(LABEL_MAP_FILE, "wb") as f:
        pickle.dump(label_map, f)


def load_all_encodings():
    """
    Get enrolled student info (compatible API).

    Returns:
        dict: {name: {"roll_no": str, "label_id": int}}
    """
    return _load_label_map()


def delete_student(name):
    """
    Remove a student's data and retrain the model.

    Args:
        name: Student name to remove

    Returns:
        bool: True if student was found and removed
    """
    label_map = _load_label_map()
    if name not in label_map:
        return False

    roll_no = label_map[name]["roll_no"]
    del label_map[name]
    _save_label_map(label_map)

    # Remove photo directory — sanitize to guard against path traversal
    student_dir = os.path.join(
        KNOWN_FACES_DIR,
        f"{_sanitize_path_component(roll_no)}_{_sanitize_path_component(name)}"
    )
    if os.path.exists(student_dir):
        shutil.rmtree(student_dir)

    if label_map:
        _train_lbph_model()
    elif os.path.exists(LBPH_MODEL_FILE):
        os.remove(LBPH_MODEL_FILE)

    return True


def get_enrolled_students():
    """
    Get a list of all enrolled students.

    Returns:
        list: List of dicts with 'name' and 'roll_no' keys
    """
    label_map = _load_label_map()
    return [
        {"name": name, "roll_no": data["roll_no"]}
        for name, data in label_map.items()
    ]
