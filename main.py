"""
Smart Attendance Management System
Main application with tkinter GUI.
Uses YOLOv8-face detection + LBPH face recognition.

Features:
  - Enroll students via webcam face capture (with augmentation)
  - Take attendance with real-time face recognition
  - Auto-log attendance to Excel sheets
  - Live FPS counter on camera feed
  - All heavy inference runs in background worker threads
"""
import os
import sys
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import numpy as np

from config import (
    WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    BG_COLOR, FG_COLOR, ACCENT_COLOR, BUTTON_COLOR,
    SUCCESS_COLOR, WARNING_COLOR, COLOR_RECOGNIZED_HEX,
    COLOR_UNKNOWN_HEX, CAMERA_INDEX, NUM_ENROLLMENT_PHOTOS,
    SHOW_FPS_COUNTER, MIN_ATTENDANCE_CONFIDENCE
)
from face_encoder import (
    enroll_from_gui, save_enrollment, load_all_encodings,
    get_enrolled_students, delete_student
)
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from attendance_logger import AttendanceLogger
from attendance_popup import show_attendance_popup


class AttendanceApp:
    """Main application class for the Attendance Management System."""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(True, True)

        # ── Camera state ────────────────────────────────────────────────
        self.cap              = None
        self.is_camera_active = False
        self.current_mode     = None     # 'enroll' or 'attendance'
        self.recognizer       = None
        self.logger           = None
        self.detector         = FaceDetector()   # shared detector

        # ── Worker-thread plumbing ───────────────────────────────────────
        # Worker thread does ALL heavy inference + PIL resize; main thread
        # only copies the pre-rendered PhotoImage to the label.
        self._frame_queue  = queue.Queue(maxsize=1)
        # Cache holds (imgtk, results) produced by the worker
        self._result_cache = None
        self._result_lock  = threading.Lock()
        self._worker_thread = None
        self._worker_stop   = threading.Event()

        # ── Camera-feed after-loop tracker ───────────────────────────────
        # Stores the root.after ID so we can cancel it before switching modes,
        # preventing duplicate _update_camera_feed loops running simultaneously.
        self._feed_after_id = None

        # ── Enrollment state ─────────────────────────────────────────────
        self.enrollment_face_images = []
        self.enrollment_frames      = []
        self.enrollment_name        = ""
        self.enrollment_roll        = ""
        self._enroll_scan_start     = 0.0

        # ── Enroll-mode FPS (for enrollment feed) ────────────────────────
        self._enroll_fps_t0    = time.perf_counter()
        self._enroll_fps_count = 0
        self._enroll_fps_last  = 0.0

        # ── UI setup ─────────────────────────────────────────────────────
        self._setup_styles()
        self._build_header()
        self._build_main_area()
        self._build_status_bar()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto-start attendance mode if students are already enrolled.
        # Run silent retrain FIRST (background) with on_complete callback so
        # attendance mode only starts once the updated CLAHE model is on disk.
        enrolled = get_enrolled_students()
        if enrolled:
            self._set_status("🔄 Updating recognition model… please wait")
            self._silent_retrain(
                on_complete=lambda: self.root.after(0, self._start_attendance_mode)
            )

    # ─── STYLES ──────────────────────────────────────────────────────────

    def _setup_styles(self):
        """Configure ttk styles for a dark, modern theme."""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.style.configure("Header.TFrame",  background="#0f0f23")
        self.style.configure("Main.TFrame",    background=BG_COLOR)
        self.style.configure("Card.TFrame",    background=ACCENT_COLOR)
        self.style.configure("Status.TFrame",  background="#0a0a1a")

        self.style.configure("Header.TLabel",
                             background="#0f0f23", foreground="#ffffff",
                             font=("Segoe UI", 18, "bold"))
        self.style.configure("SubHeader.TLabel",
                             background="#0f0f23", foreground="#8888aa",
                             font=("Segoe UI", 10))
        self.style.configure("Card.TLabel",
                             background=ACCENT_COLOR, foreground=FG_COLOR,
                             font=("Segoe UI", 11))
        self.style.configure("CardTitle.TLabel",
                             background=ACCENT_COLOR, foreground="#ffffff",
                             font=("Segoe UI", 13, "bold"))
        self.style.configure("Status.TLabel",
                             background="#0a0a1a", foreground="#8888aa",
                             font=("Segoe UI", 9))
        self.style.configure("Success.TLabel",
                             background=ACCENT_COLOR, foreground=SUCCESS_COLOR,
                             font=("Segoe UI", 11, "bold"))

        self.style.configure("Action.TButton",
                             background=BUTTON_COLOR, foreground="#ffffff",
                             font=("Segoe UI", 11, "bold"), padding=(20, 10))
        self.style.map("Action.TButton",
                       background=[("active", "#1e3a5f"), ("pressed", "#0d2137")])

        self.style.configure("Danger.TButton",
                             background="#b71c1c", foreground="#ffffff",
                             font=("Segoe UI", 10, "bold"), padding=(15, 8))
        self.style.map("Danger.TButton",
                       background=[("active", "#d32f2f")])

        self.style.configure("Success.TButton",
                             background="#1b5e20", foreground="#ffffff",
                             font=("Segoe UI", 11, "bold"), padding=(20, 10))
        self.style.map("Success.TButton",
                       background=[("active", "#2e7d32")])

    # ─── UI BUILDERS ────────────────────────────────────────────────────

    def _build_header(self):
        header = ttk.Frame(self.root, style="Header.TFrame", height=80)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title_frame = ttk.Frame(header, style="Header.TFrame")
        title_frame.pack(side=tk.LEFT, padx=20, pady=10)
        ttk.Label(title_frame, text="📷 Smart Attendance System",
                  style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(title_frame, text="YOLOv8 Face Detection • LBPH Recognition",
                  style="SubHeader.TLabel").pack(anchor=tk.W)

        btn_frame = ttk.Frame(header, style="Header.TFrame")
        btn_frame.pack(side=tk.RIGHT, padx=20, pady=15)

        self.btn_enroll = ttk.Button(btn_frame, text="📝 Enroll Students",
                                     style="Action.TButton",
                                     command=self._start_enrollment_mode)
        self.btn_enroll.pack(side=tk.LEFT, padx=5)

        self.btn_attendance = ttk.Button(btn_frame, text="✅ Take Attendance",
                                         style="Success.TButton",
                                         command=self._start_attendance_mode)
        self.btn_attendance.pack(side=tk.LEFT, padx=5)

    def _build_main_area(self):
        self.main_frame = ttk.Frame(self.root, style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.camera_frame = ttk.Frame(self.main_frame, style="Card.TFrame")
        self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.camera_label = tk.Label(
            self.camera_frame, bg="#000000",
            text="Select a mode to start\n\n"
                 "📝 Enroll Students — Register new faces\n"
                 "✅ Take Attendance — Auto-mark present students",
            fg="#555555", font=("Segoe UI", 13), justify=tk.CENTER
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.side_panel = ttk.Frame(self.main_frame, style="Card.TFrame", width=320)
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        self.side_panel.pack_propagate(False)

        self._build_default_side_panel()

    def _build_default_side_panel(self):
        for w in self.side_panel.winfo_children():
            w.destroy()

        ttk.Label(self.side_panel, text="Enrolled Students",
                  style="CardTitle.TLabel").pack(pady=(15, 10), padx=15, anchor=tk.W)

        list_frame = tk.Frame(self.side_panel, bg=ACCENT_COLOR)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.student_listbox = tk.Listbox(
            list_frame, bg="#0a0a1a", fg=FG_COLOR,
            font=("Segoe UI", 10), selectbackground="#1e3a5f",
            borderwidth=0, highlightthickness=0,
            yscrollcommand=scrollbar.set
        )
        self.student_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.student_listbox.yview)

        self._refresh_student_list()

        btn_frame = tk.Frame(self.side_panel, bg=ACCENT_COLOR)
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        ttk.Button(btn_frame, text="🗑️ Delete Selected",
                   style="Danger.TButton",
                   command=self._delete_selected_student).pack(fill=tk.X, pady=(0, 6))

        ttk.Button(btn_frame, text="🔁 Retrain Model",
                   style="Action.TButton",
                   command=self._retrain_model).pack(fill=tk.X)

    def _build_enrollment_panel(self):
        for w in self.side_panel.winfo_children():
            w.destroy()

        ttk.Label(self.side_panel, text="Enroll New Student",
                  style="CardTitle.TLabel").pack(pady=(15, 15), padx=15, anchor=tk.W)

        form_frame = tk.Frame(self.side_panel, bg=ACCENT_COLOR)
        form_frame.pack(fill=tk.X, padx=15)

        tk.Label(form_frame, text="Student Name:", bg=ACCENT_COLOR, fg=FG_COLOR,
                 font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(5, 2))
        self.name_entry = tk.Entry(form_frame, bg="#0a0a1a", fg="#ffffff",
                                    font=("Segoe UI", 11), insertbackground="#ffffff",
                                    borderwidth=1, relief="solid")
        self.name_entry.pack(fill=tk.X, pady=(0, 10), ipady=5)

        tk.Label(form_frame, text="Roll Number:", bg=ACCENT_COLOR, fg=FG_COLOR,
                 font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(5, 2))
        self.roll_entry = tk.Entry(form_frame, bg="#0a0a1a", fg="#ffffff",
                                    font=("Segoe UI", 11), insertbackground="#ffffff",
                                    borderwidth=1, relief="solid")
        self.roll_entry.pack(fill=tk.X, pady=(0, 15), ipady=5)

        self.capture_btn = ttk.Button(
            form_frame,
            text=f"📸 Capture (0/{NUM_ENROLLMENT_PHOTOS})",
            style="Success.TButton",
            command=self._capture_enrollment_photo
        )
        self.capture_btn.pack(fill=tk.X, pady=(0, 5))

        self.save_enroll_btn = ttk.Button(
            form_frame, text="💾 Save Enrollment",
            style="Action.TButton",
            command=self._save_enrollment,
            state=tk.DISABLED
        )
        self.save_enroll_btn.pack(fill=tk.X, pady=(5, 5))

        ttk.Button(form_frame, text="❌ Cancel",
                   style="Danger.TButton",
                   command=self._stop_camera).pack(fill=tk.X, pady=(5, 10))

        self.enroll_status = tk.Label(
            self.side_panel,
            text="Enter details and capture photos",
            bg=ACCENT_COLOR, fg="#8888aa",
            font=("Segoe UI", 10), wraplength=280
        )
        self.enroll_status.pack(pady=10, padx=15)

        self.progress_frame = tk.Frame(self.side_panel, bg=ACCENT_COLOR)
        self.progress_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

    def _build_attendance_panel(self):
        for w in self.side_panel.winfo_children():
            w.destroy()

        ttk.Label(self.side_panel, text="Attendance Today",
                  style="CardTitle.TLabel").pack(pady=(15, 5), padx=15, anchor=tk.W)

        date_str = datetime.now().strftime("%B %d, %Y")
        tk.Label(self.side_panel, text=f"📅 {date_str}",
                 bg=ACCENT_COLOR, fg="#8888aa",
                 font=("Segoe UI", 10)).pack(padx=15, anchor=tk.W)

        self.attendance_counter = tk.Label(
            self.side_panel, text="Present: 0",
            bg=ACCENT_COLOR, fg=SUCCESS_COLOR,
            font=("Segoe UI", 16, "bold")
        )
        self.attendance_counter.pack(pady=(10, 5), padx=15, anchor=tk.W)

        list_frame = tk.Frame(self.side_panel, bg=ACCENT_COLOR)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 10))

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.attendance_listbox = tk.Listbox(
            list_frame, bg="#0a0a1a", fg=FG_COLOR,
            font=("Segoe UI", 10), selectbackground="#1e3a5f",
            borderwidth=0, highlightthickness=0,
            yscrollcommand=scrollbar.set
        )
        self.attendance_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.attendance_listbox.yview)

        ttk.Button(self.side_panel, text="⏹️ Stop & Save Attendance",
                   style="Danger.TButton",
                   command=self._stop_attendance).pack(pady=(5, 15), padx=15, fill=tk.X)

    def _build_status_bar(self):
        status_frame = ttk.Frame(self.root, style="Status.TFrame", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = ttk.Label(
            status_frame,
            text="Ready • Select a mode to begin",
            style="Status.TLabel"
        )
        self.status_label.pack(side=tk.LEFT, padx=15, pady=5)

        self.enrolled_count_label = ttk.Label(
            status_frame, text="", style="Status.TLabel"
        )
        self.enrolled_count_label.pack(side=tk.RIGHT, padx=15, pady=5)
        self._update_enrolled_count()

    # ─── CAMERA CONTROL ──────────────────────────────────────────────────

    def _cancel_feed_loop(self):
        """Cancel any pending _update_camera_feed after-callback."""
        if self._feed_after_id is not None:
            try:
                self.root.after_cancel(self._feed_after_id)
            except Exception:
                pass
            self._feed_after_id = None

    def _start_camera(self):
        """Open the webcam and clear stale cached results."""
        self._cancel_feed_loop()           # stop any running feed loop
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error",
                                 "Could not open camera. Check if webcam is connected.")
            return False
        self.is_camera_active = True
        # Clear stale results from previous mode so they don't bleed through
        with self._result_lock:
            self._result_cache = None
        return True

    def _start_worker(self, mode):
        """Launch the background inference + render worker thread."""
        self._stop_worker()
        self._worker_stop.clear()
        with self._result_lock:
            self._result_cache = None

        target = (self._worker_attendance if mode == "attendance"
                  else self._worker_enroll)
        self._worker_thread = threading.Thread(target=target, daemon=True)
        self._worker_thread.start()

    def _stop_worker(self):
        """Signal and join the worker thread."""
        self._worker_stop.set()
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2)
        self._worker_thread = None

    # ─── WORKER THREADS ──────────────────────────────────────────────────
    # Each worker:
    #   1. Pulls a raw BGR frame from _frame_queue
    #   2. Runs heavy inference (YOLO + LBPH + MediaPipe throttled)
    #   3. Annotates the frame
    #   4. Converts annotated frame → PIL image → ImageTk.PhotoImage
    #      (PIL work happens HERE, NOT on the main thread)
    #   5. Stores (imgtk, results) in _result_cache under a lock

    def _worker_attendance(self):
        """Background: YOLO + LBPH + throttled MediaPipe mesh + PIL render."""
        while not self._worker_stop.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            # Snapshot into a local ref — prevents TOCTOU race where the main
            # thread sets self.recognizer = None between recognize_frame and
            # draw_results, causing AttributeError.
            recognizer = self.recognizer
            if recognizer is None:
                continue

            results   = recognizer.recognize_frame(frame)
            annotated = recognizer.draw_results(frame.copy(), results)

            # ── PIL resize + PhotoImage conversion — OFF the main thread ──
            imgtk = self._frame_to_imgtk(annotated)

            with self._result_lock:
                self._result_cache = (imgtk, results)

    def _worker_enroll(self):
        """Background: YOLO detection + scan-line annotation + FPS + PIL render."""
        fps_t0    = time.perf_counter()
        fps_count = 0
        fps_last  = 0.0

        while not self._worker_stop.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break

            faces    = self.detector.detect(frame)
            annotated = frame.copy()
            elapsed  = time.time() - self._enroll_scan_start

            for (x, y, w, h) in faces:
                color      = (0, 255, 0)
                glow_color = (0, 120, 0)
                corner_len = int(min(w, h) * 0.25)
                x2, y2    = x + w, y + h

                # Outer glow brackets
                for (cx, cy, dx, dy) in [
                    (x-2, y-2, 1, 1), (x2+2, y-2, -1, 1),
                    (x-2, y2+2, 1, -1), (x2+2, y2+2, -1, -1)
                ]:
                    cv2.line(annotated, (cx, cy), (cx + dx*corner_len, cy), glow_color, 1)
                    cv2.line(annotated, (cx, cy), (cx, cy + dy*corner_len), glow_color, 1)

                # Main brackets
                for (cx, cy, dx, dy) in [
                    (x, y, 1, 1), (x2, y, -1, 1),
                    (x, y2, 1, -1), (x2, y2, -1, -1)
                ]:
                    cv2.line(annotated, (cx, cy), (cx + dx*corner_len, cy), color, 2)
                    cv2.line(annotated, (cx, cy), (cx, cy + dy*corner_len), color, 2)

                # Animated scan line
                cycle    = elapsed % 2.0
                progress = cycle / 2.0 if cycle < 1.0 else (2.0 - cycle) / 1.0
                scan_y   = int(y + h * progress)
                ov       = annotated.copy()
                cv2.line(ov, (x, scan_y), (x + w, scan_y), color, 1)
                glow_h   = 6
                cv2.rectangle(ov,
                              (x, max(y, scan_y - glow_h)),
                              (x + w, min(y + h, scan_y + glow_h)),
                              color, -1)
                cv2.addWeighted(ov, 0.15, annotated, 0.85, 0, annotated)
                cv2.line(annotated, (x + 2, scan_y), (x + w - 2, scan_y), color, 1)

            # Status banner
            if len(faces) == 1:
                msg, mc = "FACE LOCKED — READY TO CAPTURE", (0, 255, 0)
            elif len(faces) == 0:
                msg, mc = "NO FACE DETECTED — MOVE CLOSER", (0, 0, 255)
            else:
                msg, mc = "MULTIPLE FACES — ONE PERSON ONLY", (0, 0, 255)

            ov2 = annotated.copy()
            cv2.rectangle(ov2, (5, 5), (510, 35), (0, 0, 0), -1)
            cv2.addWeighted(ov2, 0.5, annotated, 0.5, 0, annotated)
            cv2.putText(annotated, msg, (12, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, mc, 1, cv2.LINE_AA)

            # FPS counter (enrollment mode)
            if SHOW_FPS_COUNTER:
                fps_count += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    fps_last  = fps_count / (now - fps_t0)
                    fps_count = 0
                    fps_t0    = now
                fps_text = f"FPS: {fps_last:.1f}"
                font     = cv2.FONT_HERSHEY_SIMPLEX
                scale    = 0.6
                thick    = 1
                (tw, th), _ = cv2.getTextSize(fps_text, font, scale, thick)
                h_f, w_f   = annotated.shape[:2]
                margin  = 10
                x_pos   = w_f - tw - margin - 4
                y_pos   = margin + th + 4
                cv2.rectangle(annotated,
                              (x_pos - 4, margin),
                              (x_pos + tw + 4, margin + th + 8),
                              (20, 20, 20), -1)
                cv2.rectangle(annotated,
                              (x_pos - 4, margin),
                              (x_pos + tw + 4, margin + th + 8),
                              (0, 180, 100), 1)
                fps_color = (0, 220, 80) if fps_last >= 25 else (
                    (0, 165, 220) if fps_last >= 15 else (0, 60, 220))
                cv2.putText(annotated, fps_text, (x_pos, y_pos + 2),
                            font, scale, fps_color, thick, cv2.LINE_AA)

            imgtk = self._frame_to_imgtk(annotated)

            with self._result_lock:
                self._result_cache = (imgtk, faces)

    def _frame_to_imgtk(self, bgr_frame):
        """
        Convert BGR OpenCV frame → PIL Image → ImageTk.PhotoImage.
        Resized to fit the current camera_label dimensions.
        Runs ENTIRELY in the worker thread (never on the main thread).
        """
        frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb)

        # Read label size (winfo_width/height are safe to call from non-main threads
        # as read-only operations on Tkinter integer atoms)
        label_w = self.camera_label.winfo_width()
        label_h = self.camera_label.winfo_height()
        if label_w > 10 and label_h > 10:
            img = self._resize_with_aspect_ratio(img, label_w, label_h)

        # NOTE: ImageTk.PhotoImage MUST be created on the main thread on some
        # platforms. We create it here in the worker for speed, but keep the
        # reference alive via _result_cache until the next frame replaces it.
        return ImageTk.PhotoImage(image=img)

    def _resize_with_aspect_ratio(self, img, max_w, max_h):
        w, h  = img.size
        ratio = min(max_w / w, max_h / h)
        return img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    def _stop_camera(self):
        """Close the webcam, stop worker, and reset to default state."""
        self.is_camera_active = False
        self.current_mode     = None
        self._stop_worker()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if not self.root.winfo_exists():
            return

        self.camera_label.configure(
            image='',
            text="Select a mode to start\n\n"
                 "📝 Enroll Students — Register new faces\n"
                 "✅ Take Attendance — Auto-mark present students",
            fg="#555555", font=("Segoe UI", 13)
        )
        self._build_default_side_panel()
        self._set_status("Ready • Select a mode to begin")

    # ─── CAMERA FEED LOOP (main thread) ──────────────────────────────────

    def _update_camera_feed(self):
        """
        Main-thread loop — only grabs raw frames and pushes to worker;
        displays whatever pre-rendered ImageTk the worker last produced.
        All PIL + inference work runs in the worker thread.
        """
        self._feed_after_id = None   # Clear — we are now executing

        if not self.root.winfo_exists():
            return
        if not self.is_camera_active or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._feed_after_id = self.root.after(33, self._update_camera_feed)
            return

        frame = cv2.flip(frame, 1)   # Mirror

        # Push raw frame to worker (drop if worker is still busy)
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        # Read cached (imgtk, results) atomically
        with self._result_lock:
            cached = self._result_cache

        if cached is not None:
            imgtk, results = cached

            # Mark attendance for newly recognised faces.
            # Three independent guards must ALL pass before attendance is marked:
            #  1. result["confirmed"] — vote window reached VOTE_THRESHOLD agreement
            #  2. name is a real enrolled student (not Unknown / SCANNING...)
            #  3. confidence >= MIN_ATTENDANCE_CONFIDENCE (hard floor at 35%)
            #
            # Guard 3 was the key missing piece: the screenshot showed a 9%
            # confidence match for an absent student still triggering mark_present()
            # because guards 1 & 2 passed on a brief vote-window anomaly.
            if (self.current_mode == "attendance" and self.logger
                    and results and isinstance(results[0], dict)):
                for result in results:
                    if (
                        result.get("confirmed", False)                          # gate 1
                        and result.get("name", "Unknown") not in ("Unknown", "SCANNING...")  # gate 2
                        and result.get("confidence", 0.0) >= MIN_ATTENDANCE_CONFIDENCE       # gate 3
                    ):
                        newly_marked = self.logger.mark_present(
                            result["name"], result["roll_no"]
                        )
                        if newly_marked:
                            self.root.after(0, self._update_attendance_list)
                            # Immediately flush the attendance record to the
                            # Excel file in a background thread so the
                            # Streamlit dashboard reflects this mark within
                            # seconds (rather than waiting up to 60 s for the
                            # auto-save loop to fire).
                            _logger = self.logger
                            threading.Thread(
                                target=_logger.save, daemon=True
                            ).start()
                            # Show attendance-history popup — pass today's date
                            # so the popup shows the student as PRESENT
                            # immediately (in-memory mark visible right away).
                            _name      = result["name"]
                            _roll_no   = result["roll_no"]
                            _mark_date = self.logger.date_str   # live property
                            self.root.after(
                                200,
                                lambda n=_name, r=_roll_no, d=_mark_date:
                                    show_attendance_popup(self.root, n, r, d)
                            )

            # Display the pre-rendered image — this is now INSTANT on main thread
            if self.root.winfo_exists():
                self.camera_label.configure(image=imgtk, text="")
                self.camera_label.image = imgtk   # prevent GC

        # ≈30 FPS display refresh
        self._feed_after_id = self.root.after(33, self._update_camera_feed)

    # ─── ENROLLMENT MODE ──────────────────────────────────────────────────

    def _start_enrollment_mode(self):
        self.is_camera_active = False
        self._cancel_feed_loop()   # stop any running feed loop FIRST
        self._stop_worker()
        if not self._start_camera():
            return

        self._enroll_scan_start = time.time()
        self.current_mode = "enroll"
        self.enrollment_face_images = []
        self.enrollment_frames      = []

        self._build_enrollment_panel()
        self._set_status("Enrollment Mode • Enter student details and capture photos")
        self._start_worker("enroll")
        self._feed_after_id = self.root.after(100, self._update_camera_feed)

    def _capture_enrollment_photo(self):
        name = self.name_entry.get().strip()
        roll = self.roll_entry.get().strip()

        if not name or not roll:
            self.enroll_status.configure(
                text="⚠️ Please enter name and roll number!",
                fg=WARNING_COLOR
            )
            return

        self.enrollment_name = name
        self.enrollment_roll = roll

        if self.cap is None or not self.cap.isOpened():
            self.enroll_status.configure(text="⚠️ Camera not available!", fg=WARNING_COLOR)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.enroll_status.configure(text="⚠️ Could not read frame!", fg=WARNING_COLOR)
            return

        frame = cv2.flip(frame, 1)

        face_img, bbox = enroll_from_gui(name, roll, frame)

        if face_img is not None:
            self.enrollment_face_images.append(face_img)
            self.enrollment_frames.append(frame.copy())
            count = len(self.enrollment_face_images)

            self.capture_btn.configure(text=f"📸 Capture ({count}/{NUM_ENROLLMENT_PHOTOS})")
            self.enroll_status.configure(
                text=(
                    f"✅ Photo {count} captured! "
                    f"{'Capture more…' if count < NUM_ENROLLMENT_PHOTOS else 'Ready to save!'}"
                ),
                fg=SUCCESS_COLOR
            )
            self._update_enrollment_progress(count)

            if count >= NUM_ENROLLMENT_PHOTOS:
                self.capture_btn.configure(state=tk.DISABLED)
                self.save_enroll_btn.configure(state=tk.NORMAL)
        else:
            self.enroll_status.configure(
                text="⚠️ No face detected (or multiple faces). Try again.",
                fg=WARNING_COLOR
            )

    def _update_enrollment_progress(self, count):
        for w in self.progress_frame.winfo_children():
            w.destroy()
        for i in range(NUM_ENROLLMENT_PHOTOS):
            color = SUCCESS_COLOR if i < count else "#333333"
            tk.Label(self.progress_frame, text="●", fg=color,
                     bg=ACCENT_COLOR, font=("Segoe UI", 16)).pack(side=tk.LEFT, padx=5)

    def _save_enrollment(self):
        if len(self.enrollment_face_images) < NUM_ENROLLMENT_PHOTOS:
            self.enroll_status.configure(
                text=f"⚠️ Need {NUM_ENROLLMENT_PHOTOS} photos. "
                     f"Have {len(self.enrollment_face_images)}.",
                fg=WARNING_COLOR
            )
            return

        try:
            self.save_enroll_btn.configure(state=tk.DISABLED)
            self.capture_btn.configure(state=tk.DISABLED)
        except Exception:
            pass

        self.enroll_status.configure(text="💾 Saving & training model…", fg="#8888aa")
        self.root.update_idletasks()

        name      = self.enrollment_name
        roll      = self.enrollment_roll
        face_imgs = list(self.enrollment_face_images)
        frames    = list(self.enrollment_frames)

        def _progress(msg):
            """Called from worker thread — schedule GUI update on main thread."""
            if self.root.winfo_exists():
                self.root.after(0, lambda m=msg: self._safe_set_enroll_status(m))

        def _do_save():
            try:
                save_enrollment(name, roll, face_imgs, frames,
                                progress_callback=_progress)
                self.root.after(0, lambda: self._on_enrollment_saved(name, roll, len(face_imgs)))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to save enrollment:\n{str(e)}"
                ))

        threading.Thread(target=_do_save, daemon=True).start()

    def _safe_set_enroll_status(self, msg):
        """Update enroll_status label safely (may not exist if cancelled)."""
        try:
            if self.root.winfo_exists() and hasattr(self, 'enroll_status'):
                self.enroll_status.configure(text=msg, fg="#8888aa")
        except Exception:
            pass

    def _on_enrollment_saved(self, name, roll, photo_count):
        if not self.root.winfo_exists():
            return
        messagebox.showinfo(
            "Success",
            f"✅ {name} enrolled successfully!\n"
            f"Roll No: {roll}\n"
            f"Photos: {photo_count} captures → "
            f"{photo_count * 12} training samples"
        )
        self.enrollment_face_images = []
        self.enrollment_frames      = []
        self._update_enrolled_count()
        self._start_enrollment_mode()   # Reset for next student

    # ─── ATTENDANCE MODE ──────────────────────────────────────────────────

    def _start_attendance_mode(self):
        enrolled = get_enrolled_students()
        if not enrolled:
            messagebox.showwarning("No Students",
                                   "No students enrolled yet!\n"
                                   "Please enroll students first.")
            return

        self.is_camera_active = False
        self._cancel_feed_loop()   # stop any running feed loop FIRST
        self._stop_worker()
        if not self._start_camera():
            return

        self.current_mode = "attendance"
        self.recognizer   = None
        self.logger       = None

        self._build_attendance_panel()
        self._set_status("Attendance Mode • Loading recognition model, please wait…")
        self._feed_after_id = self.root.after(100, self._update_camera_feed)

        def _load_models():
            try:
                recognizer = FaceRecognizer()
                logger     = AttendanceLogger()
                self.root.after(0, lambda: self._on_models_loaded(
                    recognizer, logger, len(enrolled)
                ))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Model Load Error",
                    f"Failed to load face recognition model:\n{e}\n\n"
                    "Make sure students are enrolled first."
                ))

        threading.Thread(target=_load_models, daemon=True).start()

    def _on_models_loaded(self, recognizer, logger, num_enrolled):
        if not self.root.winfo_exists():
            return
        self.recognizer = recognizer
        self.logger     = logger
        self._start_worker("attendance")
        self._set_status(
            f"Attendance Mode • Recognising faces • {num_enrolled} student(s) enrolled"
        )

    def _update_attendance_list(self):
        if not self.logger or not self.root.winfo_exists():
            return
        present = self.logger.get_present_list()
        try:
            self.attendance_counter.configure(text=f"Present: {len(present)}")
            self.attendance_listbox.delete(0, tk.END)
            for i, entry in enumerate(present, 1):
                self.attendance_listbox.insert(
                    tk.END,
                    f" {i}. {entry['name']} ({entry['roll_no']}) - {entry['time']}"
                )
                self.attendance_listbox.itemconfig(
                    self.attendance_listbox.size() - 1, fg=SUCCESS_COLOR
                )
        except tk.TclError:
            pass

    def _stop_attendance(self):
        if self.logger:
            summary = self.logger.get_summary()
            self.logger.close()
            messagebox.showinfo(
                "Attendance Saved",
                f"📊 Attendance saved successfully!\n\n"
                f"Date: {summary['date']}\n"
                f"Present: {summary['total_present']}\n"
                f"File: {summary['file_path']}"
            )
            self.logger = None
        self.recognizer = None
        self._stop_camera()

    # ─── HELPERS ──────────────────────────────────────────────────────────

    def _refresh_student_list(self):
        if not hasattr(self, 'student_listbox'):
            return
        self.student_listbox.delete(0, tk.END)
        students = get_enrolled_students()
        if students:
            for i, s in enumerate(students, 1):
                self.student_listbox.insert(tk.END, f" {i}. {s['name']} (Roll: {s['roll_no']})")
        else:
            self.student_listbox.insert(tk.END, "  No students enrolled yet")

    def _retrain_model(self):
        """Re-train the LBPH model and hot-reload it in the running session."""
        students = get_enrolled_students()
        if not students:
            messagebox.showwarning("No Students",
                                   "No students enrolled yet. Enroll first.")
            return

        self._set_status("🔁 Retraining model with improved augmentation…")

        def _do_retrain():
            try:
                from face_encoder import _train_lbph_model
                _train_lbph_model()

                # ── Hot-reload: update the in-memory recognizer immediately ──
                # If attendance mode is active, swap the model without stopping
                # the camera so recognition improves in real time.
                recognizer_snap = self.recognizer
                if recognizer_snap is not None:
                    try:
                        recognizer_snap.reload_model()
                    except Exception as re:
                        print(f"[AttendanceApp] Hot-reload failed: {re}")

                def _on_done():
                    self._set_status("Ready • Model retrained — recognition improved!")
                    messagebox.showinfo(
                        "Retrain Complete",
                        f"✅ Model retrained with {len(students)} student(s)!\n\n"
                        "Recognition is now more accurate across\n"
                        "different lighting conditions (day vs. evening).\n\n"
                        "Changes are active immediately — no restart needed."
                    )
                self.root.after(0, _on_done)
            except Exception as e:
                self.root.after(0, lambda: (
                    self._set_status("Retrain failed — see error dialog"),
                    messagebox.showerror("Retrain Failed",
                                         f"Could not retrain model:\n{e}")
                ))

        threading.Thread(target=_do_retrain, daemon=True).start()

    def _silent_retrain(self, on_complete=None):
        """
        Silently retrain the LBPH model in a background thread.
        No dialog shown — status bar updates only.
        Calls on_complete() on the main thread when done (or on error).

        This ensures existing enrolled students benefit from the latest
        CLAHE + gamma augmentation without needing a manual retrain click.
        The on_complete callback is used to chain attendance-mode startup
        so the recognizer always loads the freshly trained model.
        """
        def _do():
            try:
                from face_encoder import _train_lbph_model
                _train_lbph_model()
                if self.root.winfo_exists():
                    self.root.after(
                        0, lambda: self._set_status(
                            "Ready • Model updated with improved lighting robustness"
                        )
                    )
            except Exception as e:
                print(f"[AttendanceApp] Silent retrain failed: {e}")
                if self.root.winfo_exists():
                    self.root.after(
                        0, lambda: self._set_status("Ready • Model update skipped")
                    )
            finally:
                # Always fire the callback so the app doesn't get stuck
                if on_complete and self.root.winfo_exists():
                    on_complete()

        threading.Thread(target=_do, daemon=True).start()

    def _delete_selected_student(self):
        if not hasattr(self, 'student_listbox'):
            return
        selection = self.student_listbox.curselection()
        if not selection:
            messagebox.showwarning("Select Student", "Please select a student to delete.")
            return

        students = get_enrolled_students()
        idx = selection[0]
        if idx >= len(students):
            return

        student = students[idx]
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Delete {student['name']} (Roll: {student['roll_no']})?\n"
            f"This will remove all their face data."
        )
        if confirm:
            delete_student(student["name"])
            self._refresh_student_list()
            self._update_enrolled_count()
            self._set_status(f"Deleted {student['name']}")

    def _update_enrolled_count(self):
        if not self.root.winfo_exists():
            return
        students = get_enrolled_students()
        self.enrolled_count_label.configure(text=f"Enrolled: {len(students)} student(s)")

    def _set_status(self, text):
        if self.root.winfo_exists():
            self.status_label.configure(text=text)

    def _on_close(self):
        self.is_camera_active = False
        self._stop_worker()
        if self.cap is not None:
            self.cap.release()
        if self.logger:
            self.logger.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app  = AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
