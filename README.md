# 📷 Smart Attendance Management System

A computer vision-based attendance management system that uses **real-time face recognition** to automatically mark student attendance. Built with Python, YOLOv8, OpenCV, and Tkinter.

---

## ✨ Features

- **Face Enrollment** — Register students by capturing multiple face photos via webcam
- **Real-Time Recognition** — Automatically identify enrolled students from live camera feed
- **Auto Attendance Logging** — Attendance is recorded to styled Excel (`.xlsx`) files instantly
- **Modern Dark GUI** — Sleek, dark-themed Tkinter interface with status indicators
- **No dlib Required** — Uses YOLOv8-face detector + LBPH recognizer (no C++ compilation needed)
- **Auto Model Download** — YOLO face detection model is downloaded automatically on first run
- **Student Management** — View, enroll, and delete students from the GUI

---

## 🛠️ Tech Stack

| Component            | Technology                                      |
| -------------------- | ----------------------------------------------- |
| **Language**         | Python 3.8+                                     |
| **Face Detection**   | YOLOv8-face (`ultralytics` + `yolov8n-face.pt`) |
| **Face Recognition** | OpenCV LBPH (Local Binary Patterns Histograms)  |
| **GUI**              | Tkinter + ttk (dark theme with custom styling)  |
| **Image Processing** | OpenCV, NumPy, Pillow                           |
| **Attendance Logs**  | openpyxl (styled Excel spreadsheets)            |

---

## 📁 Project Structure

```
attendance_system/
├── main.py                  # Main application & GUI (entry point)
├── config.py                # All configuration constants & paths
├── face_detector.py         # YOLOv8-face detection module
├── face_encoder.py          # Student enrollment & LBPH model training
├── face_recognizer.py       # Real-time face recognition module
├── attendance_logger.py     # Excel attendance logging module
├── requirements.txt         # Python dependencies
├── known_faces/             # Enrolled student face photos
│   └── <rollno_name>/       # Per-student directory
│       ├── face_1.jpg       # Cropped grayscale face (200×200)
│       ├── face_2.jpg
│       └── photo_1.jpg      # Full reference frame
├── encodings/               # Trained model & label data
│   ├── lbph_model.yml       # Trained LBPH recognizer model
│   └── label_map.pkl        # Name ↔ label ID mapping
├── attendance_records/      # Generated attendance Excel files
│   └── attendance_YYYY-MM-DD.xlsx
└── models/                  # YOLO model file (auto-downloaded)
    └── yolov8n-face.pt      # YOLOv8 face detection model
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** installed
- A **webcam** connected to your computer

### Installation

1.  **Clone / navigate to the project directory:**

    ```bash
    cd d:/ML/attendance_system
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    This installs:
    | Package | Version | Purpose |
    | ------------------------ | --------- | ----------------------------------------- |
    | `opencv-contrib-python` | 4.13.0.92 | Face recognition (LBPH), image processing |
    | `ultralytics` | latest | YOLOv8 face detection |
    | `numpy` | 2.4.2 | Array operations |
    | `openpyxl` | 3.1.5 | Excel file read/write |
    | `Pillow` | 12.1.1 | Image display in Tkinter |

3.  **Run the application:**

    ```bash
    python main.py
    ```

    > On first run, the YOLO face detection model (~10 MB) will be **downloaded automatically**.

---

## 📖 Usage Guide

### 1. Enroll Students

1.  Click **📝 Enroll Students** in the header
2.  Enter the student's **Name** and **Roll Number** in the side panel
3.  Face the camera — a yellow bounding box appears when a face is detected
4.  Click **📸 Capture** to take a photo (5 photos required by default)
5.  Once all photos are captured, click **💾 Save Enrollment**
6.  The LBPH model is retrained automatically with the new student

### 2. Take Attendance

1.  Click **✅ Take Attendance** in the header
2.  The camera starts and begins recognizing enrolled faces in real-time
3.  **Recognized students** are shown with a green bounding box and their name + confidence score
4.  **Unknown faces** are shown with a red bounding box
5.  Attendance is **automatically marked** when a student is recognized (each student is recorded only once per session)
6.  Click **⏹️ Stop & Save Attendance** to end the session

### 3. Manage Students

- The **side panel** shows all enrolled students when no mode is active
- Select a student and click **🗑️ Delete Selected** to remove them and their face data

### 4. View Attendance Records

- Attendance files are saved in the `attendance_records/` directory
- Files are named `attendance_YYYY-MM-DD.xlsx`
- Each Excel file contains styled columns: S.No, Roll No, Student Name, Date, Time, Status

---

## ⚙️ Configuration

All settings are in [`config.py`](config.py):

| Setting                     | Default  | Description                                             |
| --------------------------- | -------- | ------------------------------------------------------- |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5`    | Minimum confidence for YOLO face detection (0.0–1.0)    |
| `LBPH_THRESHOLD`            | `80`     | Recognition strictness (lower = stricter, range 50–100) |
| `NUM_ENROLLMENT_PHOTOS`     | `5`      | Photos captured per student during enrollment           |
| `CAMERA_INDEX`              | `0`      | Webcam index (change if you have multiple cameras)      |
| `WINDOW_WIDTH` / `HEIGHT`   | 1200×750 | Application window dimensions                           |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌────────────────┐     ┌──────────────────┐
│   main.py   │────▶│ face_detector  │────▶│  YOLOv8-face     │
│  (Tkinter   │     │  .py           │     │  Detection Model │
│   GUI)      │     └────────────────┘     └──────────────────┘
│             │
│             │     ┌────────────────┐     ┌──────────────────┐
│             │────▶│ face_encoder   │────▶│  LBPH Training   │
│             │     │  .py           │     │  (enrollment)    │
│             │     └────────────────┘     └──────────────────┘
│             │
│             │     ┌────────────────┐     ┌──────────────────┐
│             │────▶│face_recognizer │────▶│  LBPH Predict    │
│             │     │  .py           │     │  (recognition)   │
│             │     └────────────────┘     └──────────────────┘
│             │
│             │     ┌────────────────┐     ┌──────────────────┐
│             │────▶│ attendance_    │────▶│  Excel (.xlsx)   │
│             │     │ logger.py      │     │  via openpyxl    │
└─────────────┘     └────────────────┘     └──────────────────┘
```

**Pipeline:**

1. **Detection** → YOLOv8-face locates faces in each frame
2. **Recognition** → LBPH compares detected faces against trained model
3. **Logging** → Recognized students are auto-logged to a styled Excel sheet

---

## 📊 Excel Output Format

Each attendance file is a styled spreadsheet:

| S.No | Roll No | Student Name | Date       | Time     | Status  |
| ---- | ------- | ------------ | ---------- | -------- | ------- |
| 1    | 101     | John Doe     | 2026-03-08 | 09:15:23 | Present |
| 2    | 102     | Jane Smith   | 2026-03-08 | 09:15:45 | Present |

- Header row is styled with dark background and white text
- Data rows have a light green fill
- Columns are auto-sized for readability

---

## 🔧 Troubleshooting

| Issue                     | Solution                                                         |
| ------------------------- | ---------------------------------------------------------------- |
| Camera not opening        | Check webcam connection, or change `CAMERA_INDEX` in `config.py` |
| Poor recognition accuracy | Capture more enrollment photos in varied lighting/angles         |
| Too many false positives  | Lower `LBPH_THRESHOLD` (e.g., 60) in `config.py`                 |
| Too many false negatives  | Raise `LBPH_THRESHOLD` (e.g., 100) in `config.py`                |
| Model download fails      | Manually download `yolov8n-face.pt` and place it in `models/`    |

---

## 📝 License

This project is for educational purposes.
