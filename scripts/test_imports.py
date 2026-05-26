import importlib
import os
import sys

# Ensure project root is on sys.path when running from scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

modules = [
    "config",
    "attendance_logger",
    "face_encoder",
    "face_detector",
    "face_recognizer",
    "attendance_popup",
    "main",
]

for m in modules:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception as e:
        print(m, "ERROR:", repr(e))

# Quick API checks
from face_encoder import get_enrolled_students, load_all_encodings
print('get_enrolled_students ->', get_enrolled_students())
print('load_all_encodings ->', load_all_encodings())
