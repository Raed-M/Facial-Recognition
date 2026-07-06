"""All tunable settings in one place."""

# Camera / server
CAMERA_INDEX = 0
HOST = "0.0.0.0"        # reachable on the LAN; never port-forward to the internet
PORT = 8000
TOKEN = "change-me"     # admin token for the control endpoints

# Model files (created by setup_models.py)
DETECTOR_PATH = "models/version-RFB-320.onnx"
LIVENESS_PATH = "models/minifasnet_v2.onnx"
EMBEDDER_PATH = "models/mobilefacenet_int8.onnx"   # or mobilefacenet.onnx (FP32)

# Data files
DB_PATH = "face_db.npy"
LOG_PATH = "log.jsonl"

# Recognition
MATCH_THRESHOLD = 1.0     # L2 distance; lower = stricter (tune it, see README)
LIVENESS_THRESHOLD = 0.5  # required probability that the face is real
LOG_COOLDOWN = 5          # seconds between repeated logs of the same person

# Enrollment
POSES_REQUIRED = 5
SAMPLES_PER_POSE = 5
MIN_POSE_DIST = 0.3       # L2 distance that separates two distinct poses
ENROLL_TIMEOUT = 30       # seconds before an enrollment attempt is aborted
