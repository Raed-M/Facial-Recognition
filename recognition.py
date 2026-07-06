"""Camera loop: owns the camera and runs Identification / Enrollment modes."""
import json
import time
from datetime import datetime

import cv2
import numpy as np

import config
import face_lib as fl

GREEN, RED, ORANGE = (0, 255, 0), (0, 0, 255), (0, 165, 255)


class State:
    """Shared between this loop and the web server (single writer per field)."""

    def __init__(self):
        self.mode = "identify"     # "identify" or "enroll" (server sets "enroll")
        self.enroll_name = ""
        self.frame_jpeg = None     # latest annotated frame, JPEG bytes
        self.db = fl.load_db()


def log_signal(name):
    """The identity-confirmed signal. There is no consumer yet, so log it."""
    entry = {"name": name, "timestamp": datetime.now().isoformat(timespec="seconds")}
    with open(config.LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("SIGNAL:", entry)


def draw(frame, box, label, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def banner(frame, text, color=ORANGE):
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def identify_step(state, frame, boxes, models, last_logged):
    liveness, embedder = models
    for box in boxes:
        if fl.liveness_score(liveness, frame, box) < config.LIVENESS_THRESHOLD:
            draw(frame, box, "Spoof", RED)
            continue
        crop = fl.tight_crop(frame, box)
        if crop is None:
            continue
        name, dist = fl.identify(state.db, fl.embed(embedder, crop))
        draw(frame, box, f"{name} ({dist:.2f})", GREEN if name != "Unknown" else RED)
        if name != "Unknown" and time.time() - last_logged.get(name, 0) > config.LOG_COOLDOWN:
            last_logged[name] = time.time()
            log_signal(name)


def enroll_step(state, enroll, frame, boxes, models):
    liveness, embedder = models
    if time.time() > enroll["deadline"]:
        print(f"Enrollment of '{state.enroll_name}' timed out.")
        state.mode = "identify"
        return
    if len(boxes) > 1:
        banner(frame, "Warning: more than one person in frame", RED)
    if not boxes:
        banner(frame, f"Enrolling {state.enroll_name}: no face detected")
        return

    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))  # largest face
    if fl.liveness_score(liveness, frame, box) < config.LIVENESS_THRESHOLD:
        draw(frame, box, "Spoof rejected", RED)
        return
    crop = fl.tight_crop(frame, box)
    if crop is None:
        return
    e = fl.embed(embedder, crop)

    if any(np.linalg.norm(e - p) < config.MIN_POSE_DIST for p in enroll["poses"]):
        enroll["samples"] = []          # too close to a captured pose
        msg = "move to a new pose"
    elif enroll["samples"] and np.linalg.norm(e - enroll["samples"][0]) > config.MIN_POSE_DIST:
        enroll["samples"] = [e]         # moved mid-pose: restart sampling here
        msg = f"hold still (1/{config.SAMPLES_PER_POSE})"
    else:
        enroll["samples"].append(e)
        msg = f"hold still ({len(enroll['samples'])}/{config.SAMPLES_PER_POSE})"
        if len(enroll["samples"]) == config.SAMPLES_PER_POSE:
            pose = np.stack(enroll["samples"]).mean(axis=0)
            enroll["poses"].append(pose / np.linalg.norm(pose))
            enroll["samples"] = []
            msg = "pose captured - now move"

    draw(frame, box, f"Pose {len(enroll['poses'])}/{config.POSES_REQUIRED}: {msg}", ORANGE)

    if len(enroll["poses"]) == config.POSES_REQUIRED:
        fl.save_user(state.db, state.enroll_name, enroll["poses"])
        print(f"Enrolled '{state.enroll_name}'.")
        state.mode = "identify"


def run(state):
    detector = fl.load_session(config.DETECTOR_PATH)
    models = (fl.load_session(config.LIVENESS_PATH),
              fl.load_session(config.EMBEDDER_PATH))
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    last_logged = {}    # name -> time of last log (cooldown)
    enroll = None       # active enrollment progress, or None

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.5)
            continue

        boxes = fl.detect_faces(detector, frame)

        if state.mode == "enroll":
            if enroll is None:
                enroll = {"poses": [], "samples": [],
                          "deadline": time.time() + config.ENROLL_TIMEOUT}
            enroll_step(state, enroll, frame, boxes, models)
        else:
            enroll = None
            identify_step(state, frame, boxes, models, last_logged)

        state.frame_jpeg = cv2.imencode(".jpg", frame)[1].tobytes()
