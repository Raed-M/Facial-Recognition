"""Core library: face detection (UltraFace), embedding (MobileFaceNet), and enrollment DB."""
import os

import cv2
import numpy as np
import onnxruntime as ort

DETECTOR_PATH = "models/version-RFB-320.onnx"
EMBEDDER_PATH = "models/mobilefacenet.onnx"  # swap for models/mobilefacenet_int8.onnx after setup
DB_PATH = "face_db.npy"
THRESHOLD = 0.6  # L2 distance on normalized embeddings; tune with real data (see README)


def load_session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


# ---------------- Face detection (UltraFace, 320x240 input) ----------------

def detect_faces(session, frame, score_thresh=0.7, iou_thresh=0.3):
    """Return a list of (x1, y1, x2, y2) pixel boxes for faces in a BGR frame."""
    h, w = frame.shape[:2]
    img = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB)
    img = ((img.astype(np.float32) - 127.0) / 128.0).transpose(2, 0, 1)[None]

    scores, boxes = session.run(None, {session.get_inputs()[0].name: img})
    scores, boxes = scores[0, :, 1], boxes[0]  # scores: (N,), boxes: (N, 4) normalized

    keep = scores > score_thresh
    scores, boxes = scores[keep], boxes[keep]
    boxes = boxes[_nms(boxes, scores, iou_thresh)]

    boxes = (boxes * [w, h, w, h]).astype(int)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return [tuple(b) for b in boxes]


def _nms(boxes, scores, iou_thresh):
    """Plain non-maximum suppression. Returns indices of boxes to keep."""
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area(boxes[i:i + 1]) + area(boxes[rest]) - inter)
        order = rest[iou < iou_thresh]
    return keep


# ---------------- Embedding (MobileFaceNet, 112x112 input) ----------------

def embed(session, face_bgr):
    """Map a BGR face crop to a normalized 512-d embedding."""
    face = cv2.cvtColor(cv2.resize(face_bgr, (112, 112)), cv2.COLOR_BGR2RGB)
    x = ((face.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
    e = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    return e / np.linalg.norm(e)


# ---------------- Enrollment database ----------------

def load_db():
    if os.path.exists(DB_PATH):
        return np.load(DB_PATH, allow_pickle=True).item()
    return {}  # name -> normalized mean embedding


def enroll(db, name, embeddings):
    """Average 5-10 sample embeddings and store the user. No retraining involved."""
    mean = np.stack(embeddings).mean(axis=0)
    db[name] = mean / np.linalg.norm(mean)
    np.save(DB_PATH, db)


def identify(db, emb):
    """Nearest enrolled user, or 'Unknown' if the distance crosses THRESHOLD."""
    name, best = "Unknown", float("inf")
    for n, ref in db.items():
        d = np.linalg.norm(emb - ref)
        if d < best:
            name, best = n, d
    return (name if best < THRESHOLD else "Unknown"), best