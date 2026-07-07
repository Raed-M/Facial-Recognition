"""Model helpers (detection, liveness, embedding) and the enrollment database."""
import os

import cv2
import numpy as np
import onnxruntime as ort

import config


def load_session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _run(session, x):
    return session.run(None, {session.get_inputs()[0].name: x})


# ---------------- Face detection (UltraFace, 320x240 input) ----------------

def detect_faces(session, frame, score_thresh=0.7, iou_thresh=0.3):
    """Return a list of (x1, y1, x2, y2) pixel boxes for faces in a BGR frame."""
    h, w = frame.shape[:2]
    img = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB)
    img = ((img.astype(np.float32) - 127.0) / 128.0).transpose(2, 0, 1)[None]

    scores, boxes = _run(session, img)
    scores, boxes = scores[0, :, 1], boxes[0]   # (N,), (N, 4) normalized coords

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


# ---------------- Liveness (MiniFASNet V2, 80x80 input) ----------------

def liveness_score(session, frame, box):
    """Probability that the face is real (not a photo or a screen).

    MiniFASNet expects a crop with a 2.7x margin around the face box; the extra
    background (paper edges, screen bezels, moire) is how it spots spoofs.
    Input stays BGR, scaled to [0, 1].
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = 2.7 * max(x2 - x1, y2 - y1) / 2
    h, w = frame.shape[:2]
    crop = frame[int(max(cy - half, 0)):int(min(cy + half, h)),
                 int(max(cx - half, 0)):int(min(cx + half, w))]
    if crop.size == 0:
        return 0.0
    x = (cv2.resize(crop, (80, 80)).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    probs = _run(session, x)[0][0]   # already softmax'd: [live, print-attack, replay-attack]
    return float(probs[0])   # index 0 = live


# ---------------- Embedding (MobileFaceNet, 112x112 input) ----------------

def embed(session, face_bgr):
    """Normalized 512-d embedding of a tight BGR face crop."""
    face = cv2.cvtColor(cv2.resize(face_bgr, (112, 112)), cv2.COLOR_BGR2RGB)
    x = ((face.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
    e = _run(session, x)[0][0]
    return e / np.linalg.norm(e)


def tight_crop(frame, box):
    x1, y1, x2, y2 = box
    c = frame[y1:y2, x1:x2]
    return c if c.size else None


# ---------------- Enrollment database ----------------
# {name: array of shape (POSES_REQUIRED, 512)} - several pose embeddings each.

def load_db():
    if os.path.exists(config.DB_PATH):
        return np.load(config.DB_PATH, allow_pickle=True).item()
    return {}


def save_db(db):
    np.save(config.DB_PATH, db)


def save_user(db, name, pose_embeddings):
    db[name] = np.stack(pose_embeddings)
    save_db(db)


def identify(db, emb):
    """Closest enrolled person, by minimum distance to any of their poses."""
    name, best = "Unknown", float("inf")
    for n, poses in db.items():
        d = float(np.linalg.norm(poses - emb, axis=1).min())
        if d < best:
            name, best = n, d
    return (name if best < config.MATCH_THRESHOLD else "Unknown"), best
