"""Profile model sizes and per-stage latency, pre/post-quantization.

Run:   python profile_models.py
"""
import os
import time

import numpy as np

import config
import face_lib as fl

N = 100
FRAME = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
FACE = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
BOX = (280, 180, 360, 300)   # synthetic face box for the liveness crop


def avg_ms(fn):
    fn()   # warm-up
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    return (time.perf_counter() - t0) / N * 1000


def size_mb(path):
    return os.path.getsize(path) / 1e6


detector = fl.load_session(config.DETECTOR_PATH)
liveness = fl.load_session(config.LIVENESS_PATH)
det_ms = avg_ms(lambda: fl.detect_faces(detector, FRAME))
liv_ms = avg_ms(lambda: fl.liveness_score(liveness, FRAME, BOX))

print(f"\nDetector (UltraFace):  {size_mb(config.DETECTOR_PATH):5.1f} MB  {det_ms:6.1f} ms/frame")
print(f"Liveness (MiniFASNet): {size_mb(config.LIVENESS_PATH):5.1f} MB  {liv_ms:6.1f} ms/face\n")

print("| Embedder    | Size (MB) | Embed ms | Full pipeline ms | FPS  |")
print("|-------------|-----------|----------|------------------|------|")
for label, path in [("MBFNet FP32", "models/mobilefacenet.onnx"),
                    ("MBFNet INT8", "models/mobilefacenet_int8.onnx")]:
    if not os.path.exists(path):
        continue
    sess = fl.load_session(path)
    emb_ms = avg_ms(lambda: fl.embed(sess, FACE))
    total = det_ms + liv_ms + emb_ms
    print(f"| {label} | {size_mb(path):9.1f} | {emb_ms:8.1f} "
          f"| {total:16.1f} | {1000 / total:4.1f} |")
