"""Profile model size, latency per frame, and FPS — pre- and post-quantization.

Run:   python profile_models.py
"""
import os
import time

import numpy as np

import face_lib as fl

N_RUNS = 100
FRAME = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # synthetic frame
FACE = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)   # synthetic face crop


def avg_ms(fn):
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        fn()
    return (time.perf_counter() - t0) / N_RUNS * 1000


detector = fl.load_session(fl.DETECTOR_PATH)
detect_ms = avg_ms(lambda: fl.detect_faces(detector, FRAME))

print(f"\nDetector (UltraFace): {os.path.getsize(fl.DETECTOR_PATH) / 1e6:.1f} MB, "
      f"{detect_ms:.1f} ms/frame\n")

print("| Embedder      | Model Size | Embed ms | Total ms/frame | FPS  |")
print("|---------------|------------|----------|----------------|------|")
for label, path in [("FaceNet FP32", "models/facenet.onnx"),
                    ("FaceNet FP16", "models/facenet_fp16.onnx")]:
    if not os.path.exists(path):
        continue
    sess = fl.load_session(path)
    embed_ms = avg_ms(lambda: fl.embed(sess, FACE))
    total = detect_ms + embed_ms
    print(f"| {label:13} | {os.path.getsize(path) / 1e6:7.1f} MB | {embed_ms:8.1f} "
          f"| {total:14.1f} | {1000 / total:4.1f} |")
