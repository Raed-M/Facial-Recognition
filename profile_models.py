"""Profile model size, latency per frame, and FPS — pre- and post-quantization.

Run:   python profile_models.py
"""
import os
import time

import numpy as np

import face_lib as fl

N_RUNS = 100
FRAME = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # synthetic frame
FACE = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)   # synthetic face crop


def model_size(path):
    """Sum the .onnx file and its .onnx.data companion if one exists."""
    size = os.path.getsize(path)
    data_path = path + ".data"
    if os.path.exists(data_path):
        size += os.path.getsize(data_path)
    return size


def avg_ms(fn):
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        fn()
    return (time.perf_counter() - t0) / N_RUNS * 1000


detector = fl.load_session(fl.DETECTOR_PATH)
detect_ms = avg_ms(lambda: fl.detect_faces(detector, FRAME))

print(f"\nDetector (UltraFace): {model_size(fl.DETECTOR_PATH) / 1e6:.1f} MB, "
      f"{detect_ms:.1f} ms/frame\n")

print("| Embedder      | Model Size | Embed ms | Total ms/frame | FPS  |")
print("|---------------|------------|----------|----------------|------|")
for label, path in [("MBFNet FP32", "models/mobilefacenet.onnx"),
                    ("MBFNet INT8", "models/mobilefacenet_int8.onnx")]:
    if not os.path.exists(path):
        continue
    sess = fl.load_session(path)
    embed_ms = avg_ms(lambda: fl.embed(sess, FACE))
    total = detect_ms + embed_ms
    print(f"| {label:13} | {model_size(path) / 1e6:7.1f} MB | {embed_ms:8.1f} "
          f"| {total:14.1f} | {1000 / total:4.1f} |")
