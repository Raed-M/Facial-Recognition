"""One-time setup: download the UltraFace detector, export FaceNet to ONNX, make an FP16 copy.

Run once:  python setup_models.py
"""
import os
import urllib.request

os.makedirs("models", exist_ok=True)

# 1. Face detector: UltraFace RFB-320 (already an ONNX file, ~1.2 MB)
DETECTOR_URL = ("https://github.com/onnx/models/raw/main/validated/vision/"
                "body_analysis/ultraface/models/version-RFB-320.onnx")
detector_path = "models/version-RFB-320.onnx"
if not os.path.exists(detector_path):
    print("Downloading UltraFace detector...")
    urllib.request.urlretrieve(DETECTOR_URL, detector_path)
    if os.path.getsize(detector_path) < 1_000_000:
        os.remove(detector_path)
        raise RuntimeError("Download failed (got a Git-LFS pointer instead of the model).")

# 2. Embedder: export the pre-trained FaceNet (trained with triplet loss) to ONNX
import torch
from facenet_pytorch import InceptionResnetV1

if not os.path.exists("models/facenet.onnx"):
    print("Exporting FaceNet to ONNX (downloads pretrained weights on first run)...")
    model = InceptionResnetV1(pretrained="vggface2").eval()
    torch.onnx.export(model, torch.randn(1, 3, 160, 160), "models/facenet.onnx",
                      input_names=["input"], output_names=["embedding"], opset_version=17)

# 3. FP16 quantized copy (keep_io_types=True so the same inference code works for both)
import onnx
from onnxconverter_common import float16

if not os.path.exists("models/facenet_fp16.onnx"):
    print("Creating FP16 model...")
    m = onnx.load("models/facenet.onnx")
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=True),
              "models/facenet_fp16.onnx")

print("Done. Models are in ./models/")
