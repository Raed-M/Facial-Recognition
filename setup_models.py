"""One-time setup: download UltraFace detector + MobileFaceNet embedder, make INT8 copy.

Run once:  python setup_models.py
"""
import os
import urllib.request
import zipfile

os.makedirs("models", exist_ok=True)

# 1. Face detector: UltraFace RFB-320 (~1.2 MB ONNX)
DETECTOR_URL = ("https://github.com/onnx/models/raw/main/validated/vision/"
                "body_analysis/ultraface/models/version-RFB-320.onnx")
if not os.path.exists("models/version-RFB-320.onnx"):
    print("Downloading UltraFace detector...")
    urllib.request.urlretrieve(DETECTOR_URL, "models/version-RFB-320.onnx")

# 2. MobileFaceNet: pre-trained with ArcFace loss on WebFace600K (from InsightFace)
MBFNET_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
if not os.path.exists("models/mobilefacenet.onnx"):
    print("Downloading MobileFaceNet...")
    urllib.request.urlretrieve(MBFNET_URL, "models/buffalo_sc.zip")
    with zipfile.ZipFile("models/buffalo_sc.zip") as z:
        for name in z.namelist():
            if name.endswith("w600k_mbf.onnx"):
                with open("models/mobilefacenet.onnx", "wb") as f:
                    f.write(z.read(name))
                break
    os.remove("models/buffalo_sc.zip")

# 3. INT8 quantized copy (ARM Cortex-M runs INT8 natively via CMSIS-NN)
from onnxruntime.quantization import quantize_dynamic, QuantType

if not os.path.exists("models/mobilefacenet_int8.onnx"):
    print("Creating INT8 model...")
    quantize_dynamic("models/mobilefacenet.onnx", "models/mobilefacenet_int8.onnx",
                     weight_type=QuantType.QInt8)

for name in ["version-RFB-320.onnx", "mobilefacenet.onnx", "mobilefacenet_int8.onnx"]:
    path = f"models/{name}"
    if os.path.exists(path):
        print(f"  {name}: {os.path.getsize(path) / 1e6:.1f} MB")
print("Done.")