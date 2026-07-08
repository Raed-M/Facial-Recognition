"""One-time setup: download the three models and create the INT8 embedder.

Run once:  python setup_models.py
"""
import os
import urllib.request

os.makedirs("models", exist_ok=True)

DOWNLOADS = {
    # Face detector: UltraFace RFB-320 (~1.2 MB)
    "models/version-RFB-320.onnx":
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "body_analysis/ultraface/models/version-RFB-320.onnx",
    # Face embedder: InsightFace MobileFaceNet, ArcFace-trained (~13.6 MB)
    "models/mobilefacenet.onnx":
        "https://huggingface.co/WePrompt/buffalo_sc/resolve/main/w600k_mbf.onnx",
    # Liveness: MiniFASNetV2-SE trained-and-exported to INT8 ONNX (~0.6 MB).
    # 2-class [real, spoof] logits; verified input-responsive.
    "models/liveness.onnx":
        "https://raw.githubusercontent.com/facenox/face-antispoof-onnx/"
        "main/models/best_model_quantized.onnx",
}

MIN_BYTES = 100_000   # anything smaller is a Git-LFS pointer / failed download

for path, url in DOWNLOADS.items():
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        if os.path.getsize(path) < MIN_BYTES:
            os.remove(path)
            raise RuntimeError(f"Download of {path} failed - try again")

# INT8 embedder: weight-quantized copy for the Pi's ARM CPU
if not os.path.exists("models/mobilefacenet_int8.onnx"):
    print("Quantizing embedder to INT8 ...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic("models/mobilefacenet.onnx", "models/mobilefacenet_int8.onnx",
                     weight_type=QuantType.QInt8)

for f in sorted(os.listdir("models")):
    print(f"  {f}: {os.path.getsize(os.path.join('models', f)) / 1e6:.1f} MB")
print("Done.")
