"""One-time setup: download the three models and create the INT8 embedder.

Run once:  python setup_models.py
"""
import os
import shutil
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
}

for path, url in DOWNLOADS.items():
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        if os.path.getsize(path) < 500_000:   # a Git-LFS pointer, not the model
            os.remove(path)
            raise RuntimeError(f"Download of {path} failed - try again")

# Liveness: MiniFASNet V2 (~1.7 MB). Fetched via huggingface_hub: the repo is
# tiny, so we grab it whole and take whatever .onnx file it contains.
if not os.path.exists("models/minifasnet_v2.onnx"):
    print("Downloading MiniFASNet V2 ...")
    from huggingface_hub import snapshot_download
    repo = snapshot_download("garciafido/minifasnet-v2-anti-spoofing-onnx")
    onnx_file = next(f for f in os.listdir(repo) if f.endswith(".onnx"))
    shutil.copy(os.path.join(repo, onnx_file), "models/minifasnet_v2.onnx")

# INT8 embedder: weight-quantized copy for the Pi's ARM CPU
if not os.path.exists("models/mobilefacenet_int8.onnx"):
    print("Quantizing embedder to INT8 ...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic("models/mobilefacenet.onnx", "models/mobilefacenet_int8.onnx",
                     weight_type=QuantType.QInt8)

for f in sorted(os.listdir("models")):
    print(f"  {f}: {os.path.getsize(os.path.join('models', f)) / 1e6:.1f} MB")
print("Done.")
