# Few-Shot Edge Facial Recognition
The objective of this project is to develop a secure, low-data facial recognition system optimized for resource-constrained edge devices.

## Models Used:

### Face dectection: [UltraFace RFB-320](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface)
 - Pre-trained Facial Detection model.
 - Engineered specifically for edge computing devices: highly optimized and ultra-lightweight.
 - Already quantized using ONNX.

### Embedding model: [InsightFace MobileFaceNet](https://huggingface.co/WePrompt/buffalo_sc/blob/main/w600k_mbf.onnx)
 - Trained using ArcFace
 - Quantized into INT8 using ONNX.

### Liveness/Anti Spoofing: [MiniFASNet V2](https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx)


  
## Initial Setup:
1. Install the required packages using `pip install -r requirements.txt`. This will install all the necessary libraries.
2. Download the models by running the `setup_models.py` script.


## Usage:
1. **Run `server.py` on the Raspberry Pi or a laptop:**  
    - This device's camera will be used to capture the visual input.
    - The device will process the visual input, identify faces, and log them.
    - The device will host a Management web server at `http://<Device IP>:8000/`  
      
2. **Connect to the web server at `http://<Device IP>:8000/` from the same or from a different device to access the management console. Through this console, an admin can:**  
    - View the capture device's camera.
    - View the log.
    - Switch to enroll mode to enroll new users.
    - View and delete previously enrolled users.

## Modes:

### Identification Mode:
 - The default mode when the system is activated.
 - Detects faces, ensures they are not spoofs, and identifies faces by checking against the embeddings database.
 - If a previously enrolled face is detected, the detection signal is logged. This signal can be configured for other purposes.

### Enrollment Mode:
 - Used to enroll a new identity by adding a new face to the embeddings database.
 - It will enroll the most prominent face in the frame once this mode is activated.
 - Captures a new face at multiple angles to ensure it can be identified at various angles.
