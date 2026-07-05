# Few-Shot Edge Facial Recognition
The objective of this project is to develop a secure, low-data facial recognition system optimized for resource-constrained edge devices.

## Architecture:

 ### Face dectection: UltraFace RFB-320
   - Pre-trained Facial Detection model.
   - Engineered specifically for edge computing devices: highly optimized and ultra-lightweight.
   - Already quantized using ONNX.

 ### Embedding model: InceptionResnetV1
   - Pretrained on vggface2 using triplet loss.
   - Quantized into Float16 using ONNX.
  
  
## Initial Setup:
1. Install the required packages using `pip install -r requirements.txt`. This will install Pytorch, ONNX, and OpenCV.
2. Download the models by running the `setup_models.py` script.


## Usage:
This project utilizes the device's default camera.
As of right now, this project receives user input from the terminal:
  - Enter enroll mode by pressing `e`.
    - The user will need to enter a name to associate with the new identity.   
    - The system will Capture the face currently in frame and embed its features
    - The ebeddings will be stored in a file named `face_db.npy`
    - Afterwards, it will exit enrollment mode.
  - pressing `q` or `esc` will end the program.
