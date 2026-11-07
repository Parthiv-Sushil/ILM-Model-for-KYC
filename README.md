[Chinese Version](README_CN.md)|**English Version**  
![Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/logo.jpg)  

# Silent-Face-Anti-Spoofing & ILM KYC Verification Extension

This repository contains the open-source **Silent-Face-Anti-Spoofing** lightweight face liveness detection project by [Minivision Technology](https://www.minivision.cn/), extended with an **Integrated Learning Model (ILM)** framework for **Digital KYC Verification**.

---

## 1. Project Overview

Silent Face Anti-Spoofing distinguishes between genuine real faces and presentation attack (spoof) faces, such as printed paper photos, electronic screen displays, silicone masks, and 3D human portraits. 

This project incorporates:
1. **MiniFASNet Baseline**: Lightweight face anti-spoofing backbone pruned from MobileFaceNet (FLOPs reduced from 0.224G to 0.081G) with auxiliary 2D Fourier Spectrum supervision during training.
2. **Integrated Learning Model (ILM) Extension**: A multi-modal extension incorporating **Spatial**, **Temporal** (GRU/LSTM), **Liveness** (EAR/blink & motion variance), and **Frequency/Texture** (2D FFT spectral energy & Laplacian descriptors) feature branches with gated fusion.
3. **KYC Risk Scoring Engine**: Evaluates overall risk score (0-100), risk level (`LOW`, `MEDIUM`, `HIGH`), and action decision (`APPROVE`, `MANUAL REVIEW`, `REJECT`).

```
                    | Model | FLOPs | Params |
                    | :---: | :---: | :----: |
                    | MobileFaceNet | 0.224G | 0.991M |
                    | MiniFASNetV1 | 0.081G | 0.414M |
                    | MiniFASNetV2 | 0.081G | 0.435M |
```

---

## 2. Main Features

- **Multi-Modal Feature Fusion**: Combines spatial embeddings from MiniFASNet, temporal sequence modeling over video frames, physiological eye/motion dynamics, and frequency domain descriptors.
- **Configurable KYC Risk Engine**: Multi-factor scoring combining Anti-Spoofing probability, Liveness score, ID Face Match cosine similarity, and Document validity score.
- **Comparative Evaluation Suite**: Evaluates Baseline vs +Temporal vs +Liveness vs Full ILM Model across Accuracy, F1, ROC-AUC, FAR/FRR, APCER/BPCER/ACER, Latency, and saves confusion matrix plots.
- **Interactive Visual Demo**: Real-time video/webcam dashboard displaying face bounding boxes, component scores, risk score gauge, and decision badges.

---

## 3. Directory Structure

```
├── datasets/                 # Training dataset directory structure
├── images/                   # Sample images and demo screenshots
│   └── sample/               # Test sample images (image_F1.jpg, image_T1.jpg, etc.)
├── resources/                # Pretrained model weights
│   ├── anti_spoof_models/    # Pretrained MiniFASNet models (2.7_80x80_MiniFASNetV2.pth, etc.)
│   └── detection_model/      # RetinaFace Caffe detection model
├── src/                      # Source code
│   ├── data_io/              # Dataset loading & transform utilities
│   ├── model_lib/            # MiniFASNet & MultiFTNet model architectures
│   ├── ilm_model/            # ILM Spatial, Temporal, Liveness, Frequency & Fusion modules
│   └── kyc_risk/             # KYC Risk Scoring Engine & Face Matcher
├── test.py                   # Original baseline inference test script
├── train.py                  # Model training entry point
├── eval_ilm.py               # ILM evaluation & benchmarking suite
├── demo_kyc.py               # Interactive KYC verification demo
├── ARCHITECTURE.md           # Detailed system & research architecture documentation
└── requirements.txt          # Python dependencies
```

---

## 4. Installation & Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Original MiniFASNet Baseline Test
To run inference using the original pre-trained MiniFASNet baseline model:
```bash
python test.py --image_name image_F1.jpg
```
Output:
```
Image 'image_F1.jpg' is Fake Face. Score: 0.73.
Prediction cost 0.40 s
```

---

## 5. Running the ILM KYC Extension

### Run Evaluation Suite
To benchmark and compare Original MiniFASNet vs +Temporal vs +Liveness vs Full ILM Model:
```bash
python eval_ilm.py
```
This outputs a summary markdown metrics table, saves `evaluation_results.json`, and generates confusion matrix plots (`confusion_matrix_*.png`).

### Run Interactive KYC Verification Demo
- **Option 1: Live Webcam GUI Window (Pop-Up Camera Window):**
  ```bash
  python demo_kyc.py --webcam
  ```

- **Option 2: Web Browser Interactive Dashboard (Streamlit):**
  ```bash
  streamlit run app_kyc.py
  ```

- **Option 3: Live Video File Stream:**
  ```bash
  python demo_kyc.py --video_path ./path/to/video.mp4 --save_output ./demo_output.mp4
  ```

- **Option 4: With Reference ID Card Photo Matching:**
  ```bash
  python demo_kyc.py --webcam --id_image_path ./images/sample/image_T1.jpg
  ```

---

## 6. Preprocessing & Training

### Preprocessing Protocol
1. **Data Categories**: Divide training data into 3 classes (Real, Fake/Spoof, Background).
2. **Multi-Scale Patches**: Crop face bounding boxes using RetinaFace and expand box edges at scales `1.0`, `2.7`, and `4.0` resized to fixed `80x80` input dimensions.
3. **Fourier Spectrum Supervision**: Generate online 2D FFT magnitude spectrum target images using `generate_FT`.

### Train Model
```bash
python train.py --device_ids 0 --patch_info 2.7_80x80
```

---

## 7. Documentation & References

- **Technical Architecture & Formulas**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Face Detector Reference**: [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- **Original Repository**: [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

---

## 8. License
This repository is licensed under the Apache 2.0 License.
