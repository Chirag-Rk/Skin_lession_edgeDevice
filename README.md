# Skin Lesion Classification — Edge Inference & Federated Learning on Jetson Nano

## Overview

This project implements a **privacy-preserving skin cancer diagnostic system** powered by **Federated Learning** and deployed on the **NVIDIA Jetson Nano (4GB)**. A trained MobileNetV3 deep learning model classifies dermoscopic images into 7 skin lesion types in real-time, with **Grad-CAM explainability overlays** rendered directly on a 7-inch touchscreen display.

Each Jetson Nano device acts as an independent **Flower FL client** — it trains locally on its own patient data and only transmits encrypted model weight updates to a central aggregation server, ensuring **full patient data privacy** across clinics.

Built with **PyTorch**, **Flower (flwr)**, **Flask**, **OpenCV**, and **Jetson.GPIO**.

---

## Hardware Requirements

| Component | Specification |
|---|---|
| **SBC** | NVIDIA Jetson Nano 4GB (B01 Developer Kit) |
| **Display** | 7-inch Touchscreen (HDMI + USB touch) |
| **Camera** | USB Dermoscope (or Raspberry Pi Camera V2 via CSI) |
| **Illumination** | LED Ring Light (5V, controlled via GPIO) |
| **Power** | 5V 4A DC Barrel Jack Power Supply |
| **Storage** | 32GB or 64GB MicroSD Card (UHS-I Class 10) |
| **Peripherals** | USB Keyboard and Mouse (for initial setup) |

> ⚠️ **Important**: You must **short the J48 Jumper** on the Jetson Nano carrier board to enable barrel jack power. See [Jetson_Setup.md](Jetson_Setup.md) for wiring details and the LED ring light transistor protection circuit.

---

## Project Structure

```
Skin_lession_edgeDevice/
│
├── data/                              # ← Dataset (not in git)
│   └── HAM10000/
│       ├── HAM10000_metadata.csv
│       ├── HAM10000_images_part_1/
│       └── HAM10000_images_part_2/
│
├── src/                               # Core ML code
│   ├── config.py                      # Hyperparameters, CUDA/CPU device auto-detection
│   ├── dataset.py                     # SkinDataset + transforms
│   ├── model.py                       # MobileNetV3-Large + SE Attention block
│   ├── train.py                       # train_one_epoch
│   ├── validate.py                    # validate (acc, auc, f1)
│   └── utils.py                       # Mixup, compute_metrics
│
├── fl/                                # Federated Learning (Flower)
│   ├── simulation.py                  # FL simulation entry point
│   ├── client.py                      # SkinClient (NumPyClient)
│   ├── client_app.py                  # Standalone FL client launcher
│   └── server.py                      # FL aggregation server
│
├── inference/                         # Edge Inference & Touch Dashboard
│   ├── app.py                         # Flask web server (camera, GPIO, FL, metrics)
│   ├── predict.py                     # SkinClassifier inference engine
│   ├── grad_cam.py                    # Grad-CAM heatmap generator
│   ├── templates/
│   │   └── index.html                 # Touchscreen dashboard layout
│   └── static/
│       ├── style.css                  # Glassmorphism dark-mode UI
│       └── app.js                     # Frontend interaction logic
│
├── centralized_train.py               # Centralized training baseline
├── compare.py                         # FL vs Centralized comparison plots
├── plot_fl.py                         # FL metric visualization
├── gradcam.py                         # Batch Grad-CAM heatmap generation
│
├── run_jetson_gui.sh                  # One-click Jetson Nano dashboard launcher
├── run_all.py                         # Multi-process FL launcher
├── run_all_local.py                   # Automated local pipeline
│
├── Jetson_Setup.md                    # Full hardware & software setup guide
├── Quick_Deployment.md                # Copy-paste deployment cheat sheet
├── Documentation.md                   # Project report & analysis
├── requirements.txt                   # Python dependencies
├── checkpoints/                       # Saved model weights (not in git)
├── logs/                              # JSON metric logs (auto-created)
└── plots/                             # Output visualizations (auto-created)
```

---

## 16-Step Implementation Flow

This is the end-to-end diagnostic and federated learning workflow executed on the Jetson Nano:

### Setup Phase
| Step | Action |
|------|--------|
| 1 | Power on the Jetson Nano (5V 4A barrel jack, J48 jumper shorted) |
| 2 | Insert the MicroSD card containing Ubuntu and the project |
| 3 | Connect the USB dermoscope to a Jetson USB port |
| 4 | Connect the 7-inch touchscreen via HDMI + USB |
| 5 | Connect the Jetson Nano to Wi-Fi |
| 6 | Launch the Python application (`./run_jetson_gui.sh`) |

### Inference Phase
| Step | Action |
|------|--------|
| 7 | The dermoscope captures the patient's skin lesion image |
| 8 | OpenCV reads the image frame from the USB camera |
| 9 | The image is resized to 128×128 and preprocessed (ImageNet normalization) |
| 10 | MobileNetV3 + Attention predicts the lesion class on the Jetson GPU (CUDA) |
| 11 | Grad-CAM highlights the diagnostically important region |
| 12 | The prediction and Grad-CAM overlay are displayed on the touchscreen |

### Federated Learning Phase
| Step | Action |
|------|--------|
| 13 | The Flower client trains locally using the device's own patient data |
| 14 | Only the updated model weights are sent to the Flower server |
| 15 | The server aggregates updates from all clients (FedAvg) and returns the improved global model |
| 16 | The Jetson Nano receives the updated model and uses it for future predictions |

---

## Model Architecture

| Component | Detail |
|---|---|
| **Backbone** | MobileNetV3-Large (pretrained on ImageNet via `timm`) |
| **Attention** | Channel-wise Squeeze-and-Excitation (SE) block |
| **Classifier** | FC(1280 → 512 → 7) with Dropout(0.3) |
| **Classes** | 7 skin lesion types from HAM10000 |
| **Device** | Auto-detects CUDA (Jetson GPU) or falls back to CPU |

---

## Dataset Setup

Download [HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) and place it as:

```
data/HAM10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/   (*.jpg)
└── HAM10000_images_part_2/   (*.jpg)
```

---

## Quick Start (Jetson Nano)

### 1. Install CUDA-enabled PyTorch for ARM64
```bash
sudo apt-get update
sudo apt-get install -y libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev python3-pip
pip3 install --upgrade pip

# PyTorch 1.10 for JetPack 4.6 / Python 3.6
wget https://nvidia.box.com/shared/static/p57jw14tk1vbka5pwa9t7296jszo14cx.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Torchvision v0.11.1
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision && export BUILD_VERSION=0.11.1 && python3 setup.py install --user && cd ..
```

### 2. Install Project Dependencies
```bash
pip3 install -r requirements.txt
pip3 install flask psutil
```

### 3. Launch the Dashboard
```bash
chmod +x run_jetson_gui.sh
./run_jetson_gui.sh
```

This boots the Flask edge server and opens Chromium in fullscreen kiosk mode on the 7-inch display.

---

## Quick Start (Desktop / Windows)

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### FL Simulation
```bash
python -m fl.simulation
```

### Centralized Training
```bash
python centralized_train.py
```

### Comparison & Visualization
```bash
python compare.py
python plot_fl.py
python gradcam.py
```

---

## Federated Learning Configuration

| Setting | Value |
|---|---|
| Framework | Flower (flwr ≥ 1.5) |
| Aggregation | FedAvg |
| Clients | 3 |
| Rounds | 3 (default) |
| Local epochs | 1 per round |
| Split modes | IID, Non-IID, Extreme Non-IID |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| AUC (OvR) | Primary metric — multi-class ROC |
| Accuracy | Overall correct predictions |
| F1 (Macro) | Class-balanced F1 score |

---

## Edge Dashboard Features

The touchscreen web dashboard (`inference/app.py`) provides:

- **Live Camera Feed** — MJPEG stream from USB dermoscope with mock/demo fallback
- **One-Tap Diagnosis** — Capture, preprocess, classify, and render Grad-CAM in a single button press
- **LED Ring Light Control** — GPIO-driven flash sequence synchronized with image capture
- **Hardware Telemetry** — Real-time CPU, GPU, memory, and temperature monitoring
- **Federated Learning Terminal** — Start/stop FL client participation and stream training logs live
- **Configurable Settings** — Camera mode (USB/CSI/Mock), GPIO pin, and video port selection

---

## Documentation

| File | Description |
|---|---|
| [Jetson_Setup.md](Jetson_Setup.md) | Full hardware wiring guide, J48 jumper, transistor circuit, and software provisioning |
| [Quick_Deployment.md](Quick_Deployment.md) | Copy-paste deployment cheat sheet for the Jetson Nano |
| [Documentation.md](Documentation.md) | Project report and experimental analysis |

---

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Flower (flwr)](https://flower.ai/)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing)
- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
