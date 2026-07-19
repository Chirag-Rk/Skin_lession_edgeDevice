# Federated Learning for Skin Lesion Classification

## Overview

This project implements a **Federated Learning (FL)** system for skin lesion classification using deep learning. Multiple simulated clients (hospitals/devices) train locally and share only model updates with a central server — preserving data privacy.

Built with **PyTorch** and **Flower (flwr)**.

---

## Project Structure

```
skin_cancer_detection/
│
├── data/                          # ← place dataset here (not in git)
│   └── HAM10000/
│       ├── HAM10000_metadata.csv
│       ├── HAM10000_images_part_1/
│       └── HAM10000_images_part_2/
│
├── src/                           # Core ML code
│   ├── config.py                  # Hyperparameters, paths, label map
│   ├── dataset.py                 # SkinDataset + transforms
│   ├── model.py                   # MobileNetV3 + Attention
│   ├── train.py                   # train_one_epoch
│   ├── validate.py                # validate (acc, auc, f1)
│   └── utils.py                   # Mixup, compute_metrics
│
├── fl/                            # Federated Learning
│   ├── __init__.py
│   ├── simulation.py              # ← main entry point (Step 1–3)
│   ├── client.py                  # SkinClient (NumPyClient)
│   └── server.py                  # Multi-terminal server
│
├── centralized_train.py           # Step 4: Centralized baseline
├── compare.py                     # Step 5: Centralized vs FL comparison
├── plot_fl.py                     # Step 8: FL metric visualization
├── gradcam.py                     # Step 9: Grad-CAM heatmaps
│
├── checkpoints/                   # Saved models (not in git)
├── logs/                          # JSON metric logs (auto-created)
├── plots/                         # Output plots (auto-created)
│
├── requirements.txt
└── README.md
```

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

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

---

## Execution Order

### Step 1–3: FL Simulation (3 clients, 3 rounds, logging)

```bash
python -m fl.simulation
```

Logs saved to `logs/fl_metrics.json`.

#### Change split mode via environment variable:

```bash
# IID split
set SPLIT_MODE=iid && python -m fl.simulation

# Non-IID split (default)
set SPLIT_MODE=non_iid && python -m fl.simulation

# Extreme Non-IID
set SPLIT_MODE=extreme && python -m fl.simulation
```

---

### Step 4: Centralized Training (baseline)

```bash
python centralized_train.py
```

Optional: override epochs

```bash
set EPOCHS=10 && python centralized_train.py
```

Logs saved to `logs/centralized_metrics.json`.

---

### Step 5: Compare Centralized vs FL

```bash
python compare.py
```

Output: `plots/comparison.png`

---

### Step 8: Visualize FL Metrics

```bash
python plot_fl.py
```

Output: `plots/fl_metrics.png`

---

### Step 9: Grad-CAM Visualization

```bash
python gradcam.py
# or with custom checkpoint:
python gradcam.py --checkpoint checkpoints/centralized_best.pt --n 7
```

Output: `plots/gradcam/gradcam_grid.png`

---

## Model Architecture

- **Backbone**: MobileNetV3-Large (pretrained on ImageNet via `timm`)
- **Attention**: Channel-wise SE-style attention block
- **Head**: FC(1280→512→7) with Dropout(0.3)
- **Classes**: 7 skin lesion types (HAM10000)

---

## Federated Learning Details

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

## Experiments

| Experiment | Command |
|---|---|
| IID FL | `set SPLIT_MODE=iid && python -m fl.simulation` |
| Non-IID FL | `set SPLIT_MODE=non_iid && python -m fl.simulation` |
| Extreme Non-IID | `set SPLIT_MODE=extreme && python -m fl.simulation` |
| Centralized | `python centralized_train.py` |
| Compare | `python compare.py` |
| Grad-CAM | `python gradcam.py` |

---

## Team Responsibilities

### Model & ML
- Model design (MobileNetV3 + Attention)
- Training (Mixup, LR scheduling, gradient clipping)
- Metrics (AUC, F1, Accuracy)
- Grad-CAM (`gradcam.py`)

### FL & System
- FL simulation setup (`fl/simulation.py`)
- Client logic (`fl/client.py`)
- Data splitting (IID/Non-IID/Extreme)
- Logging & visualization (`plot_fl.py`, `compare.py`)

---

## Notes

- Dataset and checkpoints are **excluded from git** (see `.gitignore`)
- All paths are relative — run scripts from the project root
- CPU-optimized by default (`image_size=128`, `batch_size=8`)

---

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Flower (flwr)](https://flower.ai/)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
