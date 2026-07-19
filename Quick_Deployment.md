# Quick Deployment Cheat Sheet: Jetson Nano (4GB)

This document provides a condensed, copy-paste-friendly roadmap to wire, set up, and launch the DermOS diagnostic application on the Jetson Nano.

---

## 1. Hardware Pinout & Wiring Connections

### 5V 4A DC Power Jumper Setup
*   Locate the **J48 Jumper pins** (near the DC barrel jack).
*   **Place a jumper block** on J48 to disable micro-USB power and enable the DC barrel jack.
*   Connect the **5V 4A DC Power Supply** to the barrel jack.

### LED Ring Light Circuit (transistor switch)
Connect the LED Ring Light using a simple transistor driver (e.g., 2N2222) to protect the GPIO pins:
*   **Jetson Pin 2 (5V)** ➔ **LED Ring (+)**
*   **Jetson Pin 12 (GPIO 18 / Board Pin 12)** ➔ **1kΩ Resistor** ➔ **Transistor Base (B)**
*   **LED Ring (-)** ➔ **Transistor Collector (C)**
*   **Jetson Pin 6 (GND)** ➔ **Transistor Emitter (E)**

---

## 2. OS Setup & Provisioning

1.  Flash the JetPack SD Card Image (Ubuntu 18.04 LTS containing Python 3.6) using **BalenaEtcher**.
2.  Boot the Jetson Nano, connect to Wi-Fi, and open a terminal.

### Configure Non-Root GPIO Permissions
```bash
sudo groupadd -f -r gpio
sudo usermod -a -G gpio $USER
sudo cp /opt/nvidia/jetson-gpio/etc/99-gpio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 3. CUDA PyTorch Setup (For Maxwell GPU Acceleration)

Compile and install NVIDIA's official PyTorch wheels for ARM64:

```bash
# 1. System dependencies
sudo apt-get update
sudo apt-get install -y libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev python3-pip
pip3 install --upgrade pip

# 2. Download and install PyTorch 1.10
wget https://nvidia.box.com/shared/static/p57jw14tk1vbka5pwa9t7296jszo14cx.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# 3. Clone and compile Torchvision v0.11.1
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd ..
```

---

## 4. Install Project Requirements

Navigate to the project root directory and run:
```bash
pip3 install -r requirements.txt
pip3 install flask psutil
```

Ensure a trained model checkpoint is present inside the project at:
*   `checkpoints/centralized_best.pt` or `checkpoints/best_model.pt`

---

## 5. Execution

Make the launcher script executable and boot the dashboard:
```bash
chmod +x run_jetson_gui.sh
./run_jetson_gui.sh
```

This runs the backend server in the background and launches Chromium in fullscreen kiosk mode on your 7-inch display. Press **Ctrl+C** in the terminal to stop the server and close the kiosk window.
