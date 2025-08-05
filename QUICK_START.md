# 🚀 Quick Setup Guide

Get YOLOv7 GPU Mode running in minutes!

## ⚡ Super Quick Start (5 minutes)

### Windows Users

```bash
# 1. Clone the repository
git clone https://github.com/abdullahjave/YOLOv7-in-GPU-Mode.git
cd YOLOv7-in-GPU-Mode

# 2. Run the installation script
install.bat

# 3. Test your setup
python test_gpu.py

# 4. Run your first detection!
python detect.py --weights weights/yolov7.pt --source inference/images/horses.jpg
```

### Linux/Mac Users

```bash
# 1. Clone the repository
git clone https://github.com/abdullahjave/YOLOv7-in-GPU-Mode.git
cd YOLOv7-in-GPU-Mode

# 2. Run the installation script
chmod +x install.sh
./install.sh

# 3. Test your setup
python test_gpu.py

# 4. Run your first detection!
python detect.py --weights weights/yolov7.pt --source inference/images/horses.jpg
```

## 🔧 Manual Installation (10 minutes)

If the automatic scripts don't work for you:

### Step 1: Prerequisites
- Python 3.7+
- NVIDIA GPU with CUDA support
- 8GB+ RAM

### Step 2: Create Environment
```bash
# Create virtual environment
python -m venv yolov7-env

# Activate it
# Windows:
yolov7-env\Scripts\activate
# Linux/Mac:
source yolov7-env/bin/activate
```

### Step 3: Install PyTorch
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Dependencies
```bash
pip install -r requirements_gpu.txt
```

### Step 5: Download Weights
```bash
mkdir weights
wget -O weights/yolov7.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

## ✅ Verify Installation

Run this to check everything is working:

```python
import torch
print(f"🎮 CUDA Available: {torch.cuda.is_available()}")
print(f"🔢 GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"🏷 GPU Name: {torch.cuda.get_device_name()}")
```

## 🎯 First Detection

```bash
# Download a test image (optional)
mkdir -p inference/images
wget -O inference/images/test.jpg https://ultralytics.com/images/bus.jpg

# Run detection
python detect.py --weights weights/yolov7.pt --source inference/images/test.jpg

# Check results in runs/detect/exp/
```

## 🚨 Common Issues

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory
```bash
# Reduce batch size
python detect.py --weights yolov7.pt --source image.jpg --img-size 416

# Or use smaller image size
python detect.py --weights yolov7.pt --source image.jpg --img-size 320
```

### Missing Dependencies
```bash
# Install missing packages
pip install package_name

# Or reinstall all
pip install -r requirements_gpu.txt --force-reinstall
```

## 🎉 You're Ready!

Congratulations! YOLOv7 GPU Mode is now set up and ready to use.

### What's Next?

1. **📖 Read the [full README](README.md)** for detailed documentation
2. **🎬 Try video detection**: `python detect.py --weights weights/yolov7.pt --source your_video.mp4`
3. **📷 Use your webcam**: `python detect.py --weights weights/yolov7.pt --source 0`
4. **🏋️ Train on custom data**: See the training section in the README
5. **📤 Export your model**: Try ONNX or TensorRT export

### Need Help?

- 📖 Check the [README](README.md) for detailed guides
- 🐛 Report issues on GitHub
- 💬 Join the community discussions

Happy detecting! 🎯
