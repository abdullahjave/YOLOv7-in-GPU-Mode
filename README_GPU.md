# 🚀 YOLOv7 GPU Mode - High-Performance Object Detection

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-GPL%20v3-red.svg)](LICENSE.md)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)

<img src="../figure/performance.png" width="80%" alt="YOLOv7 Performance"/>

**🎯 State-of-the-art real-time object detection optimized for GPU acceleration**

[📖 Paper](https://arxiv.org/abs/2207.02696) • [🔧 Installation](#-installation) • [⚡ Quick Start](#-quick-start) • [🎮 Demo](#-web-demo) • [📊 Performance](#-performance)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🔧 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [🎮 Web Demo](#-web-demo)
- [📊 Performance](#-performance)
- [🚀 Usage](#-usage)
  - [Detection](#detection)
  - [Training](#training)
  - [Testing](#testing)
  - [Export](#export)
- [💡 Advanced Features](#-advanced-features)
- [🛠 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📜 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)

---

## ✨ Features

🔥 **What makes this YOLOv7 GPU implementation special:**

- ⚡ **GPU-Optimized**: Fully optimized for NVIDIA GPUs with CUDA acceleration
- 🎯 **State-of-the-Art Accuracy**: 51.4% AP on MS COCO at 161 FPS
- 🔧 **Easy Setup**: Streamlined installation and configuration process
- 📊 **Multiple Model Variants**: From tiny to extra-large models
- 🎨 **Versatile Applications**: Object detection, pose estimation, instance segmentation
- 📱 **Export Ready**: ONNX, TensorRT, CoreML export support
- 🐍 **Python-First**: Clean, readable, and maintainable codebase
- 📝 **Well Documented**: Comprehensive documentation and examples

---

## 🔧 Installation

### Prerequisites

Before you start, ensure you have:

- 🐍 **Python 3.7+**
- 🎮 **NVIDIA GPU** with CUDA support
- 💾 **8GB+ RAM** (16GB+ recommended)
- 💿 **10GB+ free disk space**

### Option 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/abdullahjave/YOLOv7-in-GPU-Mode.git
cd YOLOv7-in-GPU-Mode

# Create a virtual environment
python -m venv yolov7-env
source yolov7-env/bin/activate  # On Windows: yolov7-env\Scripts\activate

# Install GPU-optimized requirements
pip install -r requirements_gpu.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option 2: Conda Environment

```bash
# Create conda environment
conda create -n yolov7-gpu python=3.8
conda activate yolov7-gpu

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements_gpu.txt
```

### Option 3: Docker (For Advanced Users)

```bash
# Build and run Docker container
docker build -t yolov7-gpu .
docker run --gpus all -it -v $(pwd):/workspace yolov7-gpu
```

### ✅ Verify Installation

Run this to check your GPU setup:

```python
import torch
print(f"🎮 CUDA Available: {torch.cuda.is_available()}")
print(f"🔢 GPU Count: {torch.cuda.device_count()}")
print(f"🏷 GPU Name: {torch.cuda.get_device_name()}")
```

---

## ⚡ Quick Start

### 🎯 Object Detection on Images

```bash
# Download pre-trained weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# Run detection on a single image
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg

# Run detection on a folder of images
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source path/to/images/
```

### 🎬 Video Detection

```bash
# Detect objects in video
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source your_video.mp4

# Real-time webcam detection
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source 0
```

### 📸 Results

After running detection, check the `runs/detect/exp/` folder for your results!

<div align="center">
<img src="../figure/horses_prediction.jpg" width="60%" alt="Detection Example"/>
</div>

---

## 🎮 Web Demo

Try YOLOv7 instantly without installation:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb)

---

## 📊 Performance

### 🏆 MS COCO Benchmark Results

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | FPS | Params |
|-------|:---------:|:-----------------:|:------------------------------:|:------------------------------:|:---:|:------:|
| **YOLOv7** | 640 | **51.4%** | **69.7%** | **55.9%** | **161** | 37.6M |
| **YOLOv7-X** | 640 | **53.1%** | **71.2%** | **57.8%** | **114** | 71.3M |
| **YOLOv7-W6** | 1280 | **54.9%** | **72.6%** | **60.1%** | **84** | 70.4M |
| **YOLOv7-E6** | 1280 | **56.0%** | **73.5%** | **61.2%** | **56** | 97.2M |
| **YOLOv7-D6** | 1280 | **56.6%** | **74.0%** | **61.8%** | **44** | 154.7M |
| **YOLOv7-E6E** | 1280 | **56.8%** | **74.4%** | **62.1%** | **36** | 151.7M |

### 🎯 Model Comparison

```
               Speed vs Accuracy
                     ⬆ Accuracy
         YOLOv7-E6E ●
                    │
      YOLOv7-D6 ●  │
                │  │
    YOLOv7-E6 ●   │
              │   │
  YOLOv7-W6 ●     │
            │     │
   YOLOv7-X ●     │
            │     │
    YOLOv7 ●──────●────► Speed
           Fast   Slow
```

---

## 🚀 Usage

### 🎯 Detection

#### Basic Detection
```bash
# Single image
python detect.py --weights yolov7.pt --source image.jpg

# Multiple images
python detect.py --weights yolov7.pt --source path/to/images/

# Video file
python detect.py --weights yolov7.pt --source video.mp4

# Webcam
python detect.py --weights yolov7.pt --source 0
```

#### Advanced Detection Options
```bash
# Custom confidence threshold and NMS
python detect.py --weights yolov7.pt --source image.jpg --conf 0.4 --iou 0.5

# Save results with custom name
python detect.py --weights yolov7.pt --source image.jpg --name my_detection

# Hide labels or confidence scores
python detect.py --weights yolov7.pt --source image.jpg --hide-labels --hide-conf

# Save crops of detected objects
python detect.py --weights yolov7.pt --source image.jpg --save-crop
```

### 🏋️ Training

#### Data Preparation
```bash
# Download COCO dataset
bash scripts/get_coco.sh

# For custom dataset, organize like this:
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

#### Single GPU Training
```bash
# Train YOLOv7 model
python train.py --workers 8 --device 0 --batch-size 32 \
                --data data/coco.yaml --img 640 640 \
                --cfg cfg/training/yolov7.yaml --weights '' \
                --name yolov7_custom --hyp data/hyp.scratch.p5.yaml

# Train larger models
python train_aux.py --workers 8 --device 0 --batch-size 16 \
                    --data data/coco.yaml --img 1280 1280 \
                    --cfg cfg/training/yolov7-w6.yaml --weights '' \
                    --name yolov7-w6_custom --hyp data/hyp.scratch.p6.yaml
```

#### Multi-GPU Training
```bash
# 4 GPU training
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 \
       train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 \
       --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml \
       --weights '' --name yolov7_multi_gpu --hyp data/hyp.scratch.p5.yaml
```

#### Transfer Learning
```bash
# Fine-tune on custom dataset
python train.py --workers 8 --device 0 --batch-size 32 \
                --data data/custom.yaml --img 640 640 \
                --cfg cfg/training/yolov7-custom.yaml \
                --weights yolov7_training.pt --name yolov7_finetuned \
                --hyp data/hyp.scratch.custom.yaml
```

### 🧪 Testing

```bash
# Test on COCO validation set
python test.py --data data/coco.yaml --img 640 --batch 32 \
               --conf 0.001 --iou 0.65 --device 0 \
               --weights yolov7.pt --name yolov7_640_val

# Test custom model
python test.py --data data/custom.yaml --weights runs/train/exp/weights/best.pt
```

### 📤 Export

#### ONNX Export
```bash
# Basic ONNX export
python export.py --weights yolov7.pt --grid --end2end --simplify \
                 --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 \
                 --img-size 640 640 --max-wh 640
```

#### TensorRT Export
```bash
# Convert to TensorRT
python export.py --weights yolov7.pt --grid --end2end --simplify \
                 --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 \
                 --img-size 640 640

# Use TensorRT tools
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7.onnx -e yolov7.trt -p fp16
```

#### CoreML Export (macOS/iOS)
```bash
# Export for Apple devices
python export.py --weights yolov7.pt --include coreml
```

---

## 💡 Advanced Features

### 🏃 Pose Estimation

```bash
# Download pose model
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt

# Run pose detection
python detect_pose.py --weights yolov7-w6-pose.pt --source image.jpg
```

<div align="center">
<img src="../figure/pose.png" width="40%" alt="Pose Estimation"/>
</div>

### 🎭 Instance Segmentation

```bash
# Download segmentation model
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt

# Run instance segmentation
python detect_mask.py --weights yolov7-mask.pt --source image.jpg
```

<div align="center">
<img src="../figure/mask.png" width="60%" alt="Instance Segmentation"/>
</div>

### 🔧 Model Re-parameterization

For deployment optimization, use model re-parameterization:

```python
# See tools/reparameterization.ipynb for detailed examples
from models.experimental import attempt_load
model = attempt_load('yolov7.pt', map_location='cpu')
# Re-parameterization code here...
```

---

## 🛠 Troubleshooting

### 🚨 Common Issues & Solutions

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Out of Memory Error
```bash
# Reduce batch size
python train.py --batch-size 16  # instead of 32

# Use gradient accumulation
python train.py --batch-size 8 --accumulate 4  # effective batch size = 32
```

#### Slow Training
```bash
# Enable mixed precision
python train.py --amp

# Use multiple workers
python train.py --workers 8

# Optimize data loading
python train.py --cache ram  # cache images in RAM
```

### 📊 Performance Tips

1. **🎯 Optimal Batch Size**: Start with 32 and reduce if OOM
2. **⚡ Mixed Precision**: Use `--amp` flag for 2x speedup
3. **💾 Cache Images**: Use `--cache ram` for faster data loading
4. **🔄 Multi-GPU**: Scale batch size linearly with GPU count
5. **🎨 Image Size**: Use multiples of 32 (e.g., 640, 832, 1280)

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 🛠 Development Setup

```bash
# Fork and clone your fork
git clone https://github.com/YOUR_USERNAME/YOLOv7-in-GPU-Mode.git
cd YOLOv7-in-GPU-Mode

# Create development branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -r requirements_dev.txt

# Make your changes and test
python -m pytest tests/

# Submit pull request
git push origin feature/amazing-feature
```

### 📝 Guidelines

- 🔍 **Code Quality**: Follow PEP 8 and use type hints
- 🧪 **Testing**: Add tests for new features
- 📖 **Documentation**: Update docs for user-facing changes
- 🎯 **Performance**: Profile your changes for GPU optimization
- 🤝 **Review**: Be open to feedback and suggestions

---

## 📜 Citation

If you use YOLOv7 in your research, please cite:

```bibtex
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

---

## 🙏 Acknowledgements

### 🏆 Original Work
- **YOLOv7 Authors**: Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
- **Original Repository**: [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

### 🚀 Inspirations & Dependencies
- [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) - Original YOLO implementation
- [ultralytics/yolov5](https://github.com/ultralytics/yolov5) - YOLOv5 architecture
- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) - Anchor-free YOLO
- [PyTorch Team](https://pytorch.org/) - Deep learning framework
- [NVIDIA](https://developer.nvidia.com/) - CUDA and GPU acceleration

### 🌟 Special Thanks
- All contributors who made this GPU-optimized version possible
- The computer vision community for continuous innovation
- Open source maintainers for their dedication

---

<div align="center">

### 🌟 Star this repository if it helped you!

[![GitHub stars](https://img.shields.io/github/stars/abdullahjave/YOLOv7-in-GPU-Mode.svg?style=social&label=Star)](https://github.com/abdullahjave/YOLOv7-in-GPU-Mode)
[![GitHub forks](https://img.shields.io/github/forks/abdullahjave/YOLOv7-in-GPU-Mode.svg?style=social&label=Fork)](https://github.com/abdullahjave/YOLOv7-in-GPU-Mode/fork)

**Made with ❤️ for the Computer Vision Community**

[🔝 Back to Top](#-yolov7-gpu-mode---high-performance-object-detection)

</div>
