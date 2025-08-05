#!/bin/bash

# YOLOv7 GPU Mode Installation Script
# This script helps set up YOLOv7 with GPU acceleration

set -e  # Exit on any error

echo "🚀 YOLOv7 GPU Mode Installation Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if NVIDIA GPU is available
check_gpu() {
    print_status "Checking for NVIDIA GPU..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
        print_success "Found GPU: $GPU_INFO"
        return 0
    else
        print_warning "nvidia-smi not found. GPU acceleration may not be available."
        return 1
    fi
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 7 ]; then
        print_success "Python $PYTHON_VERSION is compatible"
        return 0
    else
        print_error "Python 3.7+ required. Found: $PYTHON_VERSION"
        return 1
    fi
}

# Install PyTorch with CUDA
install_pytorch() {
    print_status "Installing PyTorch with CUDA support..."
    
    # Detect CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        print_status "Detected CUDA version: $CUDA_VERSION"
    else
        print_warning "nvcc not found. Using default CUDA version."
        CUDA_VERSION="11.8"
    fi
    
    # Install PyTorch based on CUDA version
    if [[ $CUDA_VERSION == 11.8* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ $CUDA_VERSION == 12.* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        print_warning "Using CPU-only PyTorch. For GPU support, install manually."
        pip install torch torchvision torchaudio
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing YOLOv7 dependencies..."
    
    if [ -f "requirements_gpu.txt" ]; then
        pip install -r requirements_gpu.txt
    elif [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_error "Requirements file not found!"
        return 1
    fi
}

# Download pre-trained weights
download_weights() {
    print_status "Downloading pre-trained weights..."
    
    WEIGHTS_DIR="weights"
    mkdir -p $WEIGHTS_DIR
    
    # Download YOLOv7 weights
    WEIGHTS_URL="https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
    WEIGHTS_FILE="$WEIGHTS_DIR/yolov7.pt"
    
    if [ ! -f "$WEIGHTS_FILE" ]; then
        print_status "Downloading yolov7.pt..."
        wget -O "$WEIGHTS_FILE" "$WEIGHTS_URL" || curl -L -o "$WEIGHTS_FILE" "$WEIGHTS_URL"
        print_success "Downloaded yolov7.pt"
    else
        print_success "yolov7.pt already exists"
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('GPU not available - using CPU mode')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed!"
    else
        print_error "Installation test failed!"
        return 1
    fi
}

# Create test script
create_test_script() {
    print_status "Creating test script..."
    
    cat > test_gpu.py << 'EOF'
#!/usr/bin/env python3
"""
YOLOv7 GPU Test Script
Test basic functionality and GPU availability
"""

import torch
import sys
import os

def test_gpu():
    """Test GPU availability and PyTorch installation."""
    print("🧪 Testing YOLOv7 GPU Setup")
    print("=" * 40)
    
    # PyTorch info
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    
    # CUDA info
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"🎮 CUDA version: {torch.version.cuda}")
        print(f"🔢 GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"🏷  GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Memory info
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
        
        # Simple tensor operation test
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        print(f"🧮 GPU tensor operation test: {'✅ PASSED' if z.is_cuda else '❌ FAILED'}")
        
    else:
        print("❌ CUDA not available - using CPU mode")
        print("🔍 Check NVIDIA drivers and CUDA installation")
    
    print("\n🎯 Setup Complete! Ready to run YOLOv7")

if __name__ == "__main__":
    test_gpu()
EOF

    chmod +x test_gpu.py
    print_success "Created test_gpu.py"
}

# Main installation process
main() {
    echo
    print_status "Starting YOLOv7 GPU Mode installation..."
    echo
    
    # Run checks
    check_python || exit 1
    check_gpu
    
    echo
    print_status "Installing dependencies..."
    install_pytorch || exit 1
    install_dependencies || exit 1
    
    echo
    print_status "Setting up YOLOv7..."
    download_weights || print_warning "Could not download weights automatically"
    create_test_script
    
    echo
    print_status "Testing installation..."
    test_installation || exit 1
    
    echo
    print_success "🎉 Installation completed successfully!"
    echo
    print_status "Next steps:"
    echo "  1. Run: python3 test_gpu.py"
    echo "  2. Try detection: python3 detect.py --weights weights/yolov7.pt --source inference/images/horses.jpg"
    echo "  3. Check the README.md for more examples"
    echo
}

# Run main function
main "$@"
