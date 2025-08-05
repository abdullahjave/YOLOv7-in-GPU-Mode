@echo off
REM YOLOv7 GPU Mode Installation Script for Windows
REM This script helps set up YOLOv7 with GPU acceleration on Windows

setlocal enabledelayedexpansion

echo 🚀 YOLOv7 GPU Mode Installation Script (Windows)
echo ===============================================

REM Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.7+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Found Python %PYTHON_VERSION%

REM Check for NVIDIA GPU
echo [INFO] Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] NVIDIA GPU detected
    for /f "skip=1 tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader,nounits') do (
        echo [INFO] GPU: %%i
        goto :gpu_found
    )
    :gpu_found
) else (
    echo [WARNING] nvidia-smi not found. GPU acceleration may not be available.
    echo Make sure NVIDIA drivers are installed.
)

REM Create virtual environment
echo [INFO] Creating virtual environment...
if exist "yolov7-env" (
    echo [INFO] Virtual environment already exists
) else (
    python -m venv yolov7-env
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Created virtual environment
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call yolov7-env\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA
echo [INFO] Installing PyTorch with CUDA support...
REM Check CUDA version
nvcc --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=4" %%i in ('nvcc --version ^| findstr "release"') do (
        set CUDA_VERSION=%%i
        set CUDA_VERSION=!CUDA_VERSION:,=!
        echo [INFO] Detected CUDA version: !CUDA_VERSION!
    )
    
    REM Install appropriate PyTorch version
    if "!CUDA_VERSION:~0,4!"=="11.8" (
        echo [INFO] Installing PyTorch for CUDA 11.8...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else if "!CUDA_VERSION:~0,2!"=="12" (
        echo [INFO] Installing PyTorch for CUDA 12.x...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else (
        echo [WARNING] Unknown CUDA version, installing default PyTorch...
        pip install torch torchvision torchaudio
    )
) else (
    echo [WARNING] nvcc not found, installing CPU-only PyTorch...
    pip install torch torchvision torchaudio
)

REM Install YOLOv7 dependencies
echo [INFO] Installing YOLOv7 dependencies...
if exist "requirements_gpu.txt" (
    pip install -r requirements_gpu.txt
) else if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo [ERROR] Requirements file not found!
    pause
    exit /b 1
)

REM Create weights directory
if not exist "weights" mkdir weights

REM Download pre-trained weights
echo [INFO] Downloading pre-trained weights...
if not exist "weights\yolov7.pt" (
    echo [INFO] Downloading yolov7.pt...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt' -OutFile 'weights\yolov7.pt'}"
    if %errorlevel% equ 0 (
        echo [SUCCESS] Downloaded yolov7.pt
    ) else (
        echo [WARNING] Failed to download weights automatically
        echo You can download manually from: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
    )
) else (
    echo [SUCCESS] yolov7.pt already exists
)

REM Create test script
echo [INFO] Creating test script...
(
echo #!/usr/bin/env python3
echo """
echo YOLOv7 GPU Test Script for Windows
echo Test basic functionality and GPU availability
echo """
echo.
echo import torch
echo import sys
echo import os
echo.
echo def test_gpu^(^):
echo     """Test GPU availability and PyTorch installation."""
echo     print^("🧪 Testing YOLOv7 GPU Setup"^)
echo     print^("=" * 40^)
echo     
echo     # PyTorch info
echo     print^(f"🐍 Python version: {sys.version}"^)
echo     print^(f"🔥 PyTorch version: {torch.__version__}"^)
echo     
echo     # CUDA info
echo     if torch.cuda.is_available^(^):
echo         print^(f"✅ CUDA available: {torch.cuda.is_available^(^)}"^)
echo         print^(f"🎮 CUDA version: {torch.version.cuda}"^)
echo         print^(f"🔢 GPU count: {torch.cuda.device_count^(^)}"^)
echo         
echo         for i in range^(torch.cuda.device_count^(^)^):
echo             print^(f"🏷  GPU {i}: {torch.cuda.get_device_name^(i^)}"^)
echo             
echo         # Memory info
echo         memory_allocated = torch.cuda.memory_allocated^(0^) / 1024**3
echo         memory_cached = torch.cuda.memory_reserved^(0^) / 1024**3
echo         print^(f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB"^)
echo         
echo         # Simple tensor operation test
echo         device = torch.device^('cuda:0'^)
echo         x = torch.randn^(1000, 1000^).to^(device^)
echo         y = torch.randn^(1000, 1000^).to^(device^)
echo         z = torch.mm^(x, y^)
echo         print^(f"🧮 GPU tensor operation test: {'✅ PASSED' if z.is_cuda else '❌ FAILED'}"^)
echo         
echo     else:
echo         print^("❌ CUDA not available - using CPU mode"^)
echo         print^("🔍 Check NVIDIA drivers and CUDA installation"^)
echo     
echo     print^("\n🎯 Setup Complete! Ready to run YOLOv7"^)
echo     input^("Press Enter to continue..."^)
echo.
echo if __name__ == "__main__":
echo     test_gpu^(^)
) > test_gpu.py

echo [SUCCESS] Created test_gpu.py

REM Test installation
echo [INFO] Testing installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
if %errorlevel% equ 0 (
    echo [SUCCESS] Installation test passed!
) else (
    echo [ERROR] Installation test failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] 🎉 Installation completed successfully!
echo.
echo Next steps:
echo   1. Run: python test_gpu.py
echo   2. Try detection: python detect.py --weights weights/yolov7.pt --source inference/images/horses.jpg
echo   3. Check the README.md for more examples
echo.
echo To activate the environment in the future, run:
echo   yolov7-env\Scripts\activate.bat
echo.

pause
