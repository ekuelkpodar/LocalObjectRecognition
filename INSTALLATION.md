# Installation Guide

Complete installation guide for the Real-Time Object Recognition System.

## System Requirements

### Hardware Requirements

**Minimum:**
- NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 2080, etc.)
- 16GB System RAM
- 4-core CPU
- 30GB free disk space
- USB or integrated webcam

**Recommended:**
- NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4080, RTX 4090)
- 32GB System RAM
- 8+ core CPU
- 50GB free disk space (for multiple models)
- High-quality webcam (1080p)

### Software Requirements

**Operating System:**
- Linux: Ubuntu 20.04+ (recommended)
- Windows: Windows 10/11 with WSL2
- macOS: Limited GPU support (not recommended)

**Required Software:**
- Python 3.8 or higher
- CUDA Toolkit 11.8 or higher
- cuDNN compatible with your CUDA version
- Git (for cloning)

## Step-by-Step Installation

### 1. Install CUDA (if not already installed)

#### Ubuntu/Linux

```bash
# Check if CUDA is already installed
nvidia-smi

# If not installed, download from NVIDIA:
# https://developer.nvidia.com/cuda-downloads

# Example for Ubuntu 22.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### Windows

1. Download CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
2. Run the installer
3. Follow installation wizard
4. Restart computer

### 2. Set up Python Environment

#### Check Python Version

```bash
python --version  # or python3 --version

# Should show Python 3.8 or higher
```

#### Install Python (if needed)

**Ubuntu/Linux:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

**Windows:**
- Download from: https://www.python.org/downloads/
- Run installer
- Check "Add Python to PATH"

### 3. Clone/Download Project

```bash
# Navigate to desired directory
cd ~/Desktop  # or any directory you prefer

# If using git:
git clone <repository-url> LocalObjectRecognition
cd LocalObjectRecognition

# Or simply navigate to the project directory if already downloaded
cd LocalObjectRecognition
```

### 4. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# You should see (venv) in your prompt
```

### 5. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 6. Install PyTorch with CUDA Support

Visit https://pytorch.org/get-started/locally/ and select your configuration, or use:

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify PyTorch installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Should output:
```
PyTorch: 2.x.x+cu118
CUDA Available: True
```

### 7. Install Project Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- vLLM (inference engine)
- OpenCV (camera and display)
- NumPy (array operations)
- Pillow (image processing)
- PyYAML (configuration)
- psutil (system monitoring)
- transformers (model utilities)
- accelerate (optimization)

**Note**: This step may take 5-10 minutes and downloads ~4GB of packages.

### 8. Verify Installation

```bash
# Run verification script
python verify_setup.py
```

Expected output:
```
============================================================
Object Recognition System - Setup Verification
============================================================

Checking Python version...
  OK: Python 3.x.x

Checking dependencies...
  OK: opencv-python
  OK: numpy
  OK: pillow
  OK: pyyaml
  OK: torch
  OK: psutil

Checking optional/GPU dependencies...
  OK: vllm
  OK: transformers
  OK: accelerate

Checking CUDA...
  OK: CUDA available
  CUDA Version: 11.8
  GPU Count: 1
  GPU 0: NVIDIA GeForce RTX 3090 (24.0 GB)

Checking camera...
  OK: Camera accessible (device 0)

Checking configuration files...
  OK: config.yaml
  OK: prompts/templates.yaml
  OK: requirements.txt

Checking project structure...
  OK: src/
  OK: prompts/
  OK: logs/
  OK: src/__init__.py
  OK: src/camera.py
  OK: src/model.py
  OK: src/preprocessor.py
  OK: src/display.py
  OK: src/config.py
  OK: src/utils.py
  OK: main.py

============================================================
VERIFICATION SUMMARY
============================================================
✓ Python Version: PASS
✓ Dependencies: PASS
✓ CUDA/GPU: PASS
✓ Camera: PASS
✓ Configuration Files: PASS
✓ Project Structure: PASS

============================================================
All checks passed! ✓

You can now run the application:
  python main.py
============================================================
```

### 9. Test Basic Imports

```bash
python test_imports.py
```

Should show all modules importing successfully.

## Troubleshooting Installation

### Issue: CUDA not detected

**Symptoms:**
```
CUDA Available: False
```

**Solutions:**
1. Verify NVIDIA driver:
   ```bash
   nvidia-smi
   ```
2. Check CUDA installation:
   ```bash
   nvcc --version
   ```
3. Reinstall CUDA toolkit
4. Reinstall PyTorch with correct CUDA version

### Issue: vLLM installation fails

**Symptoms:**
```
ERROR: Failed building wheel for vllm
```

**Solutions:**
1. Ensure CUDA is installed first
2. Update pip:
   ```bash
   pip install --upgrade pip
   ```
3. Install with verbose output:
   ```bash
   pip install vllm -v
   ```
4. Check vLLM compatibility: https://docs.vllm.ai/

### Issue: Import errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solutions:**
1. Ensure virtual environment is activated
2. Reinstall package:
   ```bash
   pip install opencv-python
   ```
3. Check Python path

### Issue: Camera not accessible

**Symptoms:**
```
WARNING: Cannot open camera device 0
```

**Solutions:**
1. Check camera permissions (Linux):
   ```bash
   sudo usermod -a -G video $USER
   ```
   Then log out and back in
2. Try different device index in config.yaml
3. Check if camera is being used by another application
4. Test camera:
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'Failed')"
   ```

### Issue: Out of memory during model load

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. Close other GPU applications
2. Reduce `gpu_memory_util` in config.yaml:
   ```yaml
   model:
     gpu_memory_util: 0.6
   ```
3. Use smaller model (7B instead of 13B)
4. Clear GPU cache:
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

## Platform-Specific Notes

### Ubuntu/Linux

**Install system dependencies:**
```bash
sudo apt update
sudo apt install -y build-essential python3-dev python3-pip
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

**Fix camera permissions:**
```bash
sudo usermod -a -G video $USER
# Log out and log back in
```

### Windows with WSL2

1. Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
2. Install Ubuntu from Microsoft Store
3. Install CUDA in WSL2: https://docs.nvidia.com/cuda/wsl-user-guide/
4. Follow Linux installation steps above
5. For display, install VcXsrv or similar X server

### macOS

**Note**: macOS has limited GPU support with vLLM. CPU-only mode is not recommended for this application.

If attempting anyway:
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Follow standard installation steps
```

## Post-Installation

### First Run

```bash
python main.py
```

**What happens on first run:**
1. Model downloads (~15GB for LLaVA-1.5-7B)
   - Downloaded to: `~/.cache/huggingface/`
   - Takes 5-30 minutes depending on internet speed
2. Model loads into GPU (~20-30 seconds)
3. Application starts

### Configure for Your System

Edit `config.yaml`:

```yaml
camera:
  device_index: 0  # Change if using different camera

model:
  gpu_memory_util: 0.8  # Adjust based on your GPU VRAM
```

### Create Output Directory

```bash
mkdir -p output
```

This directory will store saved frames and detections.

## Verification Checklist

Before first use, verify:

- [ ] Python 3.8+ installed
- [ ] CUDA toolkit installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] PyTorch recognizes CUDA
- [ ] Camera accessible
- [ ] Configuration files present
- [ ] `python verify_setup.py` passes all checks

## Getting Help

If you encounter issues:

1. Run verification script:
   ```bash
   python verify_setup.py
   ```

2. Check logs:
   ```bash
   tail -f logs/object_recognition.log
   ```

3. Review [QUICKSTART.md](QUICKSTART.md) and [README.md](README.md)

4. Check system requirements

## Next Steps

After successful installation:

1. Read [QUICKSTART.md](QUICKSTART.md) for first-run guide
2. Review [README.md](README.md) for full documentation
3. Run the application: `python main.py`
4. Experiment with different prompts (keys 1-5)
5. Try different models (edit config.yaml)

---

**Installation complete! Ready to recognize objects in real-time.**
