# Setup Guide - PyTorch for Financial Derivatives

This guide explains how to set up your Python environment for running the PyTorch derivatives pricing examples.

---

## 🚀 Quick Start

### Automatic Setup (Recommended)

The easiest way to set up the environment is using the provided setup script:

```bash
# CPU-only installation (recommended for most users)
bash setup.sh

# GPU installation (if you have NVIDIA GPU with CUDA)
bash setup.sh --gpu

# GPU with specific CUDA version
bash setup.sh --gpu --cuda-version cu121
```

The script will:
1. ✅ Check Python version (3.8+ required)
2. ✅ Create virtual environment
3. ✅ Install PyTorch and dependencies
4. ✅ Verify installation
5. ✅ Run test suite

**Time**: ~5-10 minutes depending on internet speed

---

## 📋 Prerequisites

### Required

- **Python 3.8 or higher**
  - Check: `python3 --version`
  - Install:
    - Ubuntu/Debian: `sudo apt-get install python3 python3-venv python3-pip`
    - macOS: `brew install python3`
    - Windows: Download from [python.org](https://python.org)

### Optional (for GPU acceleration)

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** 11.8 or 12.1
  - Download from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
- **cuDNN** (usually included with PyTorch)

---

## 🛠️ Manual Setup

If you prefer to set up manually or the script doesn't work:

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv pytorch_derivatives_env

# Activate it
# On Linux/macOS:
source pytorch_derivatives_env/bin/activate

# On Windows:
pytorch_derivatives_env\Scripts\activate
```

### Step 2: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 3: Install PyTorch

**Option A: CPU-only (easier, works everywhere)**

```bash
pip install torch torchvision torchaudio
```

**Option B: GPU with CUDA 11.8**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Option C: GPU with CUDA 12.1**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Other Dependencies

```bash
pip install numpy scipy matplotlib jupyter ipython
```

Or if you have the requirements file:

```bash
pip install -r requirements_pytorch.txt
```

---

## ✅ Verify Installation

### Test PyTorch

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

Expected output: `PyTorch version: 2.x.x`

### Test CUDA (if GPU installed)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output: `CUDA available: True` (for GPU) or `False` (for CPU-only)

### Run Quick Test

```python
import torch

# Test autograd
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(f"dy/dx at x=3: {x.grad.item()}")  # Should print 8.0

print("✓ PyTorch is working correctly!")
```

---

## 🔧 Troubleshooting

### Problem: "python3: command not found"

**Solution**: Python is not installed or not in PATH
- Install Python 3.8 or higher
- Add Python to your PATH environment variable

### Problem: "No module named 'venv'"

**Solution**: venv module not installed
```bash
# Ubuntu/Debian
sudo apt-get install python3-venv

# Usually included on macOS and Windows
```

### Problem: "No module named 'torch'"

**Solution**: PyTorch not installed or wrong environment
- Make sure virtual environment is activated
- Reinstall PyTorch: `pip install torch`

### Problem: CUDA not available despite GPU

**Solution**: Multiple possible causes
1. Check NVIDIA drivers: `nvidia-smi`
2. Check CUDA version compatibility
3. Reinstall PyTorch with correct CUDA version
4. For testing purposes, CPU-only version works fine

### Problem: "Permission denied" when running setup.sh

**Solution**: Make script executable
```bash
chmod +x setup.sh
```

### Problem: Slow installation

**Solution**: Normal for first time
- PyTorch is large (~800 MB for CPU, ~2 GB for GPU)
- Use a good internet connection
- Be patient, it's a one-time setup

---

## 💻 Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# Install prerequisites
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip

# Run setup
bash setup.sh
```

### macOS

```bash
# Install prerequisites (if needed)
brew install python3

# Run setup
bash setup.sh

# Note: GPU not supported on macOS (use CPU version)
```

### Windows

**Option 1: WSL2 (Recommended)**
```bash
# Install WSL2 with Ubuntu
wsl --install

# Then follow Linux instructions above
```

**Option 2: Native Windows**
```powershell
# Use PowerShell or Command Prompt
python -m venv pytorch_derivatives_env
pytorch_derivatives_env\Scripts\activate
pip install torch numpy scipy matplotlib jupyter

# Or adapt setup.sh for Windows (requires Git Bash or similar)
```

---

## 📦 What Gets Installed

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | PyTorch - automatic differentiation |
| numpy | ≥1.20.0 | Numerical computing |
| scipy | ≥1.7.0 | Scientific computing (for validation) |
| matplotlib | ≥3.4.0 | Visualization |
| jupyter | ≥1.0.0 | Interactive notebooks |

### Disk Space

- CPU version: ~800 MB
- GPU version: ~2 GB
- With all dependencies: ~1-3 GB total

### Installation Time

- Fast internet: 5 minutes
- Slow internet: 15-20 minutes
- First time only; subsequent activations are instant

---

## 🎯 After Setup

### Activate Environment

Every time you want to use the package, activate the environment first:

```bash
# Linux/macOS
source pytorch_derivatives_env/bin/activate

# Windows
pytorch_derivatives_env\Scripts\activate
```

You'll see `(pytorch_derivatives_env)` in your prompt.

### Run Examples

```bash
# Run standalone demo
python pytorch_derivatives.py

# Open Jupyter notebook
jupyter notebook pytorch_derivatives_demo.ipynb

# Interactive Python
python
>>> import torch
>>> from pytorch_derivatives import PyTorchOptionPricer
>>> pricer = PyTorchOptionPricer()
```

### Deactivate Environment

When you're done:

```bash
deactivate
```

---

## 🔄 Updating

### Update PyTorch

```bash
# Activate environment first
source pytorch_derivatives_env/bin/activate

# Update PyTorch
pip install --upgrade torch

# Update all packages
pip install --upgrade -r requirements_pytorch.txt
```

### Recreate Environment

If something goes wrong, just recreate:

```bash
# Remove old environment
rm -rf pytorch_derivatives_env

# Run setup again
bash setup.sh
```

---

## 🎓 Understanding the Setup

### Why Virtual Environment?

Virtual environments isolate project dependencies:
- ✅ No conflicts with other Python projects
- ✅ Reproducible setup
- ✅ Easy to delete and recreate
- ✅ Doesn't affect system Python

### Why These Packages?

- **PyTorch**: Automatic differentiation engine (core requirement)
- **NumPy**: Fast numerical arrays (PyTorch's foundation)
- **SciPy**: Analytical solutions for validation (e.g., normal distribution)
- **Matplotlib**: Visualizations (Delta surfaces, convergence plots)
- **Jupyter**: Interactive learning environment

### CPU vs GPU

**CPU version**:
- ✅ Works everywhere
- ✅ Smaller download
- ✅ Sufficient for learning
- ⚠️ Slower for large simulations

**GPU version**:
- ✅ 10-100x faster for Monte Carlo
- ✅ Required for large-scale production
- ⚠️ Requires NVIDIA GPU
- ⚠️ Larger download

**Recommendation**: Start with CPU, upgrade to GPU if needed.

---

## 📞 Getting Help

### Setup Issues

1. Check this guide's troubleshooting section
2. Verify prerequisites are met
3. Try manual setup instead of script
4. Check Python and pip versions
5. Look for error messages carefully

### After Setup

Once setup is complete, the package includes:
- `README.md` - Quick start guide
- `PACKAGE_INDEX.md` - Complete documentation
- `PROJECT_COMPLETE.md` - Learning paths

### Still Stuck?

Common resources:
- PyTorch documentation: pytorch.org/get-started
- Python venv guide: docs.python.org/3/library/venv.html
- NumPy installation: numpy.org/install

---

## ✨ Quick Reference

### One-Time Setup

```bash
# Clone or download package
cd pytorch_derivatives_package

# Run setup
bash setup.sh

# Time: 5-10 minutes
```

### Daily Usage

```bash
# Activate
source pytorch_derivatives_env/bin/activate

# Use
python pytorch_derivatives.py

# Deactivate when done
deactivate
```

### Verification

```bash
# Check Python
python --version  # Should be 3.8+

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check CUDA (if GPU)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🎉 Ready!

Once setup completes successfully:

1. ✅ Virtual environment is ready
2. ✅ All packages installed
3. ✅ Installation verified
4. ✅ Tests passed

**Next steps**:
```bash
# Activate environment
source pytorch_derivatives_env/bin/activate

# Run the demo!
python pytorch_derivatives.py
```

**Happy computing!** 🚀

---

**Setup Script Version**: 1.0  
**Last Updated**: November 2024  
**Tested On**: Ubuntu 20.04+, macOS 12+, Windows 10+ (WSL2)
