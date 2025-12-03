# Phase 4 Combined Integration Test: UV + Conda with System Dependencies

## Test ID
T-P4-009

## Category
Combined Integration - UV with Conda System-Level Dependencies

## Priority
P1

## Description
This fixture demonstrates a real-world ML/Deep Learning project where Conda is **required** for system-level dependencies (CUDA, cuDNN, MKL, compilers), Conda **creates the virtual environment**, and UV manages Python packages for performance.

## Why This Matters

### System Dependencies That REQUIRE Conda:
- **CUDA Toolkit** - GPU computing libraries (not available via pip)
- **cuDNN** - Deep learning primitives for CUDA (not available via pip)
- **MKL** - Intel Math Kernel Library (optimized BLAS/LAPACK)
- **Compilers** - GCC, G++, Fortran compilers
- **System Libraries** - OpenBLAS, HDF5, FFMPEG, etc.

### The Conda+UV Hybrid Approach:
1. **Conda creates venv** - Isolated environment
2. **Conda installs system deps** - CUDA, MKL, compilers
3. **UV installs Python packages** - Fast, reproducible

## Real-World Use Case

**Scenario:** Deep Learning team training models on GPU

**Requirements:**
- CUDA 11.8 for GPU support (binary, not pip-installable)
- cuDNN for optimized neural networks
- MKL for fast CPU operations
- PyTorch with CUDA support (Python package)
- NumPy compiled against MKL (faster linear algebra)

**Solution:**
- Conda provides: CUDA, cuDNN, MKL (system binaries)
- Conda creates: Virtual environment
- UV provides: PyTorch, NumPy, and all Python packages (fast)

## Workflow

### Step 1: Conda Creates Venv + System Dependencies
```bash
# Conda creates virtual environment AND installs system libraries
conda env create -f environment.yml

# This installs:
# - Python 3.11 interpreter
# - CUDA Toolkit 11.8 (binary libraries)
# - cuDNN 8.x (binary libraries)
# - Intel MKL (optimized math libraries)
# - C/C++ compilers
```

**Conda's Role:**
✅ **Creates virtual environment** (venv)
✅ Installs system-level binaries
✅ Manages binary dependencies
✅ Sets up LD_LIBRARY_PATH and environment variables

### Step 2: UV Manages Python Packages
```bash
# Activate the Conda-created environment
conda activate ml-system-deps

# UV installs Python packages into Conda's venv
uv sync

# This installs:
# - PyTorch (compiled against CUDA 11.8)
# - NumPy (uses Conda's MKL)
# - All other Python packages
```

**UV's Role:**
✅ Fast package resolution
✅ Lock file for reproducibility
✅ Installs into Conda's venv
✅ Manages Python-only dependencies

## File Structure

```
uv_conda_system_deps/
├── environment.yml          # Conda: venv + system dependencies
├── pyproject.toml          # UV: Python packages
├── uv.lock                 # UV: lock file
├── README.md               # This file
├── setup_env.sh            # Setup script
└── verify_setup.py         # Verification script
```

## Environment Configuration

### environment.yml - Conda Creates Venv + System Deps

```yaml
name: ml-system-deps

channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  # ===== CONDA CREATES VENV WITH PYTHON =====
  - python=3.11

  # ===== SYSTEM-LEVEL DEPENDENCIES (NOT AVAILABLE VIA PIP) =====

  # CUDA Toolkit (GPU computing)
  - cudatoolkit=11.8

  # cuDNN (Deep learning primitives)
  - cudnn=8.9

  # Intel MKL (Optimized linear algebra)
  - mkl=2023.2
  - mkl-include

  # Compilers (for building extensions)
  - gxx_linux-64  # Linux only
  - gcc_linux-64  # Linux only

  # System libraries
  - hdf5  # For HDF5 file format
  - libpng  # Image processing
  - libjpeg-turbo  # JPEG support

  # DO NOT ADD PYTHON PACKAGES HERE!
  # Python packages managed by UV in pyproject.toml
```

### pyproject.toml - UV Manages Python Packages

```toml
[project]
name = "ml-system-deps"
version = "1.0.0"
description = "ML project with Conda system deps + UV packages"
requires-python = ">=3.11"

dependencies = [
    # Deep Learning
    "torch>=2.0.0",  # Will use Conda's CUDA
    "torchvision>=0.15.0",
    "torchaudio>=2.0.0",

    # Scientific Computing (will use Conda's MKL)
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.11.0",

    # ML Libraries
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",

    # Data Processing
    "h5py>=3.9.0",  # Uses Conda's HDF5
    "pillow>=10.0.0",  # Uses Conda's libjpeg/libpng

    # Utilities
    "tqdm>=4.66.0",
    "matplotlib>=3.7.0",
]
```

## Test Objectives

1. **Verify Conda creates venv** - Not just installs packages
2. **Verify system dependencies installed** - CUDA, MKL, compilers
3. **Verify UV uses Conda's venv** - Packages go into Conda environment
4. **Verify Python packages link to system deps** - NumPy uses MKL, PyTorch uses CUDA
5. **Verify no conflicts** - System deps from Conda, Python from UV
6. **Verify environment reproducibility** - Lock file + environment.yml

## Success Criteria

### Conda Environment Creation:
- [ ] Conda creates virtual environment successfully
- [ ] Python 3.11 installed in venv
- [ ] CUDA Toolkit 11.8 binaries present
- [ ] cuDNN libraries installed
- [ ] MKL libraries installed
- [ ] Environment variables set (LD_LIBRARY_PATH, etc.)

### UV Package Installation:
- [ ] UV detects Conda's Python interpreter
- [ ] UV installs packages into Conda's venv
- [ ] uv.lock generated successfully
- [ ] PyTorch links to CUDA libraries
- [ ] NumPy links to MKL libraries

### Integration Verification:
- [ ] PyTorch GPU support enabled
- [ ] CUDA devices detected
- [ ] NumPy uses MKL (fast linear algebra)
- [ ] No package conflicts
- [ ] Environment reproducible

## Setup Instructions

### Prerequisites:
- Conda (Miniconda or Anaconda) installed
- UV installed
- NVIDIA GPU (for CUDA testing)
- Linux OS (for compiler packages)

### Setup Commands:
```bash
# 1. Conda creates environment with system deps
conda env create -f environment.yml

# This creates:
# - Virtual environment: ~/miniconda3/envs/ml-system-deps/
# - Python interpreter: ~/miniconda3/envs/ml-system-deps/bin/python
# - CUDA libraries: ~/miniconda3/envs/ml-system-deps/lib/libcudart.so
# - cuDNN libraries: ~/miniconda3/envs/ml-system-deps/lib/libcudnn.so
# - MKL libraries: ~/miniconda3/envs/ml-system-deps/lib/libmkl_*.so

# 2. Activate Conda-created environment
conda activate ml-system-deps

# 3. Verify Conda environment
which python
# Should show: ~/miniconda3/envs/ml-system-deps/bin/python

echo $CONDA_PREFIX
# Should show: ~/miniconda3/envs/ml-system-deps

# 4. UV installs Python packages into Conda venv
uv sync

# 5. Verify setup
python verify_setup.py
```

## Verification Script

### verify_setup.py:
```python
#!/usr/bin/env python3
"""Verify Conda system dependencies and UV Python packages"""

import sys
import os

def check_conda_venv():
    """Verify we're in Conda environment"""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        print("❌ Not in Conda environment")
        return False
    print(f"✅ Conda environment: {conda_prefix}")
    return True

def check_cuda():
    """Verify CUDA from Conda"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   GPU devices: {torch.cuda.device_count()}")
            print(f"   GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available (no GPU or drivers)")
        return True
    except Exception as e:
        print(f"❌ PyTorch/CUDA error: {e}")
        return False

def check_mkl():
    """Verify MKL from Conda"""
    try:
        import numpy as np
        config = np.__config__.show()
        # Check if MKL is being used
        print("✅ NumPy installed")
        print("   Checking for MKL...")
        # MKL info is in config
        return True
    except Exception as e:
        print(f"❌ NumPy error: {e}")
        return False

def check_system_libs():
    """Verify system libraries from Conda"""
    import ctypes.util

    libs = {
        'CUDA Runtime': 'cudart',
        'cuDNN': 'cudnn',
        'MKL': 'mkl_rt',
        'HDF5': 'hdf5',
    }

    for name, lib in libs.items():
        path = ctypes.util.find_library(lib)
        if path:
            print(f"✅ {name}: {path}")
        else:
            print(f"⚠️  {name}: not found")

def main():
    print("="*60)
    print("Conda + UV Environment Verification")
    print("="*60)
    print()

    print("1. Conda Environment:")
    check_conda_venv()
    print()

    print("2. CUDA Support:")
    check_cuda()
    print()

    print("3. NumPy/MKL:")
    check_mkl()
    print()

    print("4. System Libraries:")
    check_system_libs()
    print()

    print("="*60)
    print("Verification Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
```

## Platform-Specific Notes

### Linux (Primary Target):
✅ Full support for CUDA, cuDNN, MKL, compilers
✅ All system dependencies available

### macOS:
⚠️  No CUDA support (no NVIDIA GPUs)
✅ Can use MKL for CPU optimization
✅ Alternative: Use MPS backend for Apple Silicon

### Windows:
⚠️  CUDA available but paths differ
⚠️  Some compilers not available
✅ Can use MKL

## Why Conda Is Required Here

### Cannot Use Pip Alone:
❌ CUDA Toolkit - Binary libraries, not Python packages
❌ cuDNN - System libraries, complex installation
❌ MKL - Proprietary Intel libraries
❌ Compilers - System-level tools
❌ Proper library linking - LD_LIBRARY_PATH setup

### Why UV Alone Isn't Enough:
UV is excellent for Python packages but cannot:
- Install system-level binaries
- Manage CUDA libraries
- Set up compiler toolchains
- Configure library paths

### The Perfect Combination:
✅ Conda: System dependencies + venv creation
✅ UV: Fast Python package management
✅ Both: Work together seamlessly

## Testing Scenarios

### Scenario 1: GPU Training
```python
import torch

# CUDA from Conda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# PyTorch from UV
model = torch.nn.Linear(10, 1).to(device)
```

### Scenario 2: Fast Linear Algebra
```python
import numpy as np

# NumPy from UV, MKL from Conda
a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)

# Uses MKL for fast computation
c = np.dot(a, b)
```

### Scenario 3: HDF5 Data
```python
import h5py

# h5py from UV, HDF5 libs from Conda
with h5py.File('data.h5', 'w') as f:
    f.create_dataset('data', data=[1, 2, 3])
```

## Common Issues & Solutions

### Issue 1: PyTorch Not Finding CUDA
**Problem:** PyTorch installed but CUDA not detected
**Solution:**
```bash
# Verify CUDA in Conda environment
conda list | grep cuda

# Reinstall PyTorch with correct CUDA version
# Use UV to install PyTorch that matches Conda's CUDA
```

### Issue 2: NumPy Not Using MKL
**Problem:** NumPy runs slow, not using MKL
**Solution:**
```bash
# Verify MKL installed
conda list | grep mkl

# NumPy from UV should automatically use Conda's MKL
python -c "import numpy; numpy.__config__.show()"
```

### Issue 3: Library Not Found
**Problem:** Libraries not found at runtime
**Solution:**
```bash
# Ensure Conda environment is activated
conda activate ml-system-deps

# Check environment variables
echo $LD_LIBRARY_PATH
echo $CONDA_PREFIX
```

## CI/CD Pipeline

```yaml
# .github/workflows/test-conda-uv.yml
name: Test Conda + UV

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Conda
        uses: conda-incubator/setup-miniconda@v2
        with:
          environment-file: environment.yml
          activate-environment: ml-system-deps
          auto-activate-base: false

      - name: Install UV
        run: pip install uv

      - name: Install packages with UV
        shell: bash -l {0}
        run: uv sync

      - name: Verify setup
        shell: bash -l {0}
        run: python verify_setup.py

      - name: Run tests
        shell: bash -l {0}
        run: pytest
```

## Mend Integration Notes

**Scanning Hybrid Projects:**
- Mend should detect environment.yml (Conda)
- Mend should detect pyproject.toml + uv.lock (UV)
- System dependencies from Conda noted but not scanned
- Python packages from UV fully scanned for vulnerabilities

**Configuration:**
```properties
# Scan UV packages
python.packageManager=uv
python.resolveDependencies=true

# Note Conda system dependencies
# (context only, not scanned for CVEs)
```

## Related Tests
- T-P4-008: UV with Conda Environment (basic)
- T-P4-003: Data Science Platform
- T-P2-009: Platform Markers

## UV Version Compatibility
- Minimum: UV 0.4.0+
- Recommended: UV 0.7.0+

## Conda Version Compatibility
- Minimum: Conda 4.x+
- Recommended: Conda 23.x+ or Mamba

## Documentation Links
- CUDA Installation: https://docs.nvidia.com/cuda/
- cuDNN: https://developer.nvidia.com/cudnn
- Intel MKL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
- Conda + UV Best Practices: [Internal documentation]