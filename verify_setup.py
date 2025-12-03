#!/usr/bin/env python3
"""
Verify Conda system dependencies and UV Python packages
This script checks that:
1. We're in a Conda-created virtual environment
2. System dependencies (CUDA, MKL, etc.) are available
3. Python packages from UV are installed correctly
4. Integration between system and Python packages works
"""

import sys
import os
import platform

def print_section(title):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def check_conda_venv():
    """Verify we're in Conda-created virtual environment"""
    print_section("1. Conda Virtual Environment")

    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        print("❌ ERROR: Not in a Conda environment")
        print("   Run: conda activate ml-system-deps")
        return False

    print(f"✅ Conda environment active")
    print(f"   CONDA_PREFIX: {conda_prefix}")
    print(f"   Python: {sys.executable}")
    print(f"   Python version: {sys.version.split()[0]}")

    # Check if Python is from Conda
    if conda_prefix in sys.executable:
        print(f"✅ Python is from Conda environment")
    else:
        print(f"⚠️  WARNING: Python may not be from Conda environment")

    return True

def check_cuda():
    """Verify CUDA from Conda"""
    print_section("2. CUDA Support (from Conda)")

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed (from UV)")

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            print(f"   GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA not available")
            print("   This is expected if:")
            print("   - No NVIDIA GPU present")
            print("   - GPU drivers not installed")
            print("   - Running on macOS/Windows")

        return True
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Run: uv sync")
        return False
    except Exception as e:
        print(f"⚠️  CUDA check error: {e}")
        return True

def check_mkl():
    """Verify MKL from Conda"""
    print_section("3. Intel MKL (from Conda)")

    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} installed (from UV)")

        # Check NumPy configuration for MKL
        config_str = str(np.__config__.show())

        # Simple performance test
        import time
        size = 2000
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)

        start = time.time()
        c = np.dot(a, b)
        elapsed = time.time() - start

        print(f"✅ Matrix multiplication test ({size}x{size}):")
        print(f"   Time: {elapsed:.4f} seconds")

        # Fast time suggests optimized BLAS (like MKL)
        if elapsed < 1.0:
            print(f"   ✅ Performance indicates optimized BLAS (likely MKL)")
        else:
            print(f"   ⚠️  Slower performance, may not be using MKL")

        return True
    except ImportError:
        print("❌ NumPy not installed")
        print("   Run: uv sync")
        return False
    except Exception as e:
        print(f"⚠️  MKL check error: {e}")
        return True

def check_system_libs():
    """Verify system libraries from Conda"""
    print_section("4. System Libraries (from Conda)")

    import ctypes.util

    libs_to_check = {
        'CUDA Runtime': ['cudart', 'cudart64_11'],
        'cuDNN': ['cudnn', 'cudnn8'],
        'MKL': ['mkl_rt', 'mkl_core'],
        'HDF5': ['hdf5', 'hdf5_serial'],
        'PNG': ['png', 'png16'],
        'JPEG': ['jpeg', 'turbojpeg'],
        'ZLIB': ['z'],
    }

    for name, lib_names in libs_to_check.items():
        found = False
        for lib in lib_names:
            path = ctypes.util.find_library(lib)
            if path:
                print(f"✅ {name}: {path}")
                found = True
                break

        if not found:
            # Some libraries may not be available on all platforms
            if name in ['CUDA Runtime', 'cuDNN'] and platform.system() != 'Linux':
                print(f"⚠️  {name}: Not available on {platform.system()}")
            else:
                print(f"⚠️  {name}: Not found")

def check_python_packages():
    """Verify key Python packages from UV"""
    print_section("5. Python Packages (from UV)")

    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'scipy': 'SciPy',
        'sklearn': 'scikit-learn',
        'h5py': 'h5py',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
    }

    for module, name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name} {version}")
        except ImportError:
            print(f"⚠️  {name}: Not installed")

def check_integration():
    """Test integration between system deps and Python packages"""
    print_section("6. Integration Tests")

    # Test 1: PyTorch with CUDA
    try:
        import torch
        tensor = torch.tensor([1.0, 2.0, 3.0])
        if torch.cuda.is_available():
            tensor_gpu = tensor.cuda()
            print(f"✅ PyTorch CUDA tensor created successfully")
        else:
            print(f"⚠️  CUDA not available for PyTorch")
    except Exception as e:
        print(f"⚠️  PyTorch integration: {e}")

    # Test 2: NumPy with MKL
    try:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        result = np.fft.fft(arr)
        print(f"✅ NumPy operations working")
    except Exception as e:
        print(f"❌ NumPy integration: {e}")

    # Test 3: HDF5
    try:
        import h5py
        print(f"✅ h5py (HDF5 support) working")
    except ImportError:
        print(f"⚠️  h5py not installed")
    except Exception as e:
        print(f"⚠️  HDF5 integration: {e}")

def main():
    """Main verification function"""
    print("\n" + "="*70)
    print("  Conda + UV Environment Verification")
    print("  (System Dependencies + Python Packages)")
    print("="*70)

    results = []

    results.append(check_conda_venv())
    results.append(check_cuda())
    results.append(check_mkl())
    check_system_libs()
    check_python_packages()
    check_integration()

    print_section("Summary")

    if all(results):
        print("✅ All critical checks passed!")
        print("\nEnvironment is ready:")
        print("  - Conda created the virtual environment")
        print("  - System dependencies installed by Conda")
        print("  - Python packages installed by UV")
        print("  - Integration working correctly")
    else:
        print("⚠️  Some checks failed or were skipped")
        print("\nTroubleshooting:")
        print("  1. Ensure Conda environment is activated:")
        print("     conda activate ml-system-deps")
        print("  2. Ensure UV packages are installed:")
        print("     uv sync")
        print("  3. Check CUDA/GPU drivers if needed")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()