#!/bin/bash
# Setup script for ML environment with Conda system deps + UV packages

set -e  # Exit on error

echo "======================================================================="
echo "  ML Environment Setup: Conda (system deps) + UV (Python packages)"
echo "======================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is installed
echo "Checking prerequisites..."
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ ERROR: Conda is not installed${NC}"
    echo ""
    echo "Please install Miniconda or Anaconda:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    exit 1
fi
echo -e "${GREEN}✅ Conda found${NC}"

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}⚠️  UV not found, installing...${NC}"
    pip install uv || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo -e "${GREEN}✅ UV found${NC}"

echo ""
echo "======================================================================="
echo "  Step 1: Conda creates virtual environment + system dependencies"
echo "======================================================================="
echo ""
echo "This will install:"
echo "  - Python 3.11 interpreter"
echo "  - CUDA Toolkit 11.8 (GPU computing)"
echo "  - cuDNN 8.9 (Deep learning primitives)"
echo "  - Intel MKL (Optimized linear algebra)"
echo "  - System libraries (HDF5, PNG, JPEG, etc.)"
echo ""

# Create or update Conda environment
if conda env list | grep -q "ml-system-deps"; then
    echo -e "${YELLOW}Environment 'ml-system-deps' exists, updating...${NC}"
    conda env update -f environment.yml --prune
else
    echo "Creating new environment 'ml-system-deps'..."
    conda env create -f environment.yml
fi

echo ""
echo -e "${GREEN}✅ Conda environment created successfully${NC}"
echo ""

echo "======================================================================="
echo "  Step 2: Next Steps - Activate and Install Python Packages"
echo "======================================================================="
echo ""
echo -e "${YELLOW}Important: You must activate the Conda environment before using UV${NC}"
echo ""
echo "Run these commands:"
echo ""
echo -e "${GREEN}  1. Activate Conda environment:${NC}"
echo "     conda activate ml-system-deps"
echo ""
echo -e "${GREEN}  2. Install Python packages with UV:${NC}"
echo "     uv sync"
echo ""
echo -e "${GREEN}  3. Verify setup:${NC}"
echo "     python verify_setup.py"
echo ""
echo -e "${GREEN}  4. Test GPU (if available):${NC}"
echo "     python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""

echo "======================================================================="
echo "  Environment Information"
echo "======================================================================="
echo ""
echo "Conda environment location:"
conda env list | grep ml-system-deps
echo ""
echo "To activate: conda activate ml-system-deps"
echo "To deactivate: conda deactivate"
echo "To remove: conda env remove -n ml-system-deps"
echo ""
echo "======================================================================="
echo "  Setup script completed!"
echo "======================================================================="