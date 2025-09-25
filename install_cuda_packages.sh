#!/bin/bash

# Installation script for CUDA packages that require special handling
# Make sure to run this AFTER installing requirements_gen3c.txt

echo "Installing CUDA packages for Lyra..."
echo "This script requires:"
echo "1. torch and torchvision already installed"
echo "2. nvcc available in PATH"
echo "3. CUDA development tools installed"
echo ""

# Activate virtual environment if it exists
if [ -d "virtualenv" ]; then
    echo "Activating virtual environment..."
    source virtualenv/bin/activate
    # Verify virtual environment is activated
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "✓ Virtual environment activated: $VIRTUAL_ENV"
    else
        echo "WARNING: Virtual environment activation may have failed"
    fi
else
    echo "WARNING: virtualenv directory not found. Make sure you're in the correct directory."
    echo "Current directory: $(pwd)"
fi

# Check Python and pip paths
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"

# Check if torch is installed
python -c "import torch; print(f'Torch version: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: torch is not installed. Please install requirements_gen3c.txt first."
    exit 1
fi

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found in PATH. CUDA development tools may not be installed."
    echo "This means CUDA packages (flash_attn, causal_conv1d, etc.) cannot be compiled."
    echo ""
    echo "REMOTE SYSTEM OPTIONS:"
    echo "1. Install CUDA development tools (requires admin access)"
    echo "2. Use CPU-only versions where available"
    echo "3. Skip CUDA packages and continue with available packages"
    echo ""
    read -p "Continue anyway? (y/N): " choice
    if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
    echo "Continuing without CUDA compilation support..."
fi

echo "Torch found, proceeding with CUDA package installation..."
echo ""

# Set CUDA environment variables
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
# Install missing dependencies first
echo "Installing build dependencies..."
python -m pip install --upgrade pip setuptools wheel packaging ninja --break-system-packages

# Set environment variables to help with installation
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

echo "Installing Flash Attention..."
if command -v nvcc &> /dev/null; then
    echo "NVCC found, attempting CUDA build..."
    python -m pip install flash_attn==2.7.4.post1 --no-build-isolation --verbose --break-system-packages
    
    if [ $? -ne 0 ]; then
        echo "❌ CUDA build failed. Attempting source build..."
        echo "Cloning Flash Attention repository..."
        git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
        cd /tmp/flash-attention
        python -m pip install . --no-build-isolation --verbose --break-system-packages
        cd - > /dev/null
        
        if [ $? -ne 0 ]; then
            echo "❌ Source build also failed."
            echo "✅ CPU fallback available: see flash_attention_fallback.py"
        fi
    fi
else
    echo "⚠️  NVCC not available, skipping Flash Attention CUDA build"
    echo "✅ CPU fallback available: see flash_attention_fallback.py"
    echo "   You can import flash_attention_fallback as flash_attn_func"
fi

echo "Installing Causal Conv1D..."
python -m pip install git+https://github.com/Dao-AILab/causal-conv1d@v1.4.0 --no-build-isolation --verbose --break-system-packages

echo "Installing GSplat..."
python -m pip install git+https://github.com/nerfstudio-project/gsplat.git@73fad53c31ec4d6b088470715a63f432990493de --no-build-isolation --verbose --break-system-packages

echo "Installing Fused SSIM..."
python -m pip install git+https://github.com/rahul-goel/fused-ssim/@8bdb59feb7b9a41b1fab625907cb21f5417deaac --no-build-isolation --verbose --break-system-packages

echo ""
echo "CUDA package installation completed!"
echo "If any package failed to install, you can try installing it individually with:"
echo "pip install --no-build-isolation --break-system-packages <package_name>"
