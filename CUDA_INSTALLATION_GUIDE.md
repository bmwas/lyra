# CUDA Installation Guide for Remote Systems

This guide helps you install CUDA development tools to build Flash Attention and other CUDA packages from source.

## Current System Status
- **OS**: Fedora 42
- **Compiler**: GCC 15.2.1 ✅
- **Python Dev Headers**: Available ✅
- **CUDA Runtime**: 11.5 (limited)
- **CUDA Development**: ❌ Not installed

## Option 1: Install CUDA Toolkit (Requires sudo)

### Method A: NVIDIA Official Repository (Recommended)
```bash
# Add NVIDIA CUDA repository for Fedora
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo

# Install CUDA development toolkit
sudo dnf install cuda-toolkit-11-8 cuda-compiler-11-8 cuda-devel

# Set environment variables (add to ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Method B: RPM Fusion (Alternative)
```bash
# Enable RPM Fusion if not already enabled
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm

# Install CUDA development tools
sudo dnf install cuda-devel
```

### Method C: Manual Installation
1. Download CUDA 11.8 from [NVIDIA Developer](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Choose "Linux > x86_64 > Fedora > rpm (local)"
3. Follow installation instructions

## Option 2: Build Flash Attention from Source

After installing CUDA toolkit:

```bash
# Activate your virtual environment
source virtualenv/bin/activate

# Method 1: Direct installation from git
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4

# Method 2: Clone and build
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation

# Method 3: Use the updated install script
bash install_cuda_packages.sh
```

## Option 3: CPU Fallback (No sudo required)

If you cannot install CUDA development tools, use the CPU fallback:

```bash
# The fallback is already created: flash_attention_fallback.py
# Use it in your code like this:

# Instead of:
# from flash_attn import flash_attn_func

# Use:
from flash_attention_fallback import flash_attn_func
```

## Verification

After installation, verify everything works:

```bash
source virtualenv/bin/activate

# Test CUDA toolkit
nvcc --version

# Test Flash Attention
python -c "
try:
    from flash_attn import flash_attn_func
    print('✅ Flash Attention CUDA version installed')
except ImportError:
    from flash_attention_fallback import flash_attn_func
    print('✅ Flash Attention CPU fallback available')
    
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Performance Notes

- **CUDA Version**: ~10-100x faster than CPU for large sequences
- **CPU Fallback**: Functional but slower, good for development/testing
- **Memory**: CUDA version is more memory efficient for long sequences

## Troubleshooting

### Common Issues:
1. **Permission denied**: Use `sudo` for system installation
2. **Version conflicts**: Ensure CUDA version matches PyTorch
3. **Build errors**: Check that all development tools are installed

### Getting Help:
- Check NVIDIA CUDA documentation
- Verify your GPU supports the CUDA version
- Ensure sufficient disk space (CUDA toolkit ~3GB)

## Next Steps

After successful installation, you can:
1. Run the complete installation: `bash install_cuda_packages.sh`
2. Test Lyra functionality with CUDA acceleration
3. Benchmark performance differences between CUDA and CPU versions
