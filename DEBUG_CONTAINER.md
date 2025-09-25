# Docker Container Debugging Guide

## Issue: Torch Module Not Found

You're encountering `ModuleNotFoundError: No module named 'torch'` even though torch should have been installed during the Docker build.

## 🔍 Debugging Steps

### 1. First, rebuild the Docker image to ensure a clean build:

```bash
# Clean rebuild (this will take ~15-30 minutes)
docker build --no-cache -t lyra:latest .
```

### 2. Test the container without overriding the entrypoint:

```bash
# Test normal entrypoint (should show debug info)
docker run -it --gpus all -v $(pwd)/assets:/app/assets lyra:latest

# Test with interactive mode
docker run -it --gpus all -v $(pwd)/assets:/app/assets lyra:latest --interactive
```

### 3. Manual package verification:

```bash
# Check what's actually installed
docker run -it --gpus all lyra:latest bash -c "pip list | grep torch"

# Verify Python environment
docker run -it --gpus all lyra:latest bash -c "python -c 'import sys; print(sys.version); print(sys.path)'"
```

## 🛠️ Potential Solutions

### Solution 1: Rebuild with verbose output
```bash
# Build with detailed output to catch errors
docker build --progress=plain --no-cache -t lyra:latest . 2>&1 | tee build.log
```

### Solution 2: Manual package installation in running container
```bash
# Start container and install packages manually
docker run -it --gpus all lyra:latest bash

# Inside container:
pip install --no-cache-dir --break-system-packages -r requirements_gen3c.txt
pip install --no-cache-dir --break-system-packages -r requirements_lyra.txt
python -c "import torch; print(f'Torch: {torch.__version__}')"
```

### Solution 3: Alternative base image approach
If the HuggingFace base image has conflicts, try a simpler approach:

```dockerfile
# Alternative Dockerfile start
FROM nvidia/cuda:12.1-devel-ubuntu22.04
USER root
# ... rest of installation
```

## 🚨 Common Causes

1. **Build cache issues**: Use `--no-cache` flag
2. **Base image conflicts**: The TGI base image might have package conflicts
3. **Python path issues**: Environment variables not set correctly
4. **Permission issues**: User switching affecting package installation

## 📋 Enhanced Entrypoint Features

The updated entrypoint now includes:
- ✅ **Detailed Python environment debugging**
- ✅ **Automatic package reinstallation if missing**
- ✅ **Graceful failure handling**
- ✅ **Better error messages and suggestions**

## 🔧 If All Else Fails

Create a minimal test:

```bash
# Test minimal torch installation
docker run -it --gpus all ubuntu:22.04 bash -c "
  apt update && apt install -y python3 python3-pip
  pip install torch==2.6.0 --break-system-packages
  python3 -c 'import torch; print(torch.__version__)'
"
```

This will help isolate whether it's a base image issue or our Dockerfile issue.
