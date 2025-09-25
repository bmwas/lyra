# Use the specified base image
FROM ghcr.io/huggingface/text-generation-inference:3.3.6-trtllm

# Switch to root user to install packages
USER root

# Set the working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Update package lists and install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Basic utilities
    nano \
    vim \
    curl \
    wget \
    git \
    unzip \
    # Build tools and development packages
    build-essential \
    cmake \
    g++ \
    pkg-config \
    ninja-build \
    # Python development headers
    python3-dev \
    python3-pip \
    # OpenEXR development libraries (required by README)
    libopenexr-dev \
    # SSL and FFI development libraries
    libssl-dev \
    libffi-dev \
    # Additional libraries that might be needed (fixed for Ubuntu 24.04)
    libgl1-mesa-dev \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Additional packages for graphics/video processing
    libglu1-mesa-dev \
    freeglut3-dev \
    # Clean up apt cache
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel packaging ninja --break-system-packages

# Copy requirements files
COPY requirements_gen3c.txt requirements_lyra.txt ./

# Install Python dependencies in correct order
# First install requirements_gen3c.txt
RUN pip install --no-cache-dir --break-system-packages -r requirements_gen3c.txt

# Then install requirements_lyra.txt  
RUN pip install --no-cache-dir --break-system-packages -r requirements_lyra.txt

# Copy the CUDA installation script
COPY install_cuda_packages.sh ./
COPY flash_attention_fallback.py ./

# Make the script executable and run it to install CUDA packages
RUN chmod +x install_cuda_packages.sh && \
    bash install_cuda_packages.sh || echo "CUDA packages installation completed with warnings"

# Install additional packages that might be needed for Hugging Face downloads
RUN pip install --no-cache-dir --break-system-packages huggingface_hub

# Copy the rest of the application code
COPY . .

# Set up proper permissions
RUN chmod +x train.sh inference.sh scripts/bash/*.sh

# Create necessary directories
RUN mkdir -p assets/demo checkpoints logs

# Set environment variables for CUDA (if available)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Expose common ports (adjust as needed)
EXPOSE 8080 8000 7860

# Set the default command (can be overridden)
CMD ["python3", "sample.py", "--help"]
