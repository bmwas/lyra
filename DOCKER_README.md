# Docker Setup for Lyra

This repository includes a comprehensive Dockerfile that builds upon the HuggingFace text-generation-inference base image and includes all necessary dependencies for running Lyra.

## Building the Docker Image

```bash
# Build the Docker image (this may take 15-30 minutes)
docker build -t lyra:latest .
```

## Running the Container

### Basic Usage

```bash
# Run the container interactively
docker run -it --gpus all -v $(pwd)/assets:/app/assets lyra:latest bash
```

### With Volume Mounts for Persistent Data

```bash
# Run with mounted directories for checkpoints and outputs
docker run -it --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/assets:/app/assets \
  -v $(pwd)/outputs:/app/outputs \
  lyra:latest bash
```

### Download Required Models

Once inside the container, download the pre-trained models:

```bash
# Download checkpoints
python scripts/download_lyra_checkpoints.py
python scripts/download_gen3c_checkpoints.py
python scripts/download_tokenizer_checkpoints.py

# Download demo data
huggingface-cli download nvidia/Lyra-Testing-Example --repo-type dataset --local-dir assets/demo
```

### Running Lyra Demos

#### Static Scene Generation (Single Image to 3D)

```bash
# Generate 3D scene from image
accelerate launch sample.py --config configs/demo/lyra_static.yaml
```

#### Dynamic Scene Generation (Video to 4D)

```bash
# Generate dynamic 3D scene from video  
accelerate launch sample.py --config configs/demo/lyra_dynamic.yaml
```

## Important Notes

1. **GPU Requirements**: Lyra requires NVIDIA GPUs. The container needs `--gpus all` flag.

2. **Memory Requirements**: Ensure your system has sufficient GPU memory (tested on H100/A100).

3. **CUDA Packages**: The Dockerfile attempts to install CUDA packages (flash_attn, etc.). If compilation fails, CPU fallbacks are available.

4. **Model Downloads**: Pre-trained models are large and should be downloaded separately rather than included in the Docker image.

5. **Data Persistence**: Use volume mounts to persist checkpoints, outputs, and input data.

## Troubleshooting

### Package Installation Issues

If you encounter package installation errors during build:

```bash
# The Dockerfile has been updated to fix Ubuntu 24.04 compatibility issues:
# 1. libgl1-mesa-glx was replaced with libgl1-mesa-dev + libgl1-mesa-dri
# 2. Added --break-system-packages flag to pip commands (safe in Docker containers)
```

### Externally Managed Environment Error

The base image uses PEP 668 externally-managed-environment protection. The Dockerfile uses `--break-system-packages` flag which is safe and appropriate for Docker containers.

### CUDA Package Installation Issues

If CUDA packages fail to compile during Docker build:

```bash
# Build without CUDA packages first
docker build --build-arg SKIP_CUDA=1 -t lyra:cpu .

# Then install CUDA packages inside the running container
docker run -it --gpus all lyra:cpu bash
# Inside container:
bash install_cuda_packages.sh
```

### Memory Issues

For systems with limited GPU memory, use the memory optimization flags:

```bash
# Add these flags to your commands inside the container
--offload_diffusion_transformer \
--offload_tokenizer \
--offload_text_encoder_model \
--offload_prompt_upsampler \
--offload_guardrail_models \
--disable_guardrail \
--disable_prompt_encoder
```

## Development

For development work, mount the source code:

```bash
docker run -it --gpus all \
  -v $(pwd):/app \
  -v $(pwd)/checkpoints:/app/checkpoints \
  lyra:latest bash
```

This allows you to edit code on your host system while running it inside the container.
