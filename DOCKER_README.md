# Docker Setup for Lyra

This repository includes a comprehensive Dockerfile that builds upon the HuggingFace text-generation-inference base image and includes all necessary dependencies for running Lyra.

## Building the Docker Image

```bash
# Build the Docker image (this may take 15-30 minutes)
docker build -t lyra:latest .
```

## Running the Container

### 🚀 Automated Demo Mode (Recommended)

The container includes an automated entrypoint that downloads models and runs demos:

```bash
# Run complete automated demo (downloads models + runs both static & dynamic demos)
docker run -it --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/assets:/app/assets \
  -v $(pwd)/outputs:/app/outputs \
  lyra:latest
```

### 🎮 Interactive Mode

```bash
# Run demos then start interactive bash session
docker run -it --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/assets:/app/assets \
  -v $(pwd)/outputs:/app/outputs \
  lyra:latest --interactive

# Or skip demos and go straight to bash
docker run -it --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/assets:/app/assets \
  -v $(pwd)/outputs:/app/outputs \
  lyra:latest bash
```

### 📋 Available Container Options

```bash
# Show help
docker run lyra:latest --help

# Run demos then keep container alive for exec commands
docker run -d --gpus all lyra:latest

# Execute commands in running container
docker exec -it <container_id> bash
```

### 🏗️ Manual Setup (if needed)

The automated entrypoint handles everything, but if you need manual control:

```bash
# Download checkpoints manually
python scripts/download_lyra_checkpoints.py
python scripts/download_gen3c_checkpoints.py
python scripts/download_tokenizer_checkpoints.py

# Download demo data manually
huggingface-cli download nvidia/Lyra-Testing-Example --repo-type dataset --local-dir assets/demo
```

### 🎬 Manual Demo Execution

```bash
# Generate 3D scene from image
accelerate launch sample.py --config configs/demo/lyra_static.yaml

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

### Python Environment Issues

The HuggingFace TGI base image uses pipx virtual environments. Our Dockerfile has been updated to:
1. **Install packages to the correct pipx environment** (`/root/.local/share/pipx/venvs/transformers/`)
2. **Fix Python path conflicts** between system and pipx environments
3. **Ensure all pip installations target the same environment**

This resolves `ModuleNotFoundError: No module named 'torch'` issues.

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
