#!/bin/bash
set -e

# Check if custom command is provided - if so, execute directly without automation
if [ $# -gt 0 ]; then
    case "$1" in
        --interactive|-i)
            # Interactive mode - run automation then bash
            ;;
        --help|-h)
            echo "Lyra Docker Container Usage:"
            echo "  docker run -it --gpus all lyra:latest           # Run automated demos"
            echo "  docker run -it --gpus all lyra:latest -i        # Run demos then interactive bash"
            echo "  docker run -it --gpus all lyra:latest bash      # Skip demos, direct bash"
            echo "  docker run -it --gpus all lyra:latest <cmd>     # Execute custom command"
            echo "  docker run -it --gpus all lyra:latest --help    # Show this help"
            exit 0
            ;;
        bash|sh|/bin/bash|/bin/sh)
            # Direct shell access - skip automation
            echo "=== Direct Shell Access ==="
            exec "$@"
            ;;
        python*|pip*|accelerate|huggingface-cli)
            # Direct command execution - skip automation  
            echo "=== Executing Custom Command: $@ ==="
            cd /app
            exec "$@"
            ;;
        *)
            # Other custom commands
            echo "=== Executing Custom Command: $@ ==="
            cd /app
            exec "$@"
            ;;
    esac
fi

echo "=== Lyra Docker Container Starting ==="
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"

# Debug Python packages
echo "=== Python Environment Debug ==="
echo "Python sys.path:"
python -c "import sys; [print(f'  {p}') for p in sys.path]"
echo ""

echo "Checking critical packages:"
python -c "
try:
    import torch
    print(f'✓ Torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except Exception as e:
    print(f'❌ Torch: {e}')
    
try:
    import transformers
    print(f'✓ Transformers: {transformers.__version__}')
except Exception as e:
    print(f'❌ Transformers: {e}')
    
try:
    import accelerate
    print(f'✓ Accelerate: {accelerate.__version__}')
except Exception as e:
    print(f'❌ Accelerate: {e}')
"

# Check if we need to install packages
echo ""
echo "Installed pip packages (relevant):"
pip list | grep -E "(torch|transformers|accelerate|diffusers)" || echo "No relevant packages found"
echo ""

# Check if critical packages are missing and attempt to reinstall
echo "=== Package Installation Check ==="
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ Critical packages missing! Attempting to reinstall..."
    echo "This might happen if there was an issue during Docker build."
    echo ""
    
    echo "Reinstalling requirements_gen3c.txt to pipx transformers environment..."
    /root/.local/share/pipx/venvs/transformers/bin/python -m pip install --no-cache-dir -r /app/requirements_gen3c.txt || echo "❌ Failed to install requirements_gen3c.txt"
    
    echo "Reinstalling requirements_lyra.txt to pipx transformers environment..."
    /root/.local/share/pipx/venvs/transformers/bin/python -m pip install --no-cache-dir -r /app/requirements_lyra.txt || echo "❌ Failed to install requirements_lyra.txt"
    
    echo "Verifying torch installation after reinstall..."
    python -c "import torch; print(f'✓ Torch reinstalled: {torch.__version__}')" || echo "❌ Torch still not available"
    echo ""
fi

# Navigate to the Lyra repository directory
cd /app

# Function to check if directory exists and has content
check_directory() {
    local dir="$1"
    local description="$2"
    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        echo "✓ $description already exists and is not empty: $dir"
        return 0
    else
        echo "⚠ $description not found or empty: $dir"
        return 1
    fi
}

# Function to download with retry
download_with_retry() {
    local cmd="$1"
    local description="$2"
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt/$max_attempts: $description"
        if eval "$cmd"; then
            echo "✓ $description completed successfully"
            return 0
        else
            echo "❌ $description failed (attempt $attempt/$max_attempts)"
            if [ $attempt -eq $max_attempts ]; then
                echo "⚠ $description failed after $max_attempts attempts, continuing..."
                return 1
            fi
            ((attempt++))
            sleep 5
        fi
    done
}

echo "=== Step 1: Downloading Pre-trained Checkpoints ==="

# Check if torch is available before proceeding with downloads
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ Cannot proceed with checkpoint downloads - torch is not available!"
    echo "⚠️  The Docker build may have failed or packages are corrupted."
    echo ""
    echo "Suggested fixes:"
    echo "1. Rebuild the Docker image: docker build -t lyra:latest ."
    echo "2. Check for build errors in the Docker build output"
    echo "3. Try running container without custom commands: docker run -it --gpus all lyra:latest"
    echo ""
    echo "Skipping automated setup and dropping to shell for manual debugging..."
    
    if [ "$1" = "--interactive" ] || [ "$1" = "-i" ]; then
        exec bash
    elif [ $# -gt 0 ]; then
        exec "$@"
    else
        echo "Container will remain running for debugging. Use: docker exec -it <container_id> bash"
        tail -f /dev/null
    fi
    exit 0
fi

# Check if checkpoints already exist
if ! check_directory "checkpoints" "Checkpoints directory"; then
    echo "Downloading Lyra checkpoints..."
    download_with_retry "python scripts/download_lyra_checkpoints.py" "Lyra checkpoints download"
    
    echo "Downloading Gen3C checkpoints..."
    download_with_retry "python scripts/download_gen3c_checkpoints.py" "Gen3C checkpoints download"
    
    echo "Downloading tokenizer checkpoints..."
    download_with_retry "python scripts/download_tokenizer_checkpoints.py" "Tokenizer checkpoints download"
else
    echo "Checkpoints already exist, skipping download."
fi

echo ""
echo "=== Step 2: Downloading Demo Data ==="

# Check if demo data already exists
if ! check_directory "assets/demo" "Demo data directory"; then
    echo "Creating assets directory..."
    mkdir -p assets/demo
    
    echo "Downloading demo data from Hugging Face..."
    download_with_retry "huggingface-cli download nvidia/Lyra-Testing-Example --repo-type dataset --local-dir assets/demo" "Demo data download"
else
    echo "Demo data already exists, skipping download."
fi

echo ""
echo "=== Step 3: Running Demos ==="

# Set up demo environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/app:$PYTHONPATH

echo "🎬 Running Static Scene Generation Demo (Single Image to 3D)..."
if download_with_retry "accelerate launch sample.py --config configs/demo/lyra_static.yaml" "Static scene generation demo"; then
    echo "✓ Static demo completed successfully!"
else
    echo "⚠ Static demo encountered issues but container will continue running"
fi

echo ""
echo "🎬 Running Dynamic Scene Generation Demo (Video to 4D)..."
if download_with_retry "accelerate launch sample.py --config configs/demo/lyra_dynamic.yaml" "Dynamic scene generation demo"; then
    echo "✓ Dynamic demo completed successfully!"
else
    echo "⚠ Dynamic demo encountered issues but container will continue running"
fi

echo ""
echo "=== All Demos Completed ==="
echo "✓ Container setup and demos finished!"
echo "✓ Container will remain running for interactive use"
echo ""
echo "You can now:"
echo "  - Run custom commands: docker exec -it <container_id> bash"
echo "  - Generate your own scenes with different configs"
echo "  - Explore the generated outputs in the mounted volumes"
echo ""
echo "Available commands inside container:"
echo "  - accelerate launch sample.py --config <config_file>"
echo "  - python scripts/download_*.py (to re-download models)"
echo "  - bash train.sh (for training)"
echo "  - bash inference.sh (for inference)"
echo ""

# Parse command line arguments for interactive mode
if [ "$1" = "--interactive" ] || [ "$1" = "-i" ]; then
    echo "=== Starting Interactive Bash Session ==="
    exec bash
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Lyra Docker Container Usage:"
    echo "  docker run -it --gpus all lyra:latest           # Run demos then keep container alive"
    echo "  docker run -it --gpus all lyra:latest -i        # Run demos then start interactive bash"
    echo "  docker run -it --gpus all lyra:latest bash      # Skip demos, start bash directly"
    echo "  docker run -it --gpus all lyra:latest --help    # Show this help"
    exit 0
elif [ $# -gt 0 ]; then
    echo "=== Executing Custom Command: $@ ==="
    exec "$@"
else
    echo "=== Keeping Container Alive ==="
    echo "Use 'docker exec -it <container_id> bash' to interact with this container"
    echo "Press Ctrl+C to stop the container"
    
    # Keep container alive and responsive
    tail -f /dev/null
fi
