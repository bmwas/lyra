#!/bin/bash
set -e

echo "=== Lyra Docker Container Starting ==="
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo ""

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
