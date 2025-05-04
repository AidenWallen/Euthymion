#!/bin/bash

echo "==============================================="
echo "Starting Euthymion setup inside /workspace..."
echo "==============================================="

# Step 0: Ensure Docker is installed
echo "🐳 Checking for Docker..."
if ! command -v docker &> /dev/null; then
  echo "📦 Docker not found. Installing..."
  apt update && apt install -y docker.io || {
    echo "❌ Failed to install Docker"
    exit 1
  }
else
  echo "✅ Docker is already installed."
fi

# Step 1: Move into project directory
cd /workspace/euthymion/docker-axolotl || {
  echo "❌ Failed to cd into /workspace/euthymion/docker-axolotl"
  exit 1
}

# Step 2: Upgrade pip
echo "⬆️  Upgrading pip..."
export PIP_ROOT_USER_ACTION=ignore
python -m pip install --upgrade pip

# Step 3: Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
  pip install --no-cache-dir -r requirements.txt || {
    echo "❌ Failed to install requirements"
    exit 1
  }
else
  echo "⚠️  Warning: requirements.txt not found!"
fi

# Step 4: Reinstall Torch for CUDA 12.1 compatibility
echo "⚙️  Installing PyTorch for CUDA 12.1..."
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 5: Skip CLI install
echo "ℹ️  Skipping text-generation-launcher CLI install — Docker provides it internally"

# Step 6: Check Docker daemon
echo "🔍 Checking Docker daemon..."
if ! docker info > /dev/null 2>&1; then
  echo "❌ Docker daemon not running or inaccessible."
  echo "💡 Make sure Docker is enabled in this environment (e.g., RunPod A100 or L40 pods)."
  exit 1
else
  echo "✅ Docker daemon is running."
fi

# Step 7: Launch Hugging Face TGI container
echo "🚀 Launching Euthymion with Hugging Face TGI Docker..."
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v /workspace/euthymion/docker-axolotl/out:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --revision main \
  --trust-remote-code \
  --max-input-length 4096 \
  --max-total-tokens 8192 \
  --quantize bf16 \
  --dtype float16 \
  --rope-scaling linear \
  --lora /data
