#!/bin/bash

echo "==============================================="
echo "Starting Euthymion setup inside /workspace..."
echo "==============================================="

# Step 1: Move into project directory
cd /workspace/euthymion/docker-axolotl || {
  echo "❌ Failed to cd into /workspace/euthymion/docker-axolotl"
  exit 1
}

# Step 2: Upgrade pip to avoid warnings
echo "⬆️  Upgrading pip..."
export PIP_ROOT_USER_ACTION=ignore
python -m pip install --upgrade pip

# Step 3: Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
  pip install --no-cache-dir -r requirements.txt || {
    echo "❌ Failed to install requirements.txt"
    exit 1
  }
else
  echo "⚠️  Warning: requirements.txt not found!"
fi

# Step 4: Skip CLI install, explain why
echo "ℹ️  Skipping text-generation-launcher CLI install — not pip-installable; Docker provides it internally"

# Step 5: Reinstall Torch for CUDA 12.1
echo "⚙️  Reinstalling Torch and Torchvision for CUDA 12.1 compatibility..."
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 6: Launch TGI via Docker
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
