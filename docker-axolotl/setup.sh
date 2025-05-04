#!/bin/bash

echo "==============================================="
echo "Starting Euthymion setup inside /workspace..."
echo "==============================================="

# Step 1: Move into project directory
cd /workspace/euthymion/docker-axolotl || {
  echo "❌ Failed to cd into /workspace/euthymion/docker-axolotl"
  exit 1
}

# Step 2: Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
  pip install --no-cache-dir -r requirements.txt || {
    echo "❌ Failed to install requirements.txt"
    exit 1
  }
else
  echo "⚠️ Warning: requirements.txt not found!"
fi

# Step 3: Install TGI CLI
echo "🧠 Installing Text Generation Inference CLI..."
pip install --no-cache-dir text-generation || {
  echo "❌ Failed to install text-generation CLI"
  exit 1
}

# Step 3.1: Check for text-generation-launcher in PATH or fix it
if ! command -v text-generation-launcher &> /dev/null; then
  echo "⚠️ text-generation-launcher not in PATH. Attempting to locate..."
  TGI_BIN=$(find / -type f -name "text-generation-launcher" 2>/dev/null | head -n 1)
  if [ -n "$TGI_BIN" ]; then
    echo "🔧 Found launcher at $TGI_BIN. Adding to PATH."
    export PATH="$PATH:$(dirname "$TGI_BIN")"
  else
    echo "❌ text-generation-launcher still not found. Aborting."
    exit 1
  fi
fi

# Step 4: Install Torch/CUDA bindings
echo "⚙️ Fixing Torch and Torchvision for CUDA 12.1..."
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 5: Install exllamav2 if present
if [ -d "./exllamav2" ]; then
  echo "📥 Installing exllamav2..."
  pip install --no-cache-dir ./exllamav2 || echo "⚠️ Warning: exllamav2 install failed. Continuing..."
else
  echo "⚠️ Warning: exllamav2 directory not found. Skipping."
fi

# Step 6: Launch TGI server
echo "🚀 Launching Euthymion with TGI..."
text-generation-launcher \
  --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --trust-remote-code \
  --revision main \
  --max-input-length 4096 \
  --max-total-tokens 8192 \
  --quantize bf16 \
  --dtype float16 \
  --rope-scaling linear \
  --lora /workspace/euthymion/docker-axolotl/out
