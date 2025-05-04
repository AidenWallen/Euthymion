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

# Step 3: Install TGI CLI from source
echo "🧠 Installing Text Generation Inference CLI manually..."
TMP_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/huggingface/text-generation-inference.git "$TMP_DIR" || {
  echo "❌ Failed to clone TGI repo"
  exit 1
}
if [ -f "$TMP_DIR/cli/pyproject.toml" ]; then
  pip install --no-cache-dir "$TMP_DIR/cli" || {
    echo "❌ Failed to install CLI from source"
    exit 1
  }
else
  echo "❌ CLI project structure not found. Cannot install."
  exit 1
fi
rm -rf "$TMP_DIR"

# Step 3.1: Check for text-generation-launcher
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

# Step 4: Reinstall Torch for CUDA 12.1
echo "⚙️ Fixing Torch and Torchvision for CUDA 12.1..."
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 5: Launch server
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
