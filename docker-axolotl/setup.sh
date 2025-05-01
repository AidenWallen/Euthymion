#!/bin/bash

echo "==============================================="
echo "Starting Euthymion setup inside /workspace..."
echo "==============================================="

# Step 1: Move into project directory
cd /workspace/euthymion/docker-axolotl || {
  echo "❌ Failed to cd into /workspace/euthymion/docker-axolotl"
  exit 1
}

# Step 2: Reinstall Python libraries
echo "📦 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt || {
  echo "❌ Failed to install requirements.txt"
  exit 1
}

# Step 3: Reinstall correct Torch & Torchvision for CUDA 12.1
echo "⚙️ Fixing Torch and Torchvision versions for CUDA 12.1..."
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install exllamav2 if present
if [ -d "./exllamav2" ]; then
  echo "📥 Installing exllamav2..."
  pip install --no-cache-dir ./exllamav2 || echo "⚠️ Warning: exllamav2 install failed. Continuing..."
else
  echo "⚠️ Warning: exllamav2 directory not found. Skipping."
fi

# Step 5: Done
echo "✅ Euthymion setup complete!"
