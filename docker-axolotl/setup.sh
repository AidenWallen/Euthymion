#!/bin/bash

echo "==============================================="
echo "Starting Euthymion setup inside /workspace..."
echo "==============================================="

# Step 1: Move into project directory
cd /workspace/euthymion/docker-axolotl || { echo "Failed to cd into /workspace/euthymion/docker-axolotl"; exit 1; }

# Step 2: Reinstall Python libraries
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Step 3: Force reinstall correct torch and torchvision for CUDA 12.1
echo "Fixing Torch and Torchvision versions for CUDA 12.1..."
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install exllamav2 if it exists
if [ -d "./exllamav2" ]; then
  echo "Installing exllamav2..."
  pip install ./exllamav2 || echo "Warning: exllamav2 install failed. Continuing..."
else
  echo "Warning: exllamav2 directory not found. Skipping."
fi

# Step 5: Launch the app
echo "Launching Euthymion..."

