# Use CUDA base with cuDNN support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Required for noninteractive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"
ENV CUDA_HOME=/usr/local/cuda

# Install system deps including Docker for DinD
RUN apt-get update && apt-get install -y \
    git git-lfs curl wget ca-certificates \
    software-properties-common build-essential \
    ninja-build docker.io \
    python3.11 python3.11-dev python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default and install pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Set working directory
WORKDIR /workspace/euthymion/docker-axolotl

# Copy code
COPY . .

# Install PyTorch (CUDA 12.1) and other dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install ./exllamav2

# (Optional) Check that Docker CLI works inside
RUN docker --version || echo "⚠️ Docker CLI not working — daemon must run externally."

# Default command (you'll override this when running with --entrypoint bash)
CMD ["bash"]
