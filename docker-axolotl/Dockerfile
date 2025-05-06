# Use CUDA base with cuDNN support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Required for noninteractive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl wget ca-certificates \
    software-properties-common build-essential \
    ninja-build \
    python3.11 python3.11-dev python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default and install pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Set working directory
WORKDIR /workspace

# Entrypoint script with Git install + clone + setup
CMD ["/bin/bash", "-c", "\
    echo '🚀 Euthymion container started — bootstrapping...'; \
    if ! command -v git &> /dev/null; then \
        echo '🛠 Git not found — attempting to install...'; \
        apt-get update && apt-get install -y git git-lfs || { echo '❌ Failed to install Git'; exit 1; }; \
    fi; \
    if [ ! -d '/workspace/euthymion/.git' ]; then \
        echo '📥 Cloning Euthymion repo...'; \
        git clone https://github.com/xoaden/euthymion.git /workspace/euthymion || { echo '❌ Git clone failed'; exit 1; }; \
    else \
        echo '🔄 Repo already present — pulling latest changes...'; \
        cd /workspace/euthymion && git pull || { echo '❌ Git pull failed'; exit 1; }; \
    fi; \
    cd /workspace/euthymion/docker-axolotl && \
    chmod +x setup.sh && \
    ./setup.sh && \
    echo '✅ Setup complete. Container idle.' && \
    tail -f /dev/null"]
