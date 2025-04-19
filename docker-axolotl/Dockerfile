# Use full CUDA development image (includes nvcc)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"
ENV CUDA_HOME=/usr/local/cuda

# Install base packages and Python 3.11
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    ca-certificates \
    software-properties-common \
    build-essential \
    ninja-build && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default and install pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Set working directory
WORKDIR /workspace

# Copy model and code files
COPY Mixtral-8x7B-v0.1-GPTQ /workspace/Mixtral-8x7B-v0.1-GPTQ
COPY exllamav2 /exllamav2
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install /exllamav2 --no-cache-dir

# Debug: confirm app.py exists
RUN echo "Checking for app.py..." && ls -la /workspace && cat /workspace/app.py || echo "app.py NOT FOUND"

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
