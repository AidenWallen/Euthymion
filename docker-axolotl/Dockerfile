FROM python:3.11-slim

# Set environment variables to avoid prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only the requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install Axolotl from GitHub
RUN pip install git+https://github.com/OpenAccess-AI-Collective/axolotl.git@main

# Install GPTQ dependencies
RUN pip install auto-gptq "gptqmodel>=1.4.3" "optimum>=1.24.0"

# Copy the rest of the project files
COPY config.yml ./
COPY data/ ./data/

# Start container in interactive shell
CMD ["bash"]
