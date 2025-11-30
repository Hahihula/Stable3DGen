# Use NVIDIA CUDA 12.4 development image based on Ubuntu 22.04
ARG CUDA_VERSION=12.4.0-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_VERSION}

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  MODEL_DIR=/stable3dgen/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
  git \
  python3.10 \
  python3.10-venv \
  python3-pip \
  curl \
  libgl1 \
  libglib2.0-0 libsm6 libxrender1 libxext6 \
  && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create working directory
WORKDIR /stable3dgen

ARG CACHEBUST=1

# Clone Stable3DGen repository
RUN git clone --recursive https://github.com/Hahihula/Stable3DGen.git .

# Upgrade pip/setuptools/wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch + TorchVision for CUDA 12.4
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# Install spconv and xformers for CUDA 12.4
RUN pip install spconv-cu124==2.3.7 xformers==0.0.27.post2

# Install other dependencies
RUN pip install -r requirements.txt

# Create volume for models
VOLUME ["/stable3dgen/models"]
VOLUME ["/root/.cache"]

# Expose port for Gradio demo
EXPOSE 7860

# Default command to run the demo
CMD ["python", "app.py", "--port", "7860", "--model-dir", "/stable3dgen/models"]
