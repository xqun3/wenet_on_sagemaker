FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    wget \
    libopenmpi-dev \
    python3-dev \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN conda create -n wenet python=3.10 -y && \
    conda clean -ya

# Set conda environment path
ENV PATH /opt/conda/envs/wenet/bin:$PATH
SHELL ["conda", "run", "-n", "wenet", "/bin/bash", "-c"]

# Install conda and pip packages
# RUN conda install -y -c conda-forge sox openssl certifi && \
RUN conda install -y -c conda-forge sox && \
    conda clean -ya && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.2.2+cu121 torchaudio==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Copy source code
COPY ./wenet_src /wenet_src

# Set working directory and install WeNet
WORKDIR /wenet_src
RUN pip install -e .

# Install SageMaker specific packages
RUN pip install --no-cache-dir sagemaker-training

# Set default command to activate conda environment
CMD ["conda", "run", "-n", "wenet", "/bin/bash"]
