# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@sha256:8f9dd0d09d3ad3900357a1cf7f887888b5b74056636cd6ef03c160c3cd4b1d95

ENV PYTHON_VERSION=3.10 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONPATH=/DFRF

RUN echo "deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb-src https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb-src https://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list

RUN apt-get update && rm -rf /var/lib/apt/lists/*

RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    gcc \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    python3-tk \
    tk-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    wget \
    curl \
    unzip \
    portaudio19-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python3 -m pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /DFRF

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install tensorflow-gpu==2.8.0 -i https://mirrors.aliyun.com/pypi/simple

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118



COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.7.1cu118

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple

# COPY . .