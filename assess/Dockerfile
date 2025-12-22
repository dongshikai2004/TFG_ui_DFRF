FROM python:3.9-slim

 RUN sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
     sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list
# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /assess

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple