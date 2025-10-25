# 使用包含 CUDA 开发工具的 PyTorch 镜像
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel
# 设置 CUDA 环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    libaio-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# 克隆 Wan2.2 的代码
RUN git clone https://ghfast.top/https://github.com/xiaolin8/Wan2.2.git .
# 升级 pip 并安装依赖
RUN pip install --upgrade pip setuptools wheel
# 分步安装，处理 flash-attn 的特殊性
RUN grep -v "flash_attn" requirements.txt > requirements_temp.txt && \
    pip install --no-cache-dir -r requirements_temp.txt
RUN pip install flash-attn --no-build-isolation
RUN pip install decord librosa peft
RUN pip install --no-cache-dir -r requirements_s2v.txt
RUN pip install --upgrade typing-extensions
RUN pip install --no-cache-dir -r requirements_animate.txt
RUN pip install -e .
# 设置工作目录
WORKDIR /workspace
# 默认启动一个交互式 shell
CMD ["/bin/bash"]