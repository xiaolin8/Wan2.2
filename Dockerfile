FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    libaio-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://ghfast.top/https://github.com/Wan-Video/Wan2.2.git .
RUN pip install --upgrade pip setuptools wheel
RUN grep -v "flash_attn" requirements.txt > requirements_temp.txt && \
    pip install --no-cache-dir -r requirements_temp.txt
RUN pip install flash-attn --no-build-isolation
RUN pip install decord librosa peft
RUN pip install --no-cache-dir -r requirements_s2v.txt
RUN pip install --upgrade typing-extensions
RUN pip install --no-cache-dir -r requirements_animate.txt
RUN pip install -e .
WORKDIR /workspace
CMD ["/bin/bash"]