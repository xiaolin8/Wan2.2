FROM 172.31.0.182/system_containers/wan22:pytorch

# 安装 TorchServe
RUN pip install torchserve torch-model-archiver accelerate

# 安装网络工具 (用于调试，如 ip a)
RUN apt-get update && apt-get install -y iproute2 net-tools && rm -rf /var/lib/apt/lists/*

# 创建模型存储目录和配置文件
ENV MODEL_STORE="/home/model-server/model-store"
ENV TS_CONFIG_FILE="/home/model-server/config.properties"
RUN mkdir -p ${MODEL_STORE}

# 复制你的 handler.py 和 entrypoint.sh
COPY handler.py /home/model-server/handler.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# 如果模型权重较大，建议使用多阶段构建或从外部挂载
# COPY model_weights.pth ${MODEL_STORE}/

# 设置入口点
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]