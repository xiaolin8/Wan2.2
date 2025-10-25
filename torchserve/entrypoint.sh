#!/bin/bash
# /usr/local/bin/entrypoint.sh

set -ex

# --- 1. 设置 NCCL 环境变量 (根据实际情况调整 HCA ID 和接口) ---
export NCCL_DEBUG="INFO"
export NCCL_DEBUG_SUBSYS="ALL"
export NCCL_IB_DISABLE="0" # 确保不禁用 InfiniBand
export NCCL_IB_HCA="mlx5_0" # 替换为你实际的 HCA ID (例如 mlx5_0, mlx5_3, mlx5_4, mlx5_7 等中的一个或多个)
export NCCL_SOCKET_IFNAME="ib0" # 替换为 Pod 内部 RDMA 接口的名称 (例如 ib0 或 net1)

# --- 2. 基于 StatefulSet 的 DNS 模式发现分布式参数 ---
# StatefulSet 名称 (在 StatefulSet YAML 中定义)
STATEFULSET_NAME="my-distributed-model"
# K8s Service 名称 ( headless service)
SERVICE_NAME="wan22-distributed-svc"
# K8s 命名空间
NAMESPACE="default"
# Master Pod 的端口
MASTER_PORT="29500" # 用于 torch.distributed 的通信端口
# 模型存储目录
MODEL_STORE="/home/model-server/model-store"
# 你的模型 .mar 文件名
MODEL_ARCHIVE="name_of_your_model.mar"
# 处理器文件
HANDLER_FILE="/home/model-server/handler.py"

# 获取当前 Pod 的索引 (例如 my-distributed-model-0 -> 0, my-distributed-model-1 -> 1)
# HOSTNAME = my-distributed-model-0, my-distributed-model-1 等等
POD_ID=$(echo $HOSTNAME | rev | cut -d'-' -f1 | rev)
RANK=${POD_ID}

# 获取所有 Pod 的数量 (即 WORLD_SIZE，这里假设是 2)
WORLD_SIZE="2" # 替换为你实际的 StatefuSet replicas 数量

# Master Pod 的 DNS 名称
MASTER_ADDR="${STATEFULSET_NAME}-0.${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local"

# 每节点 GPU 数量 (根据你的 YAML 配置)
NPROC_PER_NODE="4"

echo "--- Distributed Env Configuration ---"
echo "HOSTNAME: $HOSTNAME"
echo "POD_ID: $POD_ID"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "--- Starting TorchServe ---"

# --- 3. 设置 TorchServe 环境变量 ---
# TorchServe 需要知道这些分布式参数来传递给 handler.py
# TorchServe 启动时会自动暴露 handler.py 中的 initialize, inference 等方法

# --- 4. 启动 TorchServe ---
# TorchServe 会在内部启动多个 worker (由 --ts-config 控制，或者默认一个)
# 这里的 entrypoint.sh 启动的 TorchServe 进程，本身只是 host，handler.py 才是执行者
# handler.py 内部会根据 RANK 进一步 init_process_group
torchserve --start \
           --model-store ${MODEL_STORE} \
           --models ${MODEL_ARCHIVE} \
           --ts-config ${TS_CONFIG_FILE} \
           --nproc_per_node ${NPROC_PER_NODE} \
           # --handler ${HANDLER_FILE} # 如果 handler 不在 .mar 中，可以指定

# 为了调试，可以打印 model-store 的内容
# ls -l ${MODEL_STORE}
# cat ${TS_CONFIG_FILE}

# 让容器保持运行
tail -f /dev/null