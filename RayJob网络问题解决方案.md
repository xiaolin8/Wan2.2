# RayJob 分布式推理网络问题解决方案

## 问题诊断

### 原始错误
```
The IPv6 network addresses of (None, 29500) cannot be retrieved
```

### 根本原因
1. **Accelerate 配置不适合 Kubernetes 环境**：
   - `compute_environment: LOCAL_MACHINE` 在容器化分布式环境中不适用
   - `num_machines: 2` 与实际 RayJob 配置不匹配

2. **网络接口配置问题**：
   - NCCL 无法找到正确的网络接口进行通信
   - Kubernetes Pod 网络环境与预期不符

3. **分布式初始化时序问题**：
   - 环境变量未正确传递给子进程
   - Master 地址和端口配置不正确

## 解决方案文件

### 1. 修复后的 Accelerate 配置
**文件**: `accelerate_config_k8s.yaml`
- 改变 `compute_environment` 为 `OCI`
- 调整 `num_machines` 和 `num_processes` 以匹配 RayJob 环境
- 优化分布式参数

### 2. 修复后的 RayJob 配置
**文件**: `wan22-rayjob-k8s-fixed.yaml`
- 添加完整的网络环境变量设置
- 使用修复后的 accelerate 配置
- 包含自动 Master 地址检测

### 3. 简化版本 RayJob
**文件**: `wan22-rayjob-final.yaml`
- 移除 accelerate，直接使用简化脚本
- 包含网络诊断功能
- 更可靠的配置

### 4. 网络诊断工具
**文件**: `network_diagnostic.py`
- 检测网络接口和连通性
- 验证 PyTorch 分布式初始化
- 生成诊断报告

### 5. Kubernetes 优化脚本
**文件**: `generate_k8s_fixed.py`
- 专门针对 RayJob 环境优化
- 更好的错误处理和日志
- 自动网络环境检测

## 部署步骤

### 步骤 1: 验证当前 RayCluster 状态
```bash
# 检查 RayCluster 状态
kubectl get raycluster -n hu

# 检查现有 Pod 状态
kubectl get pods -n hu -l ray.io/cluster=wan22-ray-cluster
```

### 步骤 2: 部署网络诊断
```bash
# 创建 ConfigMap 存储新的配置文件
kubectl create configmap accelerate-config \
  --from-file=accelerate_config_k8s.yaml \
  -n hu
```

### 步骤 3: 运行诊断 RayJob
```bash
# 部署诊断 RayJob
kubectl apply -f wan22-rayjob-final.yaml

# 监控任务状态
kubectl get rayjob -n hu -w

# 查看日志
kubectl logs -f -n hu -l ray.io/cluster=wan22-ray-cluster
```

### 步骤 4: 分析诊断结果
查看生成的 `network_diagnostic_report.json` 文件，分析网络连接状况。

## 关键网络配置

### NCCL 环境变量
```bash
export NCCL_SOCKET_IFNAME=eth0,eno1,ens,ib    # 指定网络接口
export NCCL_IB_DISABLE=0                       # 启用 InfiniBand
export NCCL_DEBUG=INFO                         # 启用调试日志
export NCCL_MIN_NRINGS=1                       # 最小 ring 数量
export NCCL_MAX_NRINGS=4                       # 最大 ring 数量
```

### PyTorch 分布式设置
```bash
export MASTER_ADDR=$(hostname -I | awk '{print $1}')  # 自动检测本机 IP
export MASTER_PORT=29500                             # 主端口
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # CUDA 内存优化
```

## 常见问题排除

### 问题 1: Master 地址无法获取
**症状**: `master_addr` 未设置或为空
**解决**: 确保 `hostname -I` 命令能返回有效的 IP 地址

### 问题 2: 网络接口不存在
**症状**: NCCL 找不到指定的网络接口
**解决**: 根据实际 Pod 网络接口调整 `NCCL_SOCKET_IFNAME`

### 问题 3: 端口被占用
**症状**: `Address already in use` 错误
**解决**: 使用不同的 `MASTER_PORT` 或确保端口释放

### 问题 4: CUDA 内存不足
**症状**: `CUDA out of memory` 错误
**解决**: 
- 启用 CPU 卸载：`--enable_cpu_offload`
- 减少批处理大小和帧数
- 使用 `float16` 精度

## 性能优化建议

1. **内存优化**:
   - 启用 xFormers: `--enable_xformers`
   - 启用 CPU 卸载: `--enable_cpu_offload`
   - 使用 `float16` 精度: `--dtype float16`

2. **网络优化**:
   - 调整 `NCCL_MIN_NRINGS` 和 `NCCL_MAX_NRINGS`
   - 启用 InfiniBand: `NCCL_IB_DISABLE=0`
   - 调整超时设置

3. **推理优化**:
   - 减少 `num_inference_steps` 平衡质量和速度
   - 调整 `guidance_scale` 参数
   - 优化 `num_frames` 设置

## 监控和日志

- 使用 `kubectl logs` 查看详细日志
- 检查 Ray Dashboard: http://<head-node-ip>:8265
- 监控 GPU 使用率: `nvidia-smi`
- 查看网络连接: `ss -tuln | grep 29500`

## 紧急回滚

如果新配置出现问题，可以回滚到原始配置：
```bash
# 删除新的 RayJob
kubectl delete rayjob wan22-rayjob-k8s-fully-fixed -n hu

# 恢复原始配置
kubectl apply -f wan22-rayjob-diffusers-parametric.yaml
```