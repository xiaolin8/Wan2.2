# 多机多卡（Multi-Node Multi-GPU）训练指南

本文档详细说明了如何使用 `torchrun` 和 `docker` 在多个物理节点上进行分布式模型训练。

我们将以两台机器为例进行演示：
- **主节点 (Master Node)**: `bjdb-h20-node-080` (IP: `172.31.0.80`)
- **工作节点 (Worker Node)**: `bjdb-h20-node-082` (IP: `172.31.0.82`)

---

## 第一章：在所有节点上启动 Docker 容器

请在 **每一台** 参与训练的机器（`bjdb-h20-node-080` 和 `bjdb-h20-node-082`）上执行以下命令来启动容器：

```bash
docker run -d --gpus all --net=host --name wan22 \
     -v /data/Wan-AI/Wan2.2-T2V-A14B:/Wan2.2-T2V-A14B \
     -v $(pwd)/output:/workspace/output \
     172.31.0.182/system_containers/wan2-2:1014 \
     tail -f /dev/null
```

---

## 第二章：执行前的清理与准备

`torch.distributed.DistBackendError` 和 `Duplicate GPU detected` 错误通常表示有之前失败的训练进程仍残留在系统中，占用了 GPU 资源。

在 **每一次** 尝试运行新的训练任务之前，请务必在 **所有节点** (`bjdb-h20-node-080` 和 `bjdb-h20-node-082`) 的容器内执行以下命令，以确保清理掉所有旧的、可能冲突的进程：

```bash
# 强制杀死所有与 generate.py 相关的进程
pkill -f generate.py
```

你也可以使用 `ps` 命令来检查是否还有相关进程存在：
```bash
ps aux | grep generate.py
```
确保在两个节点上都看不到任何旧的 `torchrun` 或 `generate.py` 进程后，再继续执行下一步。

---

## 第三章：在各节点上执行分布式训练命令

**重要配置:** 根据模型架构和报错日志，我们确认了两个关键限制：

1.  `--ulysses_size` 必须等于总进程数 (`nnodes` * `nproc_per_node`)。
2.  模型的注意力头数 `num_heads` (为 40) 必须能被 `--ulysses_size` 整除。

为了满足条件，我们选择 `10` 作为总进程数 (`2` 个节点 * `5` 个 GPU/节点)，因为 10 能整除 40。

### 3.1 在主节点 (`bjdb-h20-node-080`) 上执行

```bash
nohup torchrun \
    --nproc_per_node=5 \
    --nnodes=2 \
    --node_rank=0 \
    --rdzv_id=wan22-job-$(date '+%Y%m%d') \
    --rdzv_backend=c10d \
    --rdzv_endpoint="172.31.0.80:29500" \
    generate.py \
    --task t2v-A14B \
    --size 832*480 \
    --ckpt_dir /Wan2.2-T2V-A14B \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 10 \
    --sample_guide_scale 10 \
    --prompt "A cinematic video. An astronaut in a white spacesuit walks cautiously on the red, dusty surface of Moon. The camera follows the astronaut from behind in a tracking shot. The astronaut stops and looks up in surprise at a giant, dark rock. The camera tilts up to reveal the word "tenxcloud" clearly engraved on the rock's surface. The scene is realistic, with a desolate Martian landscape and a faint sun in the background, creating a sense of mystery and discovery." \
    --save_file /workspace/output/$(date '+%Y%m%d%H%M%S').mp4  > /workspace/output/generate-tenxcloud.log 2>&1 &
```

### 3.2 在工作节点 (`bjdb-h20-node-082`) 上执行

```bash
nohup torchrun \
    --nproc_per_node=5 \
    --nnodes=2 \
    --node_rank=1 \
    --rdzv_id=wan22-job-$(date '+%Y%m%d') \
    --rdzv_backend=c10d \
    --rdzv_endpoint="172.31.0.80:29500" \
    generate.py \
    --task t2v-A14B \
    --size 832*480 \
    --ckpt_dir /Wan2.2-T2V-A14B \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 10 \
    --sample_guide_scale 10 \
    --prompt "A cinematic video. An astronaut in a white spacesuit walks cautiously on the red, dusty surface of Moon. The camera follows the astronaut from behind in a tracking shot. The astronaut stops and looks up in surprise at a giant, dark rock. The camera tilts up to reveal the word "tenxcloud" clearly engraved on the rock's surface. The scene is realistic, with a desolate Martian landscape and a faint sun in the background, creating a sense of mystery and discovery." \
    --save_file /workspace/output/$(date '+%Y%m%d%H%M%S').mp4  > /workspace/output/generate-tenxcloud.log 2>&1 &
```

---

## 第四章：Kubernetes (K8s) 部署方案与实践

从手动管理 Docker 容器和 `torchrun` 命令，到使用 Kubernetes 进行自动化部署，是实现 MLOps 和提高生产效率的关键一步。本章将深入探讨在 K8s 上部署此类分布式训练任务的两种主流方案：`PyTorchJob` 和 `RayJob`。

### 4.1 方案一：使用 PyTorchJob 进行专项训练

`PyTorchJob` 是 Kubeflow 项目的一部分，它提供了一个声明式接口，专门用于在 K8s 上运行 PyTorch 分布式训练。

- **核心理念**：为 PyTorch DDP 训练量身打造，将一次训练任务抽象为一个 K8s 资源，任务结束，资源销毁。
- **实践步骤**：
  1. **前提**: K8s 集群中已安装 [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/pytorch/)。
  2. **创建 YAML**: 使用我们之前创建的 `wan22-pytorchjob.yaml` 文件。
  3. **部署与监控**: `kubectl apply -f wan22-pytorchjob.yaml`
- **优点**: 简单直接，代码零侵入，与 Kubeflow 生态整合度高。
- **缺点**: 功能单一，灵活性低，主要用于静态资源分配。

### 4.2 方案二：使用 RayJob 构建通用计算平台

[Ray](https://www.ray.io/) 是一个通用的分布式计算框架，它允许你先构建一个可复用的 `RayCluster`，再通过 `RayJob` 提交任务。

- **核心理念**：构建一个覆盖数据处理、训练、服务等环节的端到端统一平台。
- **实践步骤**：
  1. **前提**: K8s 集群中已安装 [KubeRay Operator](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html)。
  2. **部署 `RayCluster`**: 定义并部署一个 YAML 文件来创建常驻的 Ray 集群。
  3. **适配训练代码**: 需要将代码逻辑适配到 Ray Train API。
  4. **提交 `RayJob`**: 向 Ray 集群提交你的训练脚本。
- **优点**: 功能强大，原生支持弹性伸缩，是构建复杂 ML 平台的理想选择。
- **缺点**: 学习曲线更陡峭，需要对代码进行适配和修改。

### 4.3 选型建议

- **对于当前任务**: 如果目标只是运行独立的训练脚本，**`PyTorchJob` 是最直接、最高效的选择**。
- **对于长远规划**: 如果计划构建一个包含数据处理、训练、推理等多个阶段的统一平台，**投资学习和使用 Ray 是更具战略意义的选择**。

---

