# 多机多卡（Multi-Node Multi-GPU）训练指南

本文档详细说明了如何使用 `torchrun` 和 `docker` 在多个物理节点上进行分布式模型训练。

我们将以两台机器为例进行演示：
- **主节点 (Master Node)**: `bjdb-h20-node-080` (IP: `172.31.0.80`)
- **工作节点 (Worker Node)**: `bjdb-h20-node-082` (IP: `172.31.0.82`)

---

## 第一步：在所有节点上启动 Docker 容器

为了确保容器之间能够顺畅通信，我们使用 `--net=host` 模式。这使得容器可以直接共享宿主机的网络，简化了配置。

请在 **每一台** 参与训练的机器（`bjdb-h20-node-080` 和 `bjdb-h20-node-082`）上执行以下命令来启动容器：

```bash
docker run -d --gpus all --net=host --name wan22 \
     -v /data/Wan-AI/Wan2.2-T2V-A14B:/Wan2.2-T2V-A14B \
     -v $(pwd)/output:/workspace/output \
     172.31.0.182/system_containers/wan2-2:1014 \
     tail -f /dev/null
```
**注意:**
- `--net=host`: 关键参数，用于实现跨主机通信。
- 请确保所有节点上的 `/data/Wan-AI/Wan2.2-T2V-A14B` 路径和数据都存在且一致。

---

## 第二步：在各节点上执行分布式训练命令

启动容器后，你需要分别登录到每个节点的容器中（或使用 `docker exec`）来执行 `torchrun` 命令。

**重要更正:** 根据报错日志，我们有两个限制条件：
1.  `--ulysses_size` 必须等于总进程数 (`nnodes` * `nproc_per_node`)。
2.  模型的注意力头数 `num_heads` (在此模型中为 40) 必须能被 `--ulysses_size` 整除。

我们之前的尝试中，总进程数是 `2 * 8 = 16`。但 40 不能被 16 整除，因此报错。
为了满足条件，我们需要选择一个能整除 40 的总进程数。在拥有 2 台机器、每台最多 8 个 GPU 的情况下，最优选择是使用 5 个 GPU，构成 `2 * 5 = 10` 的总进程数，因为 10 能整除 40。
因此，我们将 `--nproc_per_node` 修改为 `5`，并将 `--ulysses_size` 修改为 `10`。

### 1. 在主节点 (`bjdb-h20-node-080`) 上执行

```bash
nohup torchrun \
    --nproc_per_node=5 \
    --nnodes=2 \
    --node_rank=0 \
    --rdzv_id=wan22-job-$(date \'%Y%m%d\') \
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
    --prompt "A cinematic video. An astronaut in a white spacesuit walks cautiously on the red, dusty surface of Moon. The camera follows the astronaut from behind in a tracking shot. The astronaut stops and looks up in surprise at a giant, dark rock. The camera tilts up to reveal the word "tenxcloud" clearly engraved on the rock\'s surface. The scene is realistic, with a desolate Martian landscape and a faint sun in the background, creating a sense of mystery and discovery." \
    --save_file /workspace/output/$(date '+%Y%m%d%H%M%S').mp4  > /workspace/output/generate-tenxcloud.log 2>&1 &
```

### 2. 在工作节点 (`bjdb-h20-node-082`) 上执行

```bash
nohup torchrun \
    --nproc_per_node=5 \
    --nnodes=2 \
    --node_rank=1 \
    --rdzv_id=wan22-job-$(date \'%Y%m%d\') \
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
    --prompt "A cinematic video. An astronaut in a white spacesuit walks cautiously on the red, dusty surface of Moon. The camera follows the astronaut from behind in a tracking shot. The astronaut stops and looks up in surprise at a giant, dark rock. The camera tilts up to reveal the word "tenxcloud" clearly engraved on the rock\'s surface. The scene is realistic, with a desolate Martian landscape and a faint sun in the background, creating a sense of mystery and discovery." \
    --save_file /workspace/output/$(date '+%Y%m%d%H%M%S').mp4  > /workspace/output/generate-tenxcloud.log 2>&1 &
```

---

## 总结与排错

- **一致性:** 确保所有节点上的 `torchrun` 命令中，除了 `--node_rank` 不同外，其他参数都完全一致。
- **网络:** 确保主节点 (`172.31.0.80`) 的 `29500` 端口可以被所有工作节点访问。
- **执行顺序:** 最好先在主节点上启动命令，然后再启动工作节点上的命令。
