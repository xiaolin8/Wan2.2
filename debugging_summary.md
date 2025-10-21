# PyTorchJob 部署调试总结

本次调试主要围绕 `wan22-pytorchjob.yaml` 配置文件引发的 Kubeflow PyTorchJob 调度与运行问题。

## 遇到的主要问题及解决方案：

### 1. 错误：`CUDA error: invalid device ordinal`
*   **原因**：`wan22-pytorchjob.yaml` 中，每个 Pod 请求 1 个 GPU (`nvidia.com/gpu: "1"`)，但 `torchrun` 命令设置 `--nproc_per_node=5`，导致每个 Pod 启动 5 个进程，试图访问不存在的 GPU 索引。
*   **解决方案**：修改 `wan22-pytorchjob.yaml` 中的 `Master` 和 `Worker` 部分，将 `--nproc_per_node` 从 `5` 调整为 `1`。

### 2. 错误：`AssertionError: The number of ulysses_size should be equal to the world size.`
*   **原因**：`generate.py` 脚本要求 `ulysses_size` 参数必须与分布式任务的总进程数 (`world_size`) 相等。由于之前的配置调整，`world_size` 可能为 2 (`1 Master * 1 proc + 1 Worker * 1 proc`)，但 `--ulysses_size` 设置为 `10`，导致断言失败。
    *   **后续**：即使将 `nproc_per_node` 调整为 `2` (导致 `world_size=4`)，`ulysses_size` 最初仍为 `4`。后来调整为 7 GPUs/node 时，`ulysses_size` 调整为 14。
*   **解决方案**：根据实际的 `world_size` 调整 `--ulysses_size`。当 `world_size` 是 2 个 Pod * 2 进程/Pod = 4 时，将其设置为 `4`。当 `world_size` 是 2 个 Pod * 7 进程/Pod = 14 时，将其设置为 `14`。

### 3. 错误：`NCCL error ... No space left on device`
*   **原因**：容器内部的 `/dev/shm` 共享内存空间不足，PyTorch 的 NCCL 后端在进行多 GPU 通信时需要更大的共享内存。默认的 `/dev/shm` 大小（通常为 64MB）不足。
*   **解决方案**：修改 `wan22-pytorchjob.yaml`，在 `Master` 和 `Worker` 的 Pod 模板中增加一个名为 `dshm` 的 `emptyDir` 卷，并将其 `medium` 设置为 `Memory`、`sizeLimit` 设置为 `2Gi`，然后挂载到 `/dev/shm`。

### 4. 错误：`Signal 9 (SIGKILL)` / OOM Killer
*   **原因**：Pod 进程被操作系统 OOM (Out Of Memory) killer 终止，表明容器内存（可能是系统 RAM 或 GPU 内存）不足。
*   **解决方案**：
    *   **降低内存消耗**：将 `generate.py` 中的 `frame_num` 参数从 `81` 减少到 `41` (或更小)，以减少视频生成时的内存占用。
    *   **移除不当的 GPU 内存限制**：发现 `nvidia.com/gpumem: 64k` 这个极低的 GPU 内存限制是错误配置，将其从 `resources.limits` 中移除。

### 5. 错误：Pod 处于 `Pending` 状态，原因 `Insufficient memory`
*   **原因**：Pod 请求了 `256Gi` 的系统内存，但集群中有 8 个节点没有足够的可用内存来满足此请求。
*   **解决方案**：降低 Pod 请求的系统内存限制。将 `memory` 从 `256Gi` 降低到 `128Gi`，并为 `cpu` 添加明确的请求 (`8`) 和限制 (`16`)。

### 6. 错误：`2 NodeUnfitPod`
*   **原因**：Kubernetes 调度器无法将 Pod 调度到 GPU 节点上。尽管 `nvidia-smi` 显示这些节点有空闲 GPU，但 `hami-scheduler` 报告节点不适合。
*   **解决方案**：
    *   **添加 `nodeSelector`**：在 `wan22-pytorchjob.yaml` 的 `Master` 和 `Worker` 的 `template.spec` 中添加 `nodeSelector: gpu: "on"`，确保 Pod 只尝试调度到带有 `gpu=on` 标签的节点。
    *   **进一步排查**：如果添加 `nodeSelector` 后仍 `Pending`，则需要检查 `hami-scheduler` 和 `hami-nvidia-device-plugin` 的日志，以及 HAMI 虚拟化环境的配置，因为问题更可能出在 Kubernetes 的 GPU 资源管理层。

### 7. 错误：`AssertionError: `cfg.num_heads=40` cannot be divided evenly by `args.ulysses_size=14`.`
*   **原因**：修改配置为 7 GPU/节点时，`world_size` 变为 14，`ulysses_size` 随之也变为 14。然而模型配置中的 `num_heads` 为 40，`40` 无法被 `14` 整除，因此断言失败。
*   **解决方案**：将配置调整为每个 Pod 请求 4 张 GPU。在这种情况下，`world_size` 为 8 (2 个 Pod * 4 进程/Pod)，`ulysses_size` 设置为 8。`40` 可以被 `8` 整除，符合模型要求。

## 最终稳定配置 (基于 2 个 GPU 节点，每个节点 4 个可用 GPU)：

*   **Job 名称**：`wan22-t2v-distributed4`
*   **Master 副本数**：`1`
*   **Worker 副本数**：`1`
*   **总 Pod 数**：2 (1 个 Master, 1 个 Worker)
*   **每个 Pod 的 GPU 数**：`4`
*   **每个 Pod 的进程数 (`nproc_per_node`)**：`4`
*   **总进程数 (`world_size`)**：`2 个 Pod * 4 个进程/Pod = 8`
*   **`ulysses_size`**：`8` (与 `world_size` 匹配，并且可以整除 `cfg.num_heads=40`)
*   **`frame_num`**：已移除 (使用默认值 `81`)
*   `/dev/shm` 增加到 `2Gi`。
*   `nvidia.com/gpumem` 限制已移除。
*   **每个 Pod 的内存**：`128Gi` (请求和限制)。
*   **每个 Pod 的 CPU**：`8` 核 (请求) / `16` 核 (限制)。
*   **`nodeSelector: gpu: "on"`** 已添加到 Master 和 Worker。

**应用此配置的步骤：**
1.  删除旧的 Job (如果存在): `kubectl -n hu delete pytorchjob wan22-t2v-distributed4`
2.  应用新的配置: `kubectl apply -f /Users/xiaolin/IdeaProjects/Wan2.2/wan22-pytorchjob.yaml`
3.  检查 Master Pod 日志: `kubectl -n hu logs -f pod/wan22-t2v-distributed4-master-0`
