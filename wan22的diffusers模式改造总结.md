# Wan2.2 Diffusers 模式改造总结

## 1. 项目背景与目标

本项目旨在为 `Wan2.2` 系列模型（如 T2V, TI2V）在原生的、与项目代码紧密耦合的推理模式之外，创建一套基于 Hugging Face `diffusers` 标准库的全新推理工作流。

**核心目标**：

- **解耦与标准化**：摆脱对原生项目代码的依赖，使用业界标准的 `diffusers` API 进行模型加载和推理。
- **提升灵活性与可维护性**：获得更换采样器、集成 LoRA、无缝对接 `accelerate` 等 `diffusers` 生态的全部优势。
- **生产化**：借鉴原生脚本中的优秀设计（如 S3 上传、Redis 报告），构建一个功能完整、配置灵活、适用于生产环境的容器化解决方案。

---

## 2. 核心差异分析：原生模式 vs. Diffusers 模式

| 对比维度 | 原生模式 (`generate.py`) | Diffusers 模式 (`generate_final.py`) |
| :--- | :--- | :--- |
| **API 设计** | **一体化、高耦合**：一个巨大的 `generate` 函数，通过 `if/elif` 判断任务类型，调用内部类。 | **模块化、可组合**：以 `DiffusionPipeline` 为核心，各组件（UNet, VAE）清晰，逻辑由上层脚本控制。 |
| **灵活性** | **低**：修改任何推理逻辑（如采样循环）都需要深入修改框架源码。 | **高**：可轻易更换采样器 (`scheduler`)，或通过简单代码调整推理逻辑。 |
| **分布式方案** | **定制化、复杂**：采用 `torchrun + FSDP` 和自定义的 Ray 实现，与代码紧密绑定。 | **标准化、简单**：采用 `accelerate` 库的 `device_map="auto"` 功能，实现开箱即用的模型并行推理。 |
| **生态兼容性** | **孤立**：难以利用 `diffusers` 生态中的插件和优化。 | **完美融入**：无缝使用 `xformers`、PEFT (LoRA) 等所有 Hugging Face 生态工具。 |

---

## 3. 改造过程与挑战 (经验教训)

在将 `Wan2.2` 适配到 `diffusers` 模式的过程中，我们遇到并解决了一系列典型且有价值的技术挑战。

### 3.1. 环境构建与依赖冲突

这是本次改造中最核心的挑战，我们经历了数次失败和迭代。

- **初步错误**: `RuntimeError: operator torchvision::nms does not exist`。
- **问题根源**: 最终定位到问题出在用作起点的基础镜像 `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel` 本身。该镜像存在**环境不一致**的致命缺陷：其预装的 PyTorch 库是为 CUDA 12.8 编译的，而镜像内部的 `nvcc` 编译器版本却是 11.8。这导致任何需要编译 CUDA 扩展的库（如 `flash-attn`, `xformers`）都会失败。
- **最终方案**: **果断放弃有问题的基础镜像**，回归行业标准做法：
    1.  从一个干净、官方的 `nvidia/cuda:12.1.1-devel-ubuntu22.04` 镜像开始。
    2.  从零开始，**精确地安装一组已知互相兼容的库版本**（`torch`, `torchvision`, `xformers`, `flash-attn`, `diffusers` 等）。
- **经验教训**: 构建稳定、可复现的 AI 环境时，**必须保证基础环境的干净和一致性**。使用官方标准镜像并**固定所有核心依赖的版本号**，是避免陷入“依赖地狱”的最佳实践。

### 3.2. 网络问题与加速策略

- **挑战**: 在国内环境中，从 `apt-get`、`pip` 官方源下载依赖速度极慢，尤其是 `xformers` 等大文件容易超时失败。
- **解决方案**：
    1.  **镜像源替换**: 在 Dockerfile 中将 `apt` 和 `pip` 的源替换为国内镜像（如阿里云、清华大学）。
    2.  **离线安装**: 对于特别巨大的 `.whl` 文件，采用“先在宿主机下载，再 `COPY` 进镜像安装”的离线化策略，100% 解决下载失败问题。

### 3.3. 运行时问题修复

- **挑战**: 环境跑通后，在保存视频时出现 `ValueError: Image must have 1, 2, 3 or 4 channels`。
- **解决方案**: 这是 `diffusers` 输出的数据格式与 `imageio` 期望的格式不匹配导致的。通过 `numpy` 对输出的帧列表进行简单的后处理（调整维度、将 `float` 类型转换为 `uint8`），即可解决。

---

## 4. 最终成果：生产级 Diffusers 工作流

我们最终产出了两个核心文件，共同构成了一套功能强大且稳健的生产级工作流。

### 4.1. `Dockerfile.final` (健壮的构建文件)

- **干净的基础**: 基于 NVIDIA 官方 CUDA 镜像。
- **可复现的环境**: 精确固定了 `torch`, `torchvision`, `xformers`, `diffusers` 等所有核心库的版本号。
- **高效的构建**: 采用多阶段构建减小体积，并内置了国内镜像源加速。
- **生产化**: 内置了 `boto3` (S3) 和 `redis` 库，为业务集成做好准备。

### 4.2. `generate_final.py` (功能丰富的推理脚本)

该脚本是整个工作流的核心，它借鉴了原生 `generate.py` 的优点，并与 `diffusers` 的简洁性相结合：

- **全参数化**: 通过 `argparse` 实现对模型路径、提示词、采样步数、随机种子等所有关键参数的命令行控制。
- **分布式推理**: 默认启用 `device_map="auto"`，开箱即用地支持多 GPU 模型并行推理。
- **性能优化**: 自动检测并启用 `xformers` 以加速计算、节省显存。
- **业务集成**: 内置 S3 自动上传和 Redis 进度报告功能，并可通过命令行参数进行配置。
- **高可用性**: 包含动态文件名生成、健壮的 `try...except` 错误处理和状态上报机制。

---

## 5. 最终使用方法

1.  **构建镜像**：
    ```bash
    docker build -f Dockerfile.final -t wan22-diffusers-prod:latest .
    ```

2.  **运行一个生产级的推理任务**：
    ```bash
    # 注入云存储凭证
    # export AWS_ACCESS_KEY_ID=...
    # export AWS_SECRET_ACCESS_KEY=...
    
    docker run --rm --gpus all \
      -e AWS_ACCESS_KEY_ID \
      -e AWS_SECRET_ACCESS_KEY \
      -v /path/to/model:/models/model:ro \
      -v /path/to/outputs:/app/outputs \
      -v /path/to/generate_final.py:/app/generate.py \
      wan22-diffusers-prod:latest \
      python /app/generate.py \
        --model_path /models/model \
        --output_path /app/outputs \
        --prompt "A majestic eagle soaring through a thunderstorm" \
        --task_id "task-$(date +%s)" \
        --redis_host "my-redis.internal" \
        --s3_bucket "my-video-bucket" \
        --s3_endpoint_url "https://s3.my-company.com"
    ```

## 6. 总结

通过本次改造，我们成功地将 `Wan2.2` 模型从一个与框架紧密耦合的原生实现，转换为了一个基于 `diffusers` 标准库的、灵活、健壮且功能完备的生产级工作流。这个过程不仅解决了最初的兼容性问题，还为项目带来了更高的可维护性、可扩展性和行业生态兼容性，为后续的开发和部署铺平了道路。

![image-20251105134305532](assets/image-20251105134305532.png)
