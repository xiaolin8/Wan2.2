# Wan2.2 原生模式 vs. Diffusers 模式深度对比

## 核心思想：标准化、可组合 vs. 定制化、高耦合

本文档旨在深入对比 `Wan2.2` 项目的原生推理模式与基于 `diffusers` 库的标准化推理模式之间的核心差异。

- **原生模式 (`Wan-AI/Wan2.2-T2V-A14B`)**: 为模型研究和开发设计的、与项目代码紧密耦合的定制化方案。
- **Diffusers 模式 (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`)**: 为广泛应用和轻松部署设计的、标准化的、可组合的方案。

**结论先行：对于几乎所有生产、研究和应用场景，我们都强烈推荐使用 `Diffusers` 模式。**

---

## 1. API 接口对比：可组合的“积木” vs. 固定的“一体机”

| 对比维度 | 原生模式 (`generate.py`) | Diffusers 模式 (`DiffusionPipeline`) |
| :--- | :--- | :--- |
| **接口形态** | 一个固定的、多参数的函数，如 `generate(prompt, ...)`。 | 一个可配置、可分解的对象 `pipe`，由多个组件构成。 |
| **灵活性** | **低**。修改核心逻辑必须深入修改项目源代码。 | **高**。可轻易替换组件（如采样器），或继承并重写方法。 |
| **可组合性** | **几乎为零**。模型各部分（VAE, UNet）紧密耦合。 | **极高**。可自由“混搭”不同模型的兼容组件。 |

### 代码示例：更换采样器 (Scheduler)

**目标**：将默认采样器更换为 `DPMSolverMultistepScheduler`。

*   **原生模式**:
    ```python
    # 极其困难，需要重写 generate.py 内部的采样循环逻辑。
    # 几乎不可行。
    ```

*   **Diffusers 模式**:
    ```python
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    pipe = DiffusionPipeline.from_pretrained(...)
    
    # 仅需一行代码即可更换
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # 后续调用将自动使用新采样器
    pipe(prompt="...") 
    ```

---

## 2. 分布式推理对比：生态工具 vs. 定制化方案

对于 `Wan2.2-T2V-A14B` 这样的巨型模型，分布式推理是必需的。

### 原生模式：定制化 Ray + FSDP 方案

- **模型并行**: 通过 `FSDP` 或其他自定义并行策略，将单个模型实例拆分到多张 GPU 上。代码与模型实现紧密耦合。
- **服务分发**: 使用 `Ray Serve` 框架将服务部署到多机集群。配置（如 `wan22-rayservice.yaml`）是为该项目专门定制的。
- **优点**: 经过深度调优，可能达到极致的推理性能。
- **缺点**: **极其复杂**，与特定框架和硬件环境高度耦合，难以维护、迁移和二次开发。

### Diffusers 模式：标准化 `accelerate` 方案

- **模型并行**: 使用 Hugging Face `accelerate` 库，通过一行参数 `device_map="auto"` 即可自动将模型切分到所有可用 GPU。
- **服务分发**: **框架解耦**。可使用任何你熟悉的工具（如 Ray Serve, BentoML, KServe, TorchServe）进行部署。
- **优点**: **极其简单**，标准化，灵活，易于维护和部署到不同环境。
- **缺点**: 通用方案在极限性能上可能与手调方案有微小差距，但开发效率和通用性远超原生模式。

### 代码示例：启用模型并行

*   **原生模式**:
    ```python
    # 需要复杂的 Ray Actor 配置、FSDP 封装以及自定义的启动脚本。
    # 涉及多个 YAML 和 Python 文件。
    ```

*   **Diffusers 模式**:
    ```python
    from diffusers import DiffusionPipeline
    import torch

    # accelerate 会自动将模型的各层分布到所有可用的 GPU 上
    pipe = DiffusionPipeline.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        torch_dtype=torch.float16,
        device_map="auto"  # 魔法就在这里！
    )

    # 直接调用即可，无需关心底层 GPU 分布
    pipe(prompt="...")
    ```

---

## 3. 迁移指南与总结

从原生模式迁移到 Diffusers 模式是“一次性投资，长期受益”的正确决策。

### 核心迁移步骤

1.  **替换模型文件**: 删除旧的 `.pth` 文件。使用 `git clone` 重新下载 `Diffusers` 版本的模型文件夹。
2.  **重构加载代码**: 废弃所有手动的 `torch.load` 和模型实例化逻辑。切换为 `DiffusionPipeline.from_pretrained(model_path, ...)` 的标准加载方式。
3.  **更新依赖项**: 移除 `Wan2.2` 项目源码的依赖。确保 `requirements.txt` 中包含 `diffusers`, `transformers`, `accelerate`。

### 迁移带来的好处

- **代码极度简化**：核心推理代码从上百行缩减为几行。
- **维护性大大增强**：依赖于社区维护的工业级标准库，而非个人研究项目。
- **灵活性和扩展性**：轻松享受 `diffusers` 生态带来的所有优势（新采样器、LoRA、内存优化等）。
