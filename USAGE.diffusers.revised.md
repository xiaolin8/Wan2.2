# 如何使用 Diffusers 模式的 Docker 镜像 (模型外挂版)

本文档提供了构建和使用 `Dockerfile.diffusers.revised` 的详细步骤。此版本遵循模型与环境分离的最佳实践，镜像本身不包含模型文件。

---

## 1. 镜像特点

- **轻量级**: 镜像只包含运行环境和依赖库，体积小，构建速度快。
- **模型解耦**: 模型文件由宿主机提供，通过 Docker 的卷挂载功能在运行时载入，方便模型的更新和管理。
- **高性能**: 使用官方 PyTorch 镜像并内置 `xformers` 库，确保推理性能。
- **安全**: 使用非 root 用户 `appuser` 运行。

---

## 2. 构建 Docker 镜像

在项目根目录下，执行以下命令：

```bash
# -f 指定 Dockerfile 的文件名
# -t 给镜像起一个名字
docker build -f Dockerfile.diffusers.revised -t wan22-diffusers-runtime:latest .
```

---

## 3. 运行 Docker 容器 (核心步骤)

运行此镜像的核心在于使用 `-v` (或 `--volume`) 参数将宿主机上的模型、输入和输出目录挂载到容器中。

### 准备工作

假设你的本地文件结构如下：

```
/data/Wan-AI/Wan2.2-TI2V-5B-Diffusers/  # <-- 你的模型文件夹
/home/user/my_project/                  # <-- 你的工作目录
├── inputs/
│   └── my_image.png                    # <-- 用于 TI2V 的输入图片
├── outputs/                            # <-- 用于存放生成的视频
└── generate_ti2v.py                    # <-- 你的推理脚本
```

### 编写推理脚本 (`generate_ti2v.py`)

由于模型是 Text-Image-to-Video，调用接口需要同时提供文本和图片。请注意脚本中路径的写法，它们都是**容器内的路径**。

```python
# generate_ti2v.py
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import imageio
import os

# --- 1. 配置路径 (全部使用容器内的绝对路径) ---
MODEL_PATH = "/models/Wan2.2-TI2V-5B-Diffusers"
INPUT_IMAGE_PATH = "/app/inputs/my_image.png"
OUTPUT_DIR = "/app/outputs"

PROMPT = "A dancing astronaut in the ocean"

# --- 2. 加载并配置模型 ---
def main():
    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    # --- 3. 加载输入图片 ---
    print(f"Loading input image from {INPUT_IMAGE_PATH}...")
    input_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")

    # --- 4. 生成视频 (注意 API 可能需要 image 参数) ---
    print(f"Generating video for prompt: '{PROMPT}'")
    generator = torch.Generator(device="cuda").manual_seed(123)
    
    video_frames = pipe(
        prompt=PROMPT,
        image=input_image, # TI2V 模型需要输入图片
        num_frames=25,
        num_inference_steps=50,
        guidance_scale=9.0,
        generator=generator
    ).frames

    # --- 5. 保存视频 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "ti2v_result.mp4")
    
    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, video_frames, fps=8)
    print("Done!")

if __name__ == "__main__":
    main()
```

### 运行命令

打开终端，在你的工作目录 (`/home/user/my_project/`) 下执行：

```bash
docker run --rm --gpus all \
  -v /data/Wan-AI/Wan2.2-TI2V-5B-Diffusers:/models/Wan2.2-TI2V-5B-Diffusers:ro \
  -v $(pwd)/inputs:/app/inputs:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/generate_ti2v.py:/app/generate_ti2v.py \
  wan22-diffusers-runtime:latest \
  python /app/generate_ti2v.py
```

**命令解析**:
- `--gpus all`: 启用 GPU 支持。
- `-v /data/Wan-AI/...:/models/...:ro`: **挂载模型**。将宿主机的模型目录挂载到容器的 `/models` 路径下，并设置为只读 (`:ro`) 以防容器内意外修改。
- `-v $(pwd)/inputs:/app/inputs:ro`: **挂载输入**。将本地的 `inputs` 目录挂载到容器的 `/app/inputs`，同样设为只读。
- `-v $(pwd)/outputs:/app/outputs`: **挂载输出**。将本地的 `outputs` 目录挂载到容器的 `/app/outputs`，这样容器内生成的视频会直接出现在你的本地文件夹里。
- `-v $(pwd)/generate_ti2v.py:/app/generate_ti2v.py`: **挂载脚本**。
- `wan22-diffusers-runtime:latest`: 指定要运行的镜像。
- `python /app/generate_ti2v.py`: 在容器内要执行的命令。

```