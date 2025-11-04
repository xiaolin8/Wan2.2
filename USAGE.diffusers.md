# 如何使用 Diffusers 模式的 Docker 镜像

本文档提供了构建和使用 `Dockerfile.diffusers` 的详细步骤。

---

## 1. Dockerfile 最佳实践解析

这个 `Dockerfile.diffusers` 采用了多项最佳实践：

- **多阶段构建**: 第一阶段 `builder` 用于下载模型和安装依赖，第二阶段只拷贝必要产物，最终镜像体积更小、更干净。
- **虚拟环境**: 所有 Python 依赖都安装在 `/opt/venv` 虚拟环境中，避免了与系统库的冲突。
- **非 Root 用户**: 创建了一个名为 `appuser` 的普通用户来运行应用，这是重要的安全实践，避免了容器内的进程拥有 root 权限。
- **依赖缓存**: `RUN` 指令的顺序经过优化，以最好地利用 Docker 的层缓存机制。
- **性能优化**: 默认安装了 `xformers` 库，以在推理时获得最佳的速度和显存效率。

---

## 2. 构建 Docker 镜像

在包含 `Dockerfile.diffusers` 的项目根目录下，打开终端并执行以下命令：

```bash
# -f 指定 Dockerfile 的文件名
# -t 给镜像起一个名字，例如 wan22-diffusers:latest

docker build -f Dockerfile.diffusers -t 172.31.0.182/system_containers/wan22-diffusers:latest . && docker push 172.31.0.182/system_containers/wan22-diffusers:latest

IMAGE_NAME="172.31.0.182/system_containers/wan22-diffusers:1103"
LOG_FILE="docker-build.log"
nohup sh -c "{ docker build -f Dockerfile.diffusers.hybrid . -t \"$IMAGE_NAME\" && docker push \"$IMAGE_NAME\"; } >> \"$LOG_FILE\" 2>&1" &

```

构建过程会比较长，因为它需要下载基础镜像、安装依赖和克隆十几 GB 的模型文件。请耐心等待。

---

## 3. 运行 Docker 容器

镜像构建完成后，模型和所有依赖都已封装在内。你可以用多种方式运行它。

### 方式 A：交互式 Shell (用于调试和探索)

如果你想进入容器内部进行探索或手动执行脚本，可以使用以下命令：

```bash
# --rm       : 容器退出后自动删除
# -it        : 启动一个交互式的终端
# --gpus all : 将宿主机的所有 NVIDIA GPU 挂载到容器中 (必需)
docker run --rm -it --gpus all wan22-diffusers:latest

docker run -d --gpus all --name wan22 \
     -v /data/Wan-AI/Wan2.2-T2V-A14B-Diffusers:/Wan2.2-T2V-A14B-Diffusers \
     -v /data/Wan-AI/Wan2.2-T2V-A14B-Diffusers/output:/workspace/output \
     172.31.0.182/system_containers/wan22-diffusers:1103 \
     tail -f /dev/null

docker run --gpus all --name wan22      -v /data/Wan-AI/Wan2.2-T2V-A14B-Diffusers:/Wan2.2-T2V-A14B-Diffusers      -v /data/Wan-AI/Wan2.2-T2V-A14B-Diffusers/output:/workspace/output    wan22-diffusers-final:latest   tail -f /dev/null
```

进入容器后，你可以：
- 检查 Python 环境: `python -V`
- 查看模型文件: `ls -l /app/models/Wan2.2-T2V-A14B-Diffusers`
- 手动运行 Python 脚本。

### 方式 B：运行本地脚本 (最常用)

这是最推荐的使用方式。你在本地编写 Python 脚本，然后通过挂载卷的方式在容器中执行它。

**第一步：在本地创建一个脚本文件**

例如，在你的项目根目录下创建一个名为 `generate_video.py` 的文件，内容如下：

```python
# generate_video.py
import torch
from diffusers import DiffusionPipeline
import imageio
import os

# --- 1. 配置参数 ---
# 模型在 Docker 镜像中的路径是固定的
MODEL_PATH = "/Wan2.2-T2V-A14B-Diffusers"
OUTPUT_DIR = "/workspace/output"

PROMPT = "A beautiful sunset over the ocean, cinematic, masterpiece"
NEGATIVE_PROMPT = "low quality, blurry, watermark, signature, ugly"

# --- 2. 加载并配置模型 ---
def main():
    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True # 确保只使用本地文件
    )
    pipe.to("cuda")

    # 启用 xformers 优化
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers memory efficient attention enabled.")
    except ImportError:
        print("xFormers not installed, running with default attention.")

    # --- 3. 生成视频 ---
    print(f"Generating video for prompt: '{PROMPT}'")
    generator = torch.Generator(device="cuda").manual_seed(42)
    video_frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=25,
        num_inference_steps=50,
        guidance_scale=9.0,
        generator=generator
    ).frames

    # --- 4. 保存视频 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "diffusers.mp4")

    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, video_frames, fps=8)
    print("Done!")

if __name__ == "__main__":
    main()

```

**第二步：运行容器来执行该脚本**

```bash
# -v $(pwd):/app : 将当前目录挂载到容器的 /app 目录
# 容器启动后执行的命令是 python /app/generate_video.py
docker run --rm --gpus all -v $(pwd):/app wan22-diffusers:latest \
    python /app/generate_video.py
```

执行完毕后，你会在本地当前目录下的 `outputs` 文件夹里找到生成的 `result.mp4` 文件。
