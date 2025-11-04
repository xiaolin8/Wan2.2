# generate_video.py
import torch
from diffusers import DiffusionPipeline
import imageio
import os
import numpy as np # 导入 numpy

# --- 1. 配置参数 ---
# 这里的路径是容器内的路径
MODEL_PATH = "/Wan2.2-T2V-A14B-Diffusers"
OUTPUT_DIR = "/workspace/output"

PROMPT = "A beautiful sunset over the ocean, cinematic, masterpiece"
NEGATIVE_PROMPT = "low quality, blurry, watermark, signature, ugly"

def main():
    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    pipe.to("cuda")

    # 尝试启用 xformers 优化
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers memory efficient attention enabled.")
    except Exception:
        print("xFormers not installed or incompatible, running with default attention.")

    # --- 3. 生成视频 ---
    print(f"Generating video for prompt: '{PROMPT}'")
    generator = torch.Generator(device="cuda").manual_seed(42)
    video_frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=50,
        num_inference_steps=50,
        guidance_scale=9.0,
        generator=generator
    ).frames

    # --- 4. 核心修正：在保存前处理视频帧 ---
    print("Processing frames for saving...")
    #   a. 将 diffusers 的输出转换为 numpy 数组
    video_np = np.array(video_frames)

    #   b. 如果存在额外的批次维度，则去掉它
    if video_np.ndim == 5 and video_np.shape[0] == 1:
        video_np = video_np[0]

    #   c. 将 float32 [0,1] 转换为 uint8 [0,255]
    video_uint8 = (video_np * 255).astype(np.uint8)

    # --- 5. 保存视频 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "diffusers.mp4")
    
    print(f"Saving video to {output_path}...")
    # 使用处理后的 video_uint8 进行保存
    imageio.mimsave(output_path, video_uint8, fps=8, codec="libx264")
    print("Done!")

if __name__ == "__main__":
    main()
