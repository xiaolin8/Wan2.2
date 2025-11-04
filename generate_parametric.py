# generate_parametric.py
import torch
from diffusers import DiffusionPipeline
import imageio
import os
import numpy as np
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Run video generation with Diffusers, with all parameters configurable.")
    # --- 路径参数 ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Diffusers model directory inside the container.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video file inside the container.")
    parser.add_argument("--input_image", type=str, default=None, help="(Optional) Path to an input image for TI2V models.")

    # --- 生成内容参数 ---
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean", help="The text prompt.")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, watermark", help="The negative text prompt.")

    # --- 推理细节参数 ---
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="Guidance scale (CFG).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--fps", type=int, default=8, help="FPS for the output video.")

    return parser.parse_args()

def main():
    args = parse_args()

    print(f"--- Loading model from {args.model_path} ---")
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers memory efficient attention enabled.")
    except Exception:
        print("xFormers not available, running with default attention.")

    # 准备 pipeline 的参数字典
    pipeline_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed)
    }

    # 如果提供了输入图片，则加载并加入参数
    if args.input_image:
        print(f"--- Loading input image from {args.input_image} ---")
        try:
            input_image = Image.open(args.input_image).convert("RGB")
            pipeline_kwargs["image"] = input_image
        except FileNotFoundError:
            print(f"Warning: Input image not found at {args.input_image}. Running in T2V mode.")

    print(f"--- Generating video for prompt: '{args.prompt}' ---")
    video_frames = pipe(**pipeline_kwargs).frames

    print("--- Processing frames for saving ---")
    video_np = np.array(video_frames)
    if video_np.ndim == 5 and video_np.shape[0] == 1:
        video_np = video_np[0]
    video_uint8 = (video_np * 255).astype(np.uint8)

    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Saving video to {args.output_path} ---")
    imageio.mimsave(args.output_path, video_uint8, fps=args.fps, codec="libx264")
    print("--- Done! ---")

if __name__ == "__main__":
    main()
