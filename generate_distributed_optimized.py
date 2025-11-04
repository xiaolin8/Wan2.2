# generate_distributed.py (最终优化版)
import torch
from diffusers import DiffusionPipeline
import imageio
import os
import numpy as np
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Run Distributed Inference with Diffusers, Accelerate, and xFormers.")
    # ... (所有参数定义与之前完全相同) ...
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Diffusers model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video file.")
    parser.add_argument("--input_image", type=str, default=None, help="(Optional) Path to an input image for TI2V models.")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean", help="The text prompt.")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, watermark", help="The negative text prompt.")
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="Guidance scale (CFG).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--fps", type=int, default=8, help="FPS for the output video.")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"--- Loading model from {args.model_path} for distributed inference ---")
    
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        device_map="auto"  # 启用模型并行
    )
    print(f"Pipeline device map:\n{pipe.device_map}")

    # --- 核心补充：在加载模型后，启用 xformers 优化 ---
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers memory efficient attention enabled.")
    except Exception as e:
        print(f"Could not enable xformers: {e}. Running with default attention.")

    # 准备 pipeline 的参数字典
    pipeline_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": torch.manual_seed(args.seed)
    }

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
