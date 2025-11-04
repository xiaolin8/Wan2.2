#!/usr/bin/env python3
"""
使用WanPipeline的正确实现，参考HuggingFace模型页面示例
"""

import torch
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan2.2-T2V-A14B with WanPipeline")
    
    # --- 路径参数 ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Diffusers model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video file.")

    # --- 生成内容参数 ---
    parser.add_argument("--prompt", type=str, default="A robot surfing on a wave, cinematic", help="The text prompt.")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, watermark", help="The negative text prompt.")

    # --- 推理细节参数 ---
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale (CFG).")
    parser.add_argument("--guidance_scale_2", type=float, default=3.0, help="Second guidance scale.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed for reproducibility.")
    parser.add_argument("--fps", type=int, default=8, help="FPS for the output video.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"--- Loading Wan2.2 model from {args.model_path} ---")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # 加载VAE和pipeline（使用正确的类名）
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    pipe = WanPipeline.from_pretrained(
        args.model_path,
        vae=vae,
        torch_dtype=dtype,
        local_files_only=True
    )
    
    pipe.to(device)
    
    print("Model loaded successfully")
    
    # 生成视频
    print(f"--- Generating video for prompt: '{args.prompt}' ---")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 调用pipeline
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        num_inference_steps=args.num_inference_steps,
    ).frames[0]
    
    # 保存视频
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Saving video to {args.output_path} ---")
    export_to_video(output, args.output_path, fps=args.fps)
    
    print("--- Video generation completed! ---")

if __name__ == "__main__":
    main()