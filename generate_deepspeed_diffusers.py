#!/usr/bin/env python3
"""
Wan2.2-T2V-A14B-Diffusers 视频生成脚本
使用官方HuggingFace示例代码
"""

import torch
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan2.2-T2V-A14B-Diffusers Inference")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Diffusers model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video file.")
    parser.add_argument("--prompt", type=str, default="A robot surfing on a wave, cinematic", help="The text prompt.")
    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", help="The negative text prompt.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale.")
    parser.add_argument("--guidance_scale_2", type=float, default=3.0, help="Secondary guidance scale.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed for reproducibility.")
    parser.add_argument("--fps", type=int, default=16, help="FPS for the output video.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置设备
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- Loading model from {args.model_path} ---")
    
    # 加载VAE
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path, 
        subfolder="vae", 
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    # 加载Pipeline
    pipe = WanPipeline.from_pretrained(
        args.model_path, 
        vae=vae, 
        torch_dtype=dtype,
        local_files_only=True
    )
    pipe.to(device)
    
    print("Model loaded successfully")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 生成视频
    print(f"--- Generating video for prompt: '{args.prompt}' ---")
    
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
    print(f"--- Saving video to {args.output_path} ---")
    export_to_video(output, args.output_path, fps=args.fps)
    
    print("--- Video generation completed! ---")

if __name__ == "__main__":
    main()