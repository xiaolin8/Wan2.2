#!/usr/bin/env python3
"""
Diffusers + DeepSpeed 多机多卡实现
使用 accelerate launch 启动真正的跨节点模型并行
"""

import torch
from diffusers import DiffusionPipeline
import imageio
import os
import numpy as np
import argparse
from PIL import Image
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description="Run Multi-Node Distributed Inference with Diffusers and DeepSpeed")
    
    # --- 路径参数 ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Diffusers model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video file.")
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
    
    # 初始化 Accelerator（自动处理多机多卡）
    accelerator = Accelerator()
    
    print(f"--- Accelerator Info ---")
    print(f"Process index: {accelerator.process_index}")
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Local process index: {accelerator.local_process_index}")
    print(f"Device: {accelerator.device}")
    print(f"Is main process: {accelerator.is_main_process}")
    
    # 只在主进程打印信息
    if accelerator.is_main_process:
        print(f"--- Loading model from {args.model_path} for multi-node inference ---")
    
    # 加载模型（Accelerator 会自动处理分布式）
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    
    # 使用 Accelerator 准备模型（关键步骤）
    pipe = accelerator.prepare(pipe)
    
    # 在主进程打印设备映射信息
    if accelerator.is_main_process:
        print(f"Model successfully distributed across {accelerator.num_processes} processes")
    
    # 启用 xformers（如果可用）
    try:
        pipe.enable_xformers_memory_efficient_attention()
        if accelerator.is_main_process:
            print("xFormers memory efficient attention enabled.")
    except Exception:
        if accelerator.is_main_process:
            print("xFormers not available, running with default attention.")

    # 准备 pipeline 参数
    pipeline_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": torch.manual_seed(args.seed)
    }

    # 处理输入图片
    if args.input_image:
        if accelerator.is_main_process:
            print(f"--- Loading input image from {args.input_image} ---")
        try:
            input_image = Image.open(args.input_image).convert("RGB")
            pipeline_kwargs["image"] = input_image
        except FileNotFoundError:
            if accelerator.is_main_process:
                print(f"Warning: Input image not found at {args.input_image}. Running in T2V mode.")

    # 生成视频
    if accelerator.is_main_process:
        print(f"--- Generating video for prompt: '{args.prompt}' ---")
    
    # 使用 barrier 确保所有进程同步
    accelerator.wait_for_everyone()
    
    # 执行推理（Accelerator 会自动处理分布式计算）
    video_frames = pipe(**pipeline_kwargs).frames
    
    # 等待所有进程完成
    accelerator.wait_for_everyone()
    
    # 只在主进程处理保存
    if accelerator.is_main_process:
        print("--- Processing frames for saving ---")
        video_np = np.array(video_frames)
        if video_np.ndim == 5 and video_np.shape[0] == 1:
            video_np = video_np[0]
        video_uint8 = (video_np * 255).astype(np.uint8)

        output_dir = os.path.dirname(args.output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"--- Saving video to {args.output_path} ---")
        imageio.mimsave(args.output_path, video_uint8, fps=args.fps, codec="libx264")
        print("--- Multi-node inference completed! ---")

if __name__ == "__main__":
    main()