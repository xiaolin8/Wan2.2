#!/usr/bin/env python3
"""
Wan2.2-T2V-A14B-Diffusers - Parametric Video Generation Script
This script generates video from text prompts using the Wan2.2-T2V-A14B-Diffusers model,
with all generation parameters being configurable via command-line arguments.
Inspired by HuggingFace examples and generate_parametric.py.
"""

import torch
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
import argparse
import os

# Force network interface for torch.distributed to bypass environment propagation issues
os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker0,veth'

def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan2.2-T2V-A14B-Diffusers Parametric Inference")
    
    # --- Path Parameters ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Diffusers model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video file.")
    
    # --- Generation Content Parameters ---
    parser.add_argument("--prompt", type=str, default="A robot surfing on a wave, cinematic", help="The text prompt.")
    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", help="The negative text prompt.")
    
    # --- Inference Detail Parameters ---
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale (CFG).")
    parser.add_argument("--guidance_scale_2", type=float, default=3.0, help="Secondary guidance scale for WanPipeline.")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed for reproducibility.")
    parser.add_argument("--fps", type=int, default=16, help="FPS for the output video.")

    # --- Model Loading Parameters ---
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=['bfloat16', 'float16', 'float32'], help="Data type for the main model.")
    parser.add_argument("--vae_subfolder", type=str, default="vae", help="Subfolder for the VAE model.")
    parser.add_argument("--vae_dtype", type=str, default="float32", choices=['float16', 'float32'], help="Data type for the VAE.")
    parser.add_argument('--disable_local_files_only', dest='local_files_only', action='store_false', help="Disable loading models from local files only. It is enabled by default.")
    parser.set_defaults(local_files_only=True)
    
    # --- Performance Parameters ---
    parser.add_argument("--enable_xformers", action="store_true", help="Enable xformers memory efficient attention.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    vae_dtype_map = {"float16": torch.float16, "float32": torch.float32}
    
    main_dtype = dtype_map.get(args.dtype)
    vae_dtype = vae_dtype_map.get(args.vae_dtype)
    
    print(f"--- Using device: {device}, main_dtype: {args.dtype}, vae_dtype: {args.vae_dtype} ---")
    
    # --- Load Model ---
    print(f"--- Loading model from {args.model_path} ---")
    print(f"Local files only: {args.local_files_only}")
    
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path, 
        subfolder=args.vae_subfolder, 
        torch_dtype=vae_dtype,
        local_files_only=args.local_files_only
    )
    
    pipe = WanPipeline.from_pretrained(
        args.model_path, 
        vae=vae, 
        torch_dtype=main_dtype,
        local_files_only=args.local_files_only
    )
    pipe.to(device)
    
    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xFormers memory efficient attention enabled.")
        except Exception as e:
            print(f"Failed to enable xFormers: {e}. Running with default attention.")

    print("--- Model loaded successfully ---")
    
    # --- Prepare for generation ---
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    pipeline_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "generator": generator,
    }

    print(f"--- Generating video with the following parameters: ---")
    for key, value in pipeline_kwargs.items():
        if key != 'generator':
             print(f"  {key}: {value}")
    print(f"  seed: {args.seed}")

    # --- Generate Video ---
    output_frames = pipe(**pipeline_kwargs).frames[0]
    
    # --- Save Video ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"--- Saving video to {args.output_path} ---")
    export_to_video(output_frames, args.output_path, fps=args.fps)
    
    print("--- Video generation completed! ---")

if __name__ == "__main__":
    main()
