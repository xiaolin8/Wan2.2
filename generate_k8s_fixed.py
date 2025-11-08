#!/usr/bin/env python3
"""
Wan2.2-T2V-A14B-Diffusers - Kubernetes/Ray 环境专用脚本
针对 RayJob 和 Kubernetes 集群优化的分布式视频生成脚本
"""

import torch
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
import argparse
import os
import sys
import socket
import time

def setup_distributed():
    """设置分布式环境"""
    print("=== 分布式环境设置 ===")
    
    # 检查 PyTorch 安装
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    # 设置网络环境变量
    try:
        # 获取本机 IP
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"主机名: {hostname}")
        print(f"本地 IP: {local_ip}")
        
        # 设置分布式环境变量
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', local_ip)
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        
        print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
        
    except Exception as e:
        print(f"网络设置警告: {e}")
        # 使用默认值
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan2.2-T2V-A14B-Diffusers on Kubernetes/Ray")
    
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
    parser.add_argument("--dtype", type=str, default="float16", choices=['bfloat16', 'float16', 'float32'], help="Data type for the main model.")
    parser.add_argument("--vae_subfolder", type=str, default="vae", help="Subfolder for the VAE model.")
    parser.add_argument("--vae_dtype", type=str, default="float16", choices=['float16', 'float32'], help="Data type for the VAE.")
    parser.add_argument('--disable_local_files_only', dest='local_files_only', action='store_false', help="Disable loading models from local files only. It is enabled by default.")
    parser.set_defaults(local_files_only=True)
    
    # --- Performance Parameters ---
    parser.add_argument("--enable_xformers", action="store_true", help="Enable xformers memory efficient attention.")
    parser.add_argument("--enable_cpu_offload", action="store_true", help="Enable CPU offload for memory optimization.")

    return parser.parse_args()

def load_model_safely(args):
    """安全地加载模型"""
    print("=== 模型加载 ===")
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    vae_dtype_map = {"float16": torch.float16, "float32": torch.float32}
    
    main_dtype = dtype_map.get(args.dtype)
    vae_dtype = vae_dtype_map.get(args.vae_dtype)
    
    print(f"使用设备: {device}")
    print(f"主模型数据类型: {args.dtype}")
    print(f"VAE 数据类型: {args.vae_dtype}")
    
    # 加载 VAE
    print("加载 VAE...")
    try:
        vae = AutoencoderKLWan.from_pretrained(
            args.model_path, 
            subfolder=args.vae_subfolder, 
            torch_dtype=vae_dtype,
            local_files_only=args.local_files_only
        )
        print("VAE 加载成功")
    except Exception as e:
        print(f"VAE 加载失败: {e}")
        raise
    
    # 加载主模型
    print("加载主模型...")
    try:
        pipe = WanPipeline.from_pretrained(
            args.model_path, 
            vae=vae, 
            torch_dtype=main_dtype,
            local_files_only=args.local_files_only
        )
        print("主模型加载成功")
    except Exception as e:
        print(f"主模型加载失败: {e}")
        raise
    
    # 移动到设备
    if args.enable_cpu_offload:
        print("启用 CPU 卸载优化...")
        try:
            pipe.enable_model_cpu_offload()
            print("CPU 卸载启用成功")
        except Exception as e:
            print(f"CPU 卸载失败，继续使用 GPU: {e}")
            pipe.to(device)
    else:
        pipe.to(device)
    
    # 启用 xFormers
    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xFormers 启用成功")
        except Exception as e:
            print(f"xFormers 启用失败: {e}，使用默认注意力机制")
    
    print("=== 模型加载完成 ===")
    return pipe

def main():
    """主函数"""
    print("Wan2.2-T2V-A14B-Diffusers Kubernetes/Ray 环境启动")
    print("=" * 60)
    
    # 设置分布式环境
    setup_distributed()
    
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录创建: {output_dir}")
    
    # 记录开始时间
    start_time = time.time()
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 加载模型
        pipe = load_model_safely(args)
        
        # 准备生成参数
        generator = torch.Generator(device=pipe.device if hasattr(pipe, 'device') else "cuda").manual_seed(args.seed)
        
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
        
        print("\n=== 生成参数 ===")
        for key, value in pipeline_kwargs.items():
            if key != 'generator':
                print(f"  {key}: {value}")
        print(f"  seed: {args.seed}")
        
        # 生成视频
        print("\n开始生成视频...")
        output_frames = pipe(**pipeline_kwargs).frames[0]
        
        # 保存视频
        print(f"保存视频到: {args.output_path}")
        export_to_video(output_frames, args.output_path, fps=args.fps)
        
        # 计算耗时
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("🎉 视频生成完成！")
        print(f"输出文件: {args.output_path}")
        print(f"耗时: {duration:.2f} 秒")
        print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)