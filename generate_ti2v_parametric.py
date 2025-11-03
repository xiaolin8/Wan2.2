# generate_ti2v_parametric.py
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import imageio
import os
import argparse # 导入 argparse 库

def parse_args():
    parser = argparse.ArgumentParser(description="Run Text-Image-to-Video generation with Diffusers.")
    # --- 路径参数 ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Diffusers model directory.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video file.")
    
    # --- 生成内容参数 ---
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean", help="The text prompt.")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, watermark", help="The negative text prompt.")
    
    # --- 推理细节参数 ---
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="Guidance scale (CFG).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # --- 1. 加载并配置模型 ---
    print(f"Loading model from {args.model_path}...")
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    # --- 2. 加载输入图片 ---
    print(f"Loading input image from {args.input_image}...")
    input_image = Image.open(args.input_image).convert("RGB")

    # --- 3. 生成视频 ---
    print(f"Generating video for prompt: '{args.prompt}'")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    
    video_frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=input_image,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator
    ).frames

    # --- 4. 保存视频 ---
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving video to {args.output_path}...")
    imageio.mimsave(args.output_path, video_frames, fps=8)
    print("Done!")

if __name__ == "__main__":
    main()
