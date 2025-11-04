import torch
from diffusers import DiffusionPipeline
import imageio
import os
import numpy as np
import argparse
from PIL import Image
import ray
import socket
import logging
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run Multi-Node Distributed Inference with Diffusers and Ray")
    
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

    # --- 分布式参数 ---
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use for distributed inference.")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of GPUs per node.")
    
    return parser.parse_args()

def _init_logging(rank):
    log_level = logging.INFO if rank == 0 else logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format=f"[%(asctime)s][Node {rank}] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

@ray.remote(num_gpus=1)
class DiffusionWorker:
    def __init__(self, model_path, node_id, worker_id):
        self.node_id = node_id
        self.worker_id = worker_id
        self.model_path = model_path
        
        _init_logging(node_id)
        logging.info(f"Worker {worker_id} on node {node_id} initializing...")
        
        # 每个 worker 加载完整的模型，但使用 device_map="auto" 自动分配
        self.pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            device_map="auto"  # 关键：自动处理多GPU
        )
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logging.info(f"Worker {worker_id}: xFormers enabled")
        except Exception:
            logging.info(f"Worker {worker_id}: xFormers not available")
            
        logging.info(f"Worker {worker_id} on node {node_id} ready. Device map: {self.pipe.device_map}")
    
    def generate_frames(self, prompt, negative_prompt, num_frames, num_inference_steps, guidance_scale, seed, input_image=None):
        logging.info(f"Worker {self.worker_id} generating {num_frames} frames")
        
        pipeline_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": torch.manual_seed(seed)
        }
        
        if input_image:
            try:
                image = Image.open(input_image).convert("RGB")
                pipeline_kwargs["image"] = image
                logging.info(f"Worker {self.worker_id}: Loaded input image")
            except Exception as e:
                logging.warning(f"Worker {self.worker_id}: Failed to load image: {e}")
        
        # 执行推理
        result = self.pipe(**pipeline_kwargs)
        frames = result.frames
        
        # 处理帧数据
        video_np = np.array(frames)
        if video_np.ndim == 5 and video_np.shape[0] == 1:
            video_np = video_np[0]
        video_uint8 = (video_np * 255).astype(np.uint8)
        
        logging.info(f"Worker {self.worker_id}: Generation completed")
        return video_uint8

def main():
    args = parse_args()
    
    # 单节点模式（直接使用 device_map="auto"）
    if args.num_nodes == 1:
        _init_logging(0)
        logging.info("Running in single-node mode with device_map='auto'")
        
        pipe = DiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            device_map="auto"
        )
        
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logging.info("xFormers enabled")
        except Exception:
            logging.info("xFormers not available")
        
        pipeline_kwargs = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "generator": torch.manual_seed(args.seed)
        }
        
        if args.input_image:
            try:
                input_image = Image.open(args.input_image).convert("RGB")
                pipeline_kwargs["image"] = input_image
                logging.info("Loaded input image")
            except Exception as e:
                logging.warning(f"Failed to load image: {e}")
        
        logging.info("Generating video...")
        result = pipe(**pipeline_kwargs)
        video_frames = result.frames
        
        video_np = np.array(video_frames)
        if video_np.ndim == 5 and video_np.shape[0] == 1:
            video_np = video_np[0]
        video_uint8 = (video_np * 255).astype(np.uint8)
        
        output_dir = os.path.dirname(args.output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Saving video to {args.output_path}")
        imageio.mimsave(args.output_path, video_uint8, fps=args.fps, codec="libx264")
        logging.info("Done!")
    
    else:
        # 多节点模式
        _init_logging(0)
        logging.info(f"Running in multi-node mode: {args.num_nodes} nodes, {args.gpus_per_node} GPUs per node")
        
        if not ray.is_initialized():
            ray.init(address='auto')
        
        # 创建 worker 池
        total_workers = args.num_nodes * args.gpus_per_node
        workers = []
        
        for node_id in range(args.num_nodes):
            for worker_id in range(args.gpus_per_node):
                worker = DiffusionWorker.remote(
                    args.model_path, 
                    node_id, 
                    worker_id + node_id * args.gpus_per_node
                )
                workers.append(worker)
        
        logging.info(f"Created {len(workers)} workers")
        
        # 目前采用简单的任务分配：第一个 worker 生成完整视频
        # 未来可以扩展为帧级并行或提示词并行
        logging.info("Assigning generation task to first worker...")
        
        future = workers[0].generate_frames.remote(
            args.prompt,
            args.negative_prompt,
            args.num_frames,
            args.num_inference_steps,
            args.guidance_scale,
            args.seed,
            args.input_image
        )
        
        video_uint8 = ray.get(future)
        
        # 保存结果
        output_dir = os.path.dirname(args.output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Saving video to {args.output_path}")
        imageio.mimsave(args.output_path, video_uint8, fps=args.fps, codec="libx264")
        
        if ray.is_initialized():
            ray.shutdown()
        
        logging.info("Multi-node inference completed!")

if __name__ == "__main__":
    main()