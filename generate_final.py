# generate_final.py
import torch
from diffusers import DiffusionPipeline
import imageio
import os
import numpy as np
import argparse
from PIL import Image
import logging
import sys
from datetime import datetime
import json

# --- 基础设施集成 --- 

def setup_redis_client(args):
    if not args.task_id: return None
    try:
        import redis
        client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0, decode_responses=True)
        client.ping()
        logging.info(f"Connected to Redis at {args.redis_host}:{args.redis_port}")
        return client
    except ImportError:
        logging.warning("`redis` library not found. Progress reporting is disabled.")
    except Exception as e:
        logging.error(f"Failed to connect to Redis: {e}. Progress reporting is disabled.")
    return None

def publish_progress(client, task_id, status, **kwargs):
    if not client: return
    message = {"task_id": task_id, "status": status, **kwargs}
    try:
        client.publish(args.redis_progress_topic, json.dumps(message))
    except Exception as e:
        logging.error(f"Failed to publish progress to Redis: {e}")

def upload_to_s3(args, local_file_path):
    if not args.s3_bucket: return None
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
        s3_key = os.path.basename(local_file_path)
        if args.s3_key_prefix:
            s3_key = os.path.join(args.s3_key_prefix, s3_key)
        
        logging.info(f"Attempting to upload {local_file_path} to s3://{args.s3_bucket}/{s3_key}")
        s3_client = boto3.client('s3', endpoint_url=args.s3_endpoint_url)
        s3_client.upload_file(local_file_path, args.s3_bucket, s3_key)
        s3_url = f"s3://{args.s3_bucket}/{s3_key}"
        logging.info(f"Successfully uploaded to {s3_url}")
        return s3_url
    except ImportError:
        logging.warning("`boto3` library not found. S3 upload is disabled.")
    except Exception as e:
        logging.error(f"S3 Upload Error: {e}")
    return None

# --- 参数解析 --- 

def parse_args():
    parser = argparse.ArgumentParser(description="Production-ready script for video generation using Diffusers.")
    # --- Core Paths ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Diffusers model directory.")
    parser.add_argument("--output_path", type=str, default="/app/outputs", help="Path to save the output video. Can be a directory or a full .mp4 path.")
    parser.add_argument("--input_image", type=str, default=None, help="(Optional) Path to an input image for TI2V models.")
    # --- Generation Params ---
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean", help="The text prompt.")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, watermark", help="The negative text prompt.")
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="Guidance scale (CFG).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. If None, a random seed is used.")
    parser.add_argument("--fps", type=int, default=8, help="FPS for the output video.")
    # --- Infrastructure Params (from generate.py) ---
    parser.add_argument("--task_id", type=str, default=None, help="Unique ID for the current task, used for progress reporting.")
    parser.add_argument("--redis_host", type=str, default="localhost", help="Redis server hostname.")
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis server port.")
    parser.add_argument("--redis_progress_topic", type=str, default="wan22-progress", help="Redis topic for progress messages.")
    parser.add_argument("--s3_bucket", type=str, default=None, help="S3 bucket to upload the result to.")
    parser.add_argument("--s3_endpoint_url", type=str, default=None, help="S3 compatible endpoint URL.")
    parser.add_argument("--s3_key_prefix", type=str, default="videos", help="Prefix for the S3 object key.")
    return parser.parse_args()

# --- 主逻辑 ---

def main(args):
    # 动态生成文件名
    if not args.output_path.endswith(".mp4"):
        seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.prompt[:30].replace(' ', '_')}_{seed}_{timestamp}.mp4"
        args.output_path = os.path.join(args.output_path, filename)

    # 设置日志
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", stream=sys.stdout)

    # 连接 Redis
    redis_client = setup_redis_client(args)

    try:
        publish_progress(redis_client, args.task_id, "STARTED")

        # 加载模型
        publish_progress(redis_client, args.task_id, "LOADING_MODEL")
        logging.info(f"Loading model from {args.model_path}")
        pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, local_files_only=True, device_map="auto")
        logging.info(f"Pipeline device map: {pipe.device_map}")
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logging.info("xFormers memory efficient attention enabled.")
        except Exception as e:
            logging.warning(f"Could not enable xformers: {e}. Running with default attention.")

        # 准备参数
        seed = args.seed if args.seed is not None else torch.initial_seed()
        pipeline_kwargs = {"prompt": args.prompt, "negative_prompt": args.negative_prompt, "num_frames": args.num_frames, "num_inference_steps": args.num_inference_steps, "guidance_scale": args.guidance_scale, "generator": torch.manual_seed(seed)}
        if args.input_image:
            logging.info(f"Loading input image from {args.input_image}")
            pipeline_kwargs["image"] = Image.open(args.input_image).convert("RGB")

        # 生成视频
        publish_progress(redis_client, args.task_id, "GENERATING_VIDEO")
        logging.info(f"Generating video for prompt: '{args.prompt}'")
        video_frames = pipe(**pipeline_kwargs).frames

        # 后处理与保存
        publish_progress(redis_client, args.task_id, "SAVING_LOCALLY")
        logging.info("Processing frames for saving...")
        video_uint8 = (np.array(video_frames)[0] * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        logging.info(f"Saving video to {args.output_path}...")
        imageio.mimsave(args.output_path, video_uint8, fps=args.fps, codec="libx264")

        # 上传 S3
        final_url = upload_to_s3(args, args.output_path)
        if final_url:
            publish_progress(redis_client, args.task_id, "COMPLETED", final_url=final_url)
        else:
            publish_progress(redis_client, args.task_id, "COMPLETED", final_url=args.output_path)

        logging.info("--- Done! ---")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        publish_progress(redis_client, args.task_id, "FAILED", error_message=str(e))
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    main(args)
