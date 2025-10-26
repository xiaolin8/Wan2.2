# wan/core_logic.py
# This file contains the refactored, core logic for video generation.
# It is designed to be importable and used by different execution backends (like Ray Serve).

import logging
import os
import random
import sys
import types
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.distributed.util import get_world_size, init_distributed_group
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)s: %(message)s",
                            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.INFO)

def run_generation_task(rank: int, world_size: int, local_rank: int, config: dict):
    """
    Runs a single video generation task.

    Args:
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.
        local_rank (int): The local rank on the current node.
        config (dict): A dictionary containing all necessary parameters for generation,
                       such as 'task', 'prompt', 'ckpt_dir', 'sampling_steps', etc.
    Returns:
        A tuple of (video_tensor, config) if rank is 0, otherwise (None, None).
    """
    device = local_rank
    _init_logging(rank)

    # --- Distributed Setup ---
    # This part will be handled by Ray's ProcessGroup in the final version,
    # but we keep the logic here for clarity.
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        # The 'env://' init method is standard for torchrun and Ray.
        dist.init_process_group(backend="nccl", init_method="env://")

    # --- Parameter Extraction ---
    task = config.get('task')
    cfg = WAN_CONFIGS[task]
    
    # Populate config with defaults from model config if not provided
    config.setdefault('sample_steps', cfg.sample_steps)
    config.setdefault('sample_shift', cfg.sample_shift)
    config.setdefault('sample_guide_scale', cfg.sample_guide_scale)
    config.setdefault('frame_num', cfg.frame_num)
    config.setdefault('base_seed', random.randint(0, sys.maxsize))

    logging.info(f"[{rank}] Running generation task with config: {config}")

    # --- Pipeline Creation (Simplified from generate.py) ---
    # In a real implementation, you would have a factory or switch-case here
    # to create the correct pipeline (i2v, t2v, etc.)
    # For this example, we'll focus on the I2V pipeline logic.

    if "i2v" not in task:
        if rank == 0:
            logging.warning(f"This refactored function currently focuses on the I2V task. Task '{task}' may not be fully supported yet.")
    
    logging.info(f"[{rank}] Creating WanI2V pipeline.")
    wan_pipeline = wan.WanI2V(
        config=cfg,
        checkpoint_dir=config.get('ckpt_dir'),
        device_id=device,
        rank=rank,
        t5_fsdp=config.get('t5_fsdp', False),
        dit_fsdp=config.get('dit_fsdp', False),
        use_sp=(config.get('ulysses_size', 1) > 1),
        t5_cpu=config.get('t5_cpu', False),
        convert_model_dtype=config.get('convert_model_dtype', False),
    )

    # --- Data Preparation ---
    img = Image.open(config.get('image')).convert("RGB") if config.get('image') else None
    if img is None and "i2v" in task:
        raise ValueError("Image path must be provided for I2V task.")

    # Broadcast seed from rank 0 to all other ranks to ensure consistency
    if dist.is_initialized():
        seed_tensor = torch.tensor([config.get('base_seed')], dtype=torch.long, device=device)
        dist.broadcast(seed_tensor, src=0)
        config['base_seed'] = seed_tensor.item()

    # --- Generation ---
    logging.info(f"[{rank}] Starting video generation...")
    video_tensor = wan_pipeline.generate(
        input_prompt=config.get('prompt'),
        img=img,
        max_area=MAX_AREA_CONFIGS.get(config.get('size')),
        frame_num=config.get('frame_num'),
        shift=config.get('sample_shift'),
        sample_solver=config.get('sample_solver', 'unipc'),
        sampling_steps=config.get('sample_steps'),
        guide_scale=config.get('sample_guide_scale'),
        seed=config.get('base_seed'),
        offload_model=config.get('offload_model', True),
        task_id=config.get('task_id'),
        progress_redis_client=config.get('progress_redis_client'),
        progress_topic=config.get('redis_progress_topic')
    )

    # --- Cleanup and Return ---
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    logging.info(f"[{rank}] Generation finished.")

    # Only rank 0 has the final decoded video tensor
    if rank == 0:
        return video_tensor, cfg
    else:
        return None, None
