# wan/core_logic.py
# This file contains the fully refactored, core logic for all video generation tasks.
# It is designed to be a modular and importable engine for different execution backends.

import logging
import os
import random
import sys

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander


def _init_logging(rank: int):
    """Initializes logging for the current process."""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"[%(asctime)s][Rank {rank}] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )

def run_generation_task(rank: int, world_size: int, local_rank: int, config: dict):
    """
    Runs a single, complete video generation task for any supported task type.

    Args:
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.
        local_rank (int): The local rank on the current node.
        config (dict): A dictionary containing all necessary parameters.

    Returns:
        A tuple of (video_tensor, model_config) if rank is 0, otherwise (None, None).
    """
    device = local_rank
    _init_logging(rank)

    # --- Distributed Setup (if not already initialized by Ray) ---
    if world_size > 1 and not dist.is_initialized():
        logging.info("Initializing torch.distributed process group...")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    # --- Parameter Extraction and Validation ---
    task = config.get('task')
    if not task or task not in WAN_CONFIGS:
        raise ValueError(f"Invalid or missing task in config: {task}")
    
    model_cfg = WAN_CONFIGS[task]
    config.setdefault('sample_steps', model_cfg.sample_steps)
    config.setdefault('sample_shift', model_cfg.sample_shift)
    config.setdefault('sample_guide_scale', model_cfg.sample_guide_scale)
    config.setdefault('frame_num', model_cfg.frame_num)
    config.setdefault('base_seed', random.randint(0, sys.maxsize))

    logging.info(f"Running generation task with config: {config}")

    # Broadcast seed from rank 0 to all other ranks for consistency
    if dist.is_initialized():
        seed_tensor = torch.tensor([config['base_seed']], dtype=torch.long, device=device)
        dist.broadcast(seed_tensor, src=0)
        config['base_seed'] = seed_tensor.item()

    # --- Data and Prompt Preparation ---
    img = Image.open(config['image']).convert("RGB") if config.get('image') else None
    prompt = config.get('prompt', '')

    # (Optional) Prompt extension logic can be added here if needed

    # --- Pipeline Creation and Execution ---
    # This block replicates the logic from the original generate.py
    video_tensor = None
    pipeline_class = None
    pipeline_args = {
        'config': model_cfg,
        'checkpoint_dir': config.get('ckpt_dir'),
        'device_id': device,
        'rank': rank,
        't5_fsdp': config.get('t5_fsdp', False),
        'dit_fsdp': config.get('dit_fsdp', False),
        'use_sp': (config.get('ulysses_size', 1) > 1),
        't5_cpu': config.get('t5_cpu', False),
        'convert_model_dtype': config.get('convert_model_dtype', False),
    }
    generate_args = {
        'shift': config.get('sample_shift'),
        'sample_solver': config.get('sample_solver', 'unipc'),
        'sampling_steps': config.get('sample_steps'),
        'guide_scale': config.get('sample_guide_scale'),
        'seed': config.get('base_seed'),
        'offload_model': config.get('offload_model', True),
        'task_id': config.get('task_id'),
        'progress_redis_client': config.get('progress_redis_client'),
        'progress_topic': config.get('redis_progress_topic')
    }

    if "t2v" in task:
        pipeline_class = wan.WanT2V
        generate_args.update({
            'prompt': prompt,
            'size': SIZE_CONFIGS[config.get('size')],
            'frame_num': config.get('frame_num'),
        })
    elif "ti2v" in task:
        pipeline_class = wan.WanTI2V
        generate_args.update({
            'prompt': prompt,
            'img': img,
            'size': SIZE_CONFIGS[config.get('size')],
            'max_area': MAX_AREA_CONFIGS[config.get('size')],
            'frame_num': config.get('frame_num'),
        })
    elif "animate" in task:
        pipeline_class = wan.WanAnimate
        pipeline_args['use_relighting_lora'] = config.get('use_relighting_lora', False)
        generate_args.update({
            'src_root_path': config.get('src_root_path'),
            'replace_flag': config.get('replace_flag', False),
            'refert_num': config.get('refert_num', 77),
            'clip_len': config.get('frame_num'),
        })
    elif "s2v" in task:
        pipeline_class = wan.WanS2V
        generate_args.update({
            'input_prompt': prompt,
            'ref_image_path': config.get('image'),
            'audio_path': config.get('audio'),
            'enable_tts': config.get('enable_tts', False),
            # ... add other s2v-specific args ...
            'max_area': MAX_AREA_CONFIGS[config.get('size')],
            'infer_frames': config.get('infer_frames', 80),
            'init_first_frame': config.get('start_from_ref', False),
        })
    elif "i2v" in task:
        pipeline_class = wan.WanI2V
        generate_args.update({
            'input_prompt': prompt,
            'img': img,
            'max_area': MAX_AREA_CONFIGS[config.get('size')],
            'frame_num': config.get('frame_num'),
        })
    else:
        raise NotImplementedError(f"Task '{task}' is not supported in this refactored logic.")

    logging.info(f"Creating {pipeline_class.__name__} pipeline.")
    pipeline = pipeline_class(**pipeline_args)
    
    logging.info("Generating video...")
    video_tensor = pipeline.generate(**generate_args)

    # --- Cleanup and Return ---
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    logging.info(f"[{rank}] Generation finished.")

    # Only rank 0 holds the final decoded video tensor
    if rank == 0:
        return video_tensor, model_cfg
    else:
        return None, None