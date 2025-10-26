# serve_app.py
# This file defines the Ray Serve application, including the API entrypoint
# and the distributed worker group for video generation.

import asyncio
import logging
import os
import uuid
from typing import Dict

import ray
import redis
from ray import serve
from starlette.requests import Request

# Import the refactored core logic
from wan.core_logic import run_generation_task

# --- Constants ---
# These should be loaded from a config file in a real application
DIST_GROUP_SIZE = 2  # Number of nodes/pods for one distributed job
GPUS_PER_NODE = 2    # Number of GPUs per node/pod
TOTAL_WORKERS = DIST_GROUP_SIZE * GPUS_PER_NODE

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
PROGRESS_TOPIC = "wan22-progress-stream"


@serve.deployment(
    name="VideoGenerator",
    num_replicas=TOTAL_WORKERS,
    ray_actor_options={"num_gpus": 1}
)
class VideoGenerator:
    """The distributed worker group that performs the actual generation."""

    def __init__(self):
        # This __init__ method is called for each replica (actor).
        # Ray ensures that all actors in this deployment are ready before serving requests.
        
        # 1. Initialize torch.distributed group
        # This is the magic that replaces torchrun. Ray handles the setup.
        # Note: This is a conceptual example. The exact API might vary slightly
        # with `ray.train.torch.prepare_actor` or similar utilities.
        # For simplicity, we assume environment variables are set for `init_process_group`
        # In a real Ray setup, this is handled more elegantly.
        logging.info(f"Initializing torch.distributed for worker rank: {ray.get_rank()}")
        # setup_torch_distributed() # This would be a helper function

        # 2. Load the model (once per actor)
        # This ensures the model is always warm in GPU memory.
        # self.model = load_model() 
        logging.info(f"Model loaded on worker rank: {ray.get_rank()}")

        # 3. Initialize Redis client for this actor
        try:
            self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
            self.redis_client.ping()
        except Exception as e:
            logging.error(f"Worker {ray.get_rank()} could not connect to Redis: {e}")
            self.redis_client = None

    def generate_task(self, task_config: Dict) -> None:
        """
        This method is called remotely to start a generation task.
        It runs the core logic and reports progress.
        """
        rank = ray.get_rank()
        world_size = TOTAL_WORKERS # Ray provides ways to get this dynamically
        local_rank = rank % GPUS_PER_NODE

        task_id = task_config.get("task_id")
        logging.info(f"[{rank}] Starting generation for task_id: {task_id}")

        # Add redis info to the config for the core logic function
        task_config['progress_redis_client'] = self.redis_client
        task_config['redis_progress_topic'] = PROGRESS_TOPIC

        try:
            # Call the modular, core generation logic
            # This is a long-running, blocking call within this actor
            run_generation_task(rank, world_size, local_rank, task_config)

            # After completion, publish a final "success" message
            if rank == 0 and self.redis_client:
                final_message = {
                    "task_id": task_id,
                    "status": "COMPLETED",
                    "progress": "100",
                    "url": f"s3://{task_config.get('s3_bucket')}/{task_config.get('s3_key_prefix')}/final.mp4" # Example URL
                }
                self.redis_client.xadd(PROGRESS_TOPIC, final_message)

        except Exception as e:
            logging.error(f"[{rank}] Task {task_id} failed: {e}")
            # Publish a "failure" message
            if rank == 0 and self.redis_client:
                error_message = {
                    "task_id": task_id,
                    "status": "FAILED",
                    "error": str(e)
                }
                self.redis_client.xadd(PROGRESS_TOPIC, error_message)


@serve.deployment(name="APIEntrypoint")
class APIEntrypoint:
    """The lightweight HTTP server that accepts requests from the Go backend."""

    def __init__(self, generator_handle: "ray.serve.handle.DeploymentHandle"):
        self.generator_handle = generator_handle

    async def generate(self, request: Request) -> Dict:
        """
        The main API endpoint. It starts a generation task in the background
        and immediately returns a task ID.
        """
        try:
            task_config = await request.json()
            task_id = str(uuid.uuid4())
            task_config["task_id"] = task_id

            logging.info(f"Received task. Assigning task_id: {task_id}")

            # CRITICAL: Call the distributed worker group asynchronously.
            # .remote() is non-blocking and returns immediately.
            # The task will be executed in the background by the VideoGenerator actors.
            self.generator_handle.generate_task.remote(task_config)

            # Immediately respond to the client (the Go backend)
            return {"status": "task_queued", "task_id": task_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Build the application by binding the deployments together.
# The APIEntrypoint gets a handle to call the VideoGenerator.
app = APIEntrypoint.bind(VideoGenerator.bind())
