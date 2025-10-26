# serve_app.py (最终实现版)
# 本文件定义了完整的 Ray Serve 应用，是新架构下的主程序入口。

import logging
import os
import uuid
from typing import Dict

import ray
import redis
import torch
import torch.distributed as dist
from ray import serve
from starlette.requests import Request

# 1. 从重构后的模块中导入核心生成函数
from wan.core_logic import run_generation_task

# --- 全局配置 (在生产环境中应通过环境变量或配置文件管理) ---
# 分布式推理所需的总进程数 (e.g., 2个节点 x 2卡/节点 = 4个进程)
TOTAL_WORKERS = int(os.environ.get("TOTAL_WORKERS", "4"))

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
PROGRESS_TOPIC = "wan22-progress-stream"


# --- 分布式工作组 (核心计算单元) ---
@serve.deployment(
    name="VideoGenerator",
    num_replicas=TOTAL_WORKERS,
    # 每个副本/Actor都需要1个GPU
    ray_actor_options={"num_gpus": 1}
)
class VideoGenerator:
    """
    一个由多个 Actor 组成的分布式工作组。
    模型在这里被加载并常驻显存，随时准备执行推理任务。
    """
    def __init__(self):
        # 这个构造函数会在每个 Actor 副本启动时被调用。
        # Ray 会确保所有副本的 __init__ 都完成后，服务才开始对外提供。

        # 关键步骤1: 初始化 torch.distributed 分布式环境
        # Ray 会自动为这个 Deployment 中的所有 Actor 注入 RANK, WORLD_SIZE 等环境变量，
        # 因此我们可以像在 PyTorchJob 中一样，直接使用 "env://" 方法进行初始化。
        # 这完美地替代了 torchrun 的功能。
        dist.init_process_group(backend="nccl", init_method="env://")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)

        logging.info(f"[Rank {self.rank}] Distributed group initialized. World size: {self.world_size}.")

        # 关键步骤2: 加载模型 (一次性操作)
        # 在这里执行您的一次性、耗时的模型加载和 FSDP 封装等操作。
        # self.model = load_model_and_wrap_with_fsdp(...)
        logging.info(f"[Rank {self.rank}] Model loaded into GPU memory.")

        # 关键步骤3: 初始化 Redis 客户端
        try:
            self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
            self.redis_client.ping()
        except Exception as e:
            logging.error(f"[Rank {self.rank}] Could not connect to Redis: {e}")
            self.redis_client = None

    def generate_task(self, task_config: Dict) -> None:
        """
        执行一次完整的生成任务。这个方法将被 APIEntrypoint 远程调用。
        Ray Serve 的 broadcast 功能会确保所有 Actor 同时执行此方法。
        """
        try:
            # 为核心逻辑函数补充 Redis 和分布式信息
            task_config['progress_redis_client'] = self.redis_client
            task_config['redis_progress_topic'] = PROGRESS_TOPIC

            # 调用我们重构好的、纯粹的核心生成逻辑
            video_tensor, model_cfg = run_generation_task(
                self.rank, self.world_size, self.local_rank, task_config
            )

            # 只有主进程 (rank 0) 会收到返回的视频张量，并负责后续处理
            if self.rank == 0 and video_tensor is not None:
                logging.info(f"[Rank 0] Generation complete for task {task_config.get('task_id')}. Starting post-processing...")
                
                # 在这里，调用后处理逻辑 (保存文件、上传S3等)
                # final_url = post_process_and_upload(video_tensor, model_cfg, task_config)
                final_url = f"s3://{task_config.get('s3_bucket')}/{task_config.get('s3_key_prefix')}.mp4"

                # 任务成功后，由主进程发送最终成功消息
                if self.redis_client:
                    final_message = {
                        "task_id": task_config.get('task_id'),
                        "status": "COMPLETED",
                        "progress": "100",
                        "url": final_url
                    }
                    self.redis_client.xadd(PROGRESS_TOPIC, final_message)
                    logging.info(f"[Rank 0] Published completion message for task {task_config.get('task_id')}.")

        except Exception as e:
            logging.error(f"[{self.rank}] Task {task_config.get('task_id')} failed: {e}")
            # 任务失败后，由主进程发送失败消息
            if self.rank == 0 and self.redis_client:
                error_message = {"task_id": task_config.get('task_id'), "status": "FAILED", "error": str(e)}
                self.redis_client.xadd(PROGRESS_TOPIC, error_message)

# --- API 入口 (轻量级网关) ---
@serve.deployment(name="APIEntrypoint")
class APIEntrypoint:
    """
    接收来自 Go 控制平面的 HTTP 请求，并将其分发给后台的分布式工作组。
    """
    def __init__(self, generator_handle: "ray.serve.handle.DeploymentHandle"):
        self.generator_handle = generator_handle

    async def generate(self, request: Request) -> Dict:
        """
        主 API 端点。它启动一个后台生成任务，并立即返回一个任务 ID。
        """
        try:
            task_config = await request.json()
            task_id = task_config.get("task_id", str(uuid.uuid4()))
            task_config["task_id"] = task_id

            logging.info(f"APIEntrypoint: Received task, assigning task_id: {task_id}")

            # 关键：向 VideoGenerator 工作组的所有 Actor 广播任务
            # broadcast() 方法会异步地在所有 Actor 上调用指定的方法，且是非阻塞的。
            self.generator_handle.options(method_name="generate_task").broadcast(task_config)

            # 立即向客户端（Go 控制平面）响应
            return {"status": "task_started", "task_id": task_id}
        except Exception as e:
            logging.error(f"APIEntrypoint error: {e}")
            return {"status": "error", "message": str(e)}

# --- 应用构建 ---
# 将 API 入口和分布式工作组绑定在一起，构建成一个完整的 Ray Serve 应用
# Ray Serve 会自动创建一个名为 `generator_handle` 的句柄并传递给 APIEntrypoint
app = APIEntrypoint.bind(VideoGenerator.bind())