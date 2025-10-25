import os
import torch
import torch.distributed as dist
from ts.torch_handler.base_handler import BaseHandler

# 假设你的视频生成模型相关代码在 MyDistributedVideoModel 类中
# from my_model_package import MyDistributedVideoModel

class VideoGenerationHandler(BaseHandler):
    """
    TorchServe handler for distributed multi-GPU video generation.
    """
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None
        self.rank = -1
        self.world_size = -1
        self.master_addr = None
        self.master_port = None
        self.device = None
        self.is_master = False # 用于判断是否为主节点

    def initialize(self, context):
        """
        Initializes the model and sets up the distributed environment.
        This method is called once per worker.
        """
        if self.initialized:
            return

        # 1. 获取分布式环境变量
        self.rank = int(os.environ.get("RANK", -1))
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))
        self.master_addr = os.environ.get("MASTER_ADDR")
        self.master_port = os.environ.get("MASTER_PORT")
        self.device = torch.device(f"cuda:{self.rank}")
        self.is_master = (self.rank == 0)

        if self.rank == -1 or self.world_size == -1 or not self.master_addr or not self.master_port:
            raise RuntimeError("Distributed environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) not set!")

        print(f"[{self.rank}/{self.world_size}] Initializing distributed process group "
              f"with MASTER_ADDR={self.master_addr}, MASTER_PORT={self.master_port}")

        # 2. 初始化分布式环境
        # 注意: 如果 NCCL_SOCKET_IFNAME 或 NCCL_IB_HCA 未在 entrypoint.sh 或 K8s Env 中指定
        # 可能会在这里导致 NCCL 回退到 TCP 或报错
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://{self.master_addr}:{self.master_port}"
        )
        print(f"[{self.rank}/{self.world_size}] Distributed process group initialized.")
        torch.cuda.set_device(self.rank) # 设置当前进程使用的 GPU

        # 3. 加载分布式模型 (替换为你的模型加载逻辑)
        # 假设你的模型 MyDistributedVideoModel 已经处理了 FSDP 包装
        # 它会在加载时从 S3 或挂载的 PV 读取权重
        if self.is_master:
            print(f"[{self.rank}] Loading distributed model...")
        # self.model = MyDistributedVideoModel.load_from_checkpoint(
        #     self.manifest['model']['uri'],
        #     map_location=self.device
        # ).to(self.device).eval()
        # 假设这里是你的模型加载代码
        self.model = torch.nn.Linear(10,10).to(self.device) # 替换为你的实际模型
        if self.is_master:
            print(f"[{self.rank}] Distributed model loaded.")

        dist.barrier() # 等待所有进程加载完成

        self.initialized = True
        print(f"[{self.rank}] Handler initialization complete.")

    def preprocess(self, data):
        """
        Preprocesses the input data (e.g., parse prompt, parameters).
        """
        # data 是一个列表，每个元素代表一个请求
        input_data = []
        for row in data:
            prompt = row['data'].get("prompt") # 从请求中获取 prompt
            params = row['data'].get("params", {}) # 获取其他参数
            # 你的预处理逻辑，例如 prompt tokenization
            input_data.append({"prompt": prompt, "params": params})
        return input_data

    def inference(self, model_input):
        """
        Performs the model inference.
        Note: Model input is already a list (batch) from preprocess results.
        Master (rank=0) will receive all batched inputs.
        Then you need to distribute them to other ranks if your model expects it.
        Or the FSDP model itself handles the distribution.
        """
        if not self.initialized:
            raise RuntimeError("Handler not initialized. Please call initialize() first.")

        # 假设你的模型会自动处理 FSDP 的分布式推理
        # Master (rank=0) 接收 batch inputs，然后 FSDP 内部会协调
        if self.is_master:
            print(f"[{self.rank}] Performing inference for batch size {len(model_input)} prompts...")
            # 示例：你的模型推理逻辑
            # video_output = self.model.generate(model_input)
            video_output = [f"video_url_for_{item['prompt']}" for item in model_input]
            print(f"[{self.rank}] Inference complete.")
        else:
            # Worker (rank >0) 仅参与 FSDP 的计算，不直接接收请求
            # 确保 FSDP 内部的通信机制能让 worker 参与计算
            video_output = None # Worker 可能只返回 None 或部分结果，Master 负责聚合
            # 示例：等待 Master 广播任务或通过 FSDP 参与计算
            # self.model.generate_worker_part()

        dist.barrier() # 等待所有进程完成推理部分

        return video_output # Master 返回聚合结果，Workers 返回 None 或部分结果

    def postprocess(self, inference_output):
        """
        Postprocesses the inference output (e.g., format video URLs).
        Only master (rank=0) returns the actual results.
        """
        if self.is_master:
            # 假设 inference_output 是一个列表 (批处理结果)
            results = []
            for item in inference_output:
                results.append({"generated_video_url": item}) # 替换为你的实际输出格式
            return results
        else:
            return [] # Workers 不返回结果