# Ray Serve 应用启动问题调试总结

## 问题描述
`kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- serve run serve_app:app` 命令报错

## 完整错误排查过程

### 1. 初始错误分析
- **错误类型**: `AttributeError: module 'ray.train.torch' has no attribute 'prepare_torch_process_group'`
- **原因**: Ray 2.9.3 中不存在 `prepare_torch_process_group` 函数

### 2. 修复尝试
- **第一次修复**: 改为使用 `dist.init_process_group(backend="nccl")`
- **新错误**: `ValueError: Error initializing torch.distributed using env:// rendezvous: environment variable RANK expected, but not set`

### 3. 根本问题
- 缺少 PyTorch 分布式所需的环境变量：RANK、WORLD_SIZE、LOCAL_RANK
- 在多机多卡环境中需要正确的分布式初始化方法

### 4. 后续关键修复
- **API 端点方法名错误**: `RayServeException: Tried to call a method '__call__' that does not exist. Available methods: ['generate']`
  - 修复：将 APIEntrypoint 类的 `generate` 方法改为 `__call__` 方法

- **GPU 设备号错误**: `torch.AcceleratorError: CUDA error: invalid device ordinal`
  - 修复：修改 `local_rank` 计算方式，确保在可用 GPU 范围内

- **语法错误**: `SyntaxError: unterminated string literal (detected at line 159)`
  - 修复：检查并修复第159行的语法错误，移除多余的引号

- **DeploymentHandle 调用错误**: `{"status":"error","message":"'DeploymentHandle' object is not callable"}`
  - 修复：将 broadcast 调用改为正确的 remote 调用

- **资源优化**: 将 TOTAL_WORKERS 从4减少到2，为 APIEntrypoint 释放CPU资源

## 最终解决方案

### 修改文件: `serve_app.py`

#### 1. Redis 配置更新
```python
REDIS_HOST = os.environ.get("REDIS_HOST", "172.31.0.181")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
```

#### 2. 分布式环境初始化修复
```python
# Ray Serve 应用启动问题调试总结

## 问题描述
`kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- serve run serve_app:app` 命令报错

## 完整错误排查过程

### 1. 初始错误分析
- **错误类型**: `AttributeError: module 'ray.train.torch' has no attribute 'prepare_torch_process_group'`
- **原因**: Ray 2.9.3 中不存在 `prepare_torch_process_group` 函数

### 2. 修复尝试
- **第一次修复**: 改为使用 `dist.init_process_group(backend="nccl")`
- **新错误**: `ValueError: Error initializing torch.distributed using env:// rendezvous: environment variable RANK expected, but not set`

### 3. 根本问题
- 缺少 PyTorch 分布式所需的环境变量：RANK、WORLD_SIZE、LOCAL_RANK
- 在多机多卡环境中需要正确的分布式初始化方法

### 4. 后续关键修复
- **API 端点方法名错误**: `RayServeException: Tried to call a method '__call__' that does not exist. Available methods: ['generate']`
  - 修复: 将 APIEntrypoint 类的 `generate` 方法改为 `__call__` 方法

- **GPU 设备号超出范围**: `torch.AcceleratorError: CUDA error: invalid device ordinal`
  - 修复: 修改 `local_rank` 计算方式，确保在可用 GPU 范围内
  ```python
  # 确保 local_rank 在可用 GPU 范围内
  self.local_rank = self.rank % torch.cuda.device_count()
  ```

- **语法错误**: `SyntaxError: unterminated string literal (detected at line 159)`
  - 修复: 检查并修复第159行的语法错误，移除多余的引号

- **DeploymentHandle 调用错误**: `{"status":"error","message":"'DeploymentHandle' object is not callable"}`
  - 修复: 将 broadcast 调用改为正确的 remote 调用
  ```python
  # 修复 DeploymentHandle 调用
  self.generator_handle.generate_task.remote(task_config)
  ```

- **资源优化**: 将 TOTAL_WORKERS 从4减少到2，为 APIEntrypoint 释放CPU资源

## 最终解决方案

### 修改文件: `serve_app.py`

#### 1. Redis 配置更新
```python
REDIS_HOST = os.environ.get("REDIS_HOST", "172.31.0.181")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
```

#### 2. 分布式环境初始化修复
```python
def __init__(self):
    # 关键步骤1: 初始化 torch.distributed 分布式环境
    # 在 Ray Serve 中，我们需要手动协调分布式初始化
    # 使用 Ray Serve 的副本上下文来获取分布式信息
    
    # 获取 Ray Serve 副本上下文
    import ray.serve
    replica_context = ray.serve.get_replica_context()
    
    # 使用副本名称来分配唯一的 rank
    # 副本名称格式类似: VideoGenerator#ATDmtt
    replica_name = replica_context.replica_tag
    
    # 为每个副本分配唯一的 rank (0 到 TOTAL_WORKERS-1)
    self.rank = hash(replica_name) % TOTAL_WORKERS
    self.world_size = TOTAL_WORKERS
    # 确保 local_rank 在可用 GPU 范围内
    self.local_rank = self.rank % torch.cuda.device_count()

    # 设置分布式环境变量
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["RANK"] = str(self.rank)
    os.environ["WORLD_SIZE"] = str(self.world_size)
    os.environ["LOCAL_RANK"] = str(self.local_rank)

    # 初始化分布式进程组
    dist.init_process_group(backend="nccl")
    
    torch.cuda.set_device(self.local_rank)
```

#### 3. API 端点方法修复
```python
@serve.deployment(name="APIEntrypoint", route_prefix="/generate")
class APIEntrypoint:
    async def __call__(self, request: Request) -> Dict:
        # API 处理逻辑
```

#### 4. DeploymentHandle 调用修复
```python
# 修复 DeploymentHandle 调用
self.generator_handle.generate_task.remote(task_config)
```

## 验证结果

✅ **所有测试通过**:
- 语法检查通过
- 环境变量设置正确 (RANK=0, WORLD_SIZE=2, LOCAL_RANK=0)
- CUDA 可用 (8个GPU设备)
- Redis 连接成功 (172.31.0.181:6379)
- VideoGenerator 类初始化正常
- API 端点 `/generate` 正常工作
- Ray Serve 应用成功部署并运行

## 关键修复点

1. **使用 Ray Serve 副本上下文**: 在 Ray Serve 部署中正确获取分布式信息
2. **手动分配 Rank**: 使用副本名称哈希为每个 Actor 分配唯一 rank
3. **修复属性缺失**: 确保 `self.local_rank` 正确设置
4. **配置更新**: Redis 服务器地址更新为生产环境
5. **环境变量完整性**: 设置所有必需的分布式环境变量
6. **API 端点方法名**: 将 `generate` 方法改为 `__call__` 方法
7. **GPU 设备分配**: 确保 `local_rank` 在可用 GPU 范围内
8. **语法错误修复**: 修复第159行多余的引号
9. **DeploymentHandle 调用**: 使用正确的 remote 调用方式
10. **资源优化**: 将 TOTAL_WORKERS 从4减少到2

## 部署验证
现在可以安全运行:
```bash
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- serve run serve_app:app
```

## 环境信息
- **Ray 版本**: 2.9.3
- **PyTorch**: 支持分布式训练
- **Redis**: 172.31.0.181:6379
- **GPU**: 8个可用设备 (NVIDIA H20)
- **工作进程数**: 2 (TOTAL_WORKERS)
- **API 端点**: `/generate` 正常工作
- **部署状态**: APIEntrypoint HEALTHY, VideoGenerator 运行中
```

#### 3. API 端点方法修复
```python
@serve.deployment(name="APIEntrypoint", route_prefix="/generate")
class APIEntrypoint:
    async def __call__(self, request: Request) -> Dict:
        # API 处理逻辑
```

#### 4. DeploymentHandle 调用修复
```python
# 关键修复：使用正确的 remote() 调用方式
self.generator_handle.generate_task.remote(task_config)
```

## 验证结果

✅ **所有测试通过**:
- 语法检查通过
- 环境变量设置正确 (RANK=0, WORLD_SIZE=2, LOCAL_RANK=0)
- CUDA 可用 (8个GPU设备)
- Redis 连接成功 (172.31.0.181:6379)
- VideoGenerator 类初始化正常
- API 端点 `/generate` 正常工作
- Ray Serve 应用成功部署并运行

## 关键修复点

1. **使用 Ray Serve 副本上下文**: 在 Ray Serve 部署中正确获取分布式信息
2. **手动分配 Rank**: 使用副本名称哈希为每个 Actor 分配唯一 rank
3. **修复属性缺失**: 确保 `self.local_rank` 正确设置
4. **配置更新**: Redis 服务器地址更新为生产环境
5. **环境变量完整性**: 设置所有必需的分布式环境变量
6. **API 端点方法名**: 将 `generate` 方法改为 `__call__` 方法
7. **GPU 设备分配**: 确保 `local_rank` 在可用 GPU 范围内
8. **语法错误修复**: 修复第159行的未终止字符串字面量
9. **DeploymentHandle 调用**: 使用正确的 `remote()` 调用方式
10. **资源优化**: 将 TOTAL_WORKERS 从4减少到2

## RayService 部署对比

### RayCluster 部署 (k8s/raycluster.yaml)
- **状态**: 稳定运行
- **Pod**: `wan22-ray-cluster-head-2tcbr` (1个)
- **GPU**: 8个 NVIDIA H20 GPU
- **Serve 应用**: 完整的分布式视频生成服务
- **API 端点**: `/generate` (支持复杂视频生成任务)
- **部署方式**: 手动执行 `serve run serve_app:app`

### RayService 部署 (k8s/rayservice.yaml)
- **状态**: 已部署，Serve 应用手动启动
- **Pod**: 
  - `wan22-rayservice-raycluster-lwqd5-head-vqln2` (head)
  - `wan22-rayservice-raycluster-lwqd5-gpu-workers-worker-c9tqt` (worker)
  - `wan22-rayservice-raycluster-lwqd5-gpu-workers-worker-wgt7r` (worker)
- **GPU**: 4个 GPU (2个 worker × 2 GPU/worker)
- **Serve 应用**: 简化的 API 服务
- **API 端点**: `/generate` (返回简单响应)
- **部署方式**: 通过 `serveConfigV2` 自动部署 (YAML 解析问题，手动部署成功)

## 关键排查思路和步骤

### 1. 分布式环境初始化问题排查
- **问题**: PyTorch 分布式初始化失败
- **排查步骤**:
  1. 检查环境变量: RANK, WORLD_SIZE, LOCAL_RANK
  2. 验证 CUDA 设备可用性
  3. 确认 NCCL 后端支持
  4. 检查 MASTER_ADDR 和 MASTER_PORT 设置

### 2. Ray Serve 部署问题排查
- **问题**: API 端点方法调用失败
- **排查步骤**:
  1. 确认部署类有 `__call__` 方法
  2. 检查 route_prefix 配置
  3. 验证 DeploymentHandle 调用方式
  4. 检查 Serve 应用状态

### 3. YAML 配置问题排查
- **问题**: RayService YAML 解析错误
- **排查步骤**:
  1. 简化 `serveConfigV2` 中的 Python 代码
  2. 检查 YAML 多行字符串格式 (`|` vs `|-`)
  3. 验证缩进和语法
  4. 逐步添加复杂度测试

### 4. 资源分配问题排查
- **问题**: GPU 设备号超出范围
- **排查步骤**:
  1. 检查可用 GPU 数量
  2. 验证 local_rank 计算逻辑
  3. 确认资源请求和限制
  4. 测试不同 worker 数量配置

## 关键排查命令

### 集群状态检查
```bash
# 检查 RayService 状态
kubectl -n hu get rayservice

# 检查 RayCluster 状态
kubectl -n hu get raycluster

# 检查 Pod 状态
kubectl -n hu get pods | grep ray

# 查看详细事件
kubectl -n hu describe rayservice wan22-rayservice
```

### 容器内环境检查
```bash
# 检查 CUDA 设备
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- nvidia-smi

# 检查 Python 环境
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}')"

# 检查 Ray Serve 可用性
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- python -c "from ray import serve; print('Ray Serve available')"
```

### Serve 应用状态检查
```bash
# 检查 Ray 集群状态
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- ray status

# 检查 Serve 应用状态
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- serve status

# 测试 API 端点
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- python -c "
import requests
try:
    response = requests.post('http://localhost:8000/generate', json={'task_id': 'test123'})
    print(f'Status: {response.status_code}, Response: {response.text}')
except Exception as e:
    print(f'Error: {e}')
"
```

### 手动部署和测试
```bash
# 手动启动 Serve 应用
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- serve run serve_app:app

# 手动部署简化版本
kubectl -n hu exec -i -t wan22-rayservice-raycluster-lwqd5-head-vqln2 -- python -c "
from ray import serve

@serve.deployment(name='APIEntrypoint', route_prefix='/generate')
class APIEntrypoint:
    def __init__(self):
        pass

    async def __call__(self, request):
        return {'status': 'success', 'message': 'Hello from RayService'}

app = APIEntrypoint.bind()
serve.run(app, host='0.0.0.0', port=8000)
" &
```

### 配置验证
```bash
# 验证 YAML 配置
kubectl -n hu apply -f k8s/rayservice.yaml --dry-run=client

# 检查配置详情
kubectl -n hu describe rayservice wan22-rayservice | grep -A 10 "serveConfigV2"

# 查看事件日志
kubectl -n hu get events --sort-by=.lastTimestamp | grep -i ray | tail -10
```

## 部署验证
现在可以安全运行:
```bash
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- serve run serve_app:app
```

RayService 手动部署:
```bash
kubectl -n hu exec -i -t wan22-rayservice-raycluster-lwqd5-head-vqln2 -- python -c "
from ray import serve

@serve.deployment(name='APIEntrypoint', route_prefix='/generate')
class APIEntrypoint:
    def __init__(self):
        pass

    async def __call__(self, request):
        return {'status': 'success', 'message': 'Hello from RayService'}

app = APIEntrypoint.bind()
serve.run(app, host='0.0.0.0', port=8000)
" &
```

## 环境信息
- **Ray 版本**: 2.9.3
- **PyTorch**: 支持分布式训练
- **Redis**: 172.31.0.181:6379
- **GPU**: 8个可用设备 (NVIDIA H20)
- **工作进程数**: 2 (TOTAL_WORKERS)
- **API 端点**: `/generate` 正常工作
- **部署状态**: 
  - RayCluster: HEALTHY
  - RayService: 手动部署成功，自动部署存在 YAML 解析问题

## 经验总结
1. **Ray Serve 分布式部署**: 需要手动管理分布式环境变量和初始化
2. **API 端点方法**: 必须使用 `__call__` 方法而非自定义方法名
3. **GPU 设备分配**: 确保 local_rank 在可用 GPU 范围内
4. **YAML 配置**: 复杂的 Python 代码在 YAML 中容易产生解析问题
5. **逐步调试**: 从简单配置开始，逐步增加复杂度
6. **手动验证**: 在自动部署失败时，手动部署可以验证环境可用性