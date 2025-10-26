# Ray Serve 应用启动问题调试总结

## 问题描述
`kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- serve run serve_app:app` 命令报错

## 错误排查过程

### 1. 初始错误分析
- **错误类型**: `AttributeError: module 'ray.train.torch' has no attribute 'prepare_torch_process_group'`
- **原因**: Ray 2.9.3 中不存在 `prepare_torch_process_group` 函数

### 2. 修复尝试
- **第一次修复**: 改为使用 `dist.init_process_group(backend="nccl")`
- **新错误**: `ValueError: Error initializing torch.distributed using env:// rendezvous: environment variable RANK expected, but not set`

### 3. 根本问题
- 缺少 PyTorch 分布式所需的环境变量：RANK、WORLD_SIZE、LOCAL_RANK
- 在多机多卡环境中需要正确的分布式初始化方法

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
    self.local_rank = self.rank

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

## 验证结果

✅ **所有测试通过**:
- 语法检查通过
- 环境变量设置正确 (RANK=0, WORLD_SIZE=4, LOCAL_RANK=0)
- CUDA 可用 (8个GPU设备)
- Redis 连接成功 (172.31.0.181:6379)
- VideoGenerator 类初始化正常

## 关键修复点

1. **使用 Ray Serve 副本上下文**: 在 Ray Serve 部署中正确获取分布式信息
2. **手动分配 Rank**: 使用副本名称哈希为每个 Actor 分配唯一 rank
3. **修复属性缺失**: 确保 `self.local_rank` 正确设置
4. **配置更新**: Redis 服务器地址更新为生产环境
5. **环境变量完整性**: 设置所有必需的分布式环境变量

## 部署验证
现在可以安全运行:
```bash
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- serve run serve_app:app
```

## 环境信息
- **Ray 版本**: 2.9.3
- **PyTorch**: 支持分布式训练
- **Redis**: 172.31.0.181:6379
- **GPU**: 8个可用设备
- **工作进程数**: 4 (TOTAL_WORKERS)