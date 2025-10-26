# RayService 测试和视频生成指南

## 当前部署状态

### RayService 部署
- **名称**: `wan22-rayservice`
- **状态**: `WaitForServeDeploymentReady` (自动部署失败，手动部署成功)
- **Head Pod**: `wan22-rayservice-raycluster-lwqd5-head-vqln2`
- **Worker Pods**: 2个 GPU worker
- **Service IP**: `172.30.9.54` (集群内部)

### RayCluster 部署 (对比)
- **名称**: `wan22-ray-cluster`
- **状态**: 稳定运行
- **Head Pod**: `wan22-ray-cluster-head-2tcbr`

## 测试方法

### 方法1: 通过 Pod 直接测试 (推荐)

#### 测试简化 API (当前 RayService 配置)
```bash
# 测试简化版本的 API
kubectl -n hu exec -i -t wan22-rayservice-raycluster-lwqd5-head-vqln2 -- python -c "
import requests
import json

try:
    response = requests.post('http://localhost:8000/generate', 
                           json={'task_id': 'test123'})
    print(f'Status Code: {response.status_code}')
    print(f'Response: {response.text}')
except Exception as e:
    print(f'Error: {e}')
"
```

#### 测试完整视频生成 API (RayCluster)
```bash
# 测试完整功能的视频生成
kubectl -n hu exec -i -t wan22-ray-cluster-head-2tcbr -- python -c "
import requests
import json

# 完整的视频生成请求
video_request = {
    'task_id': 'video_test_001',
    'prompt': 'A beautiful sunset over mountains',
    'duration': 5,
    'resolution': '512x512'
}

try:
    response = requests.post('http://localhost:8000/generate', 
                           json=video_request)
    print(f'Status Code: {response.status_code}')
    print(f'Response: {response.text}')
except Exception as e:
    print(f'Error: {e}')
"
```

### 方法2: 通过端口转发测试

#### 端口转发到本地
```bash
# 转发 RayService 的 8000 端口到本地
kubectl -n hu port-forward wan22-rayservice-raycluster-lwqd5-head-vqln2 8000:8000 &

# 转发 RayCluster 的 8000 端口到本地 (使用不同端口)
kubectl -n hu port-forward wan22-ray-cluster-head-2tcbr 8001:8000 &
```

#### 本地测试命令
```bash
# 测试 RayService (简化版本)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"task_id": "test_rayservice"}'

# 测试 RayCluster (完整版本)
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "video_test_001",
    "prompt": "A beautiful sunset over mountains",
    "duration": 5,
    "resolution": "512x512"
  }'
```

### 方法3: 创建外部服务访问

#### 创建 NodePort 服务
```yaml
apiVersion: v1
kind: Service
metadata:
  name: rayservice-external
  namespace: hu
spec:
  type: NodePort
  selector:
    ray.io/node-type: head
    ray.io/cluster-name: wan22-rayservice-raycluster-lwqd5
  ports:
    - name: serve
      port: 8000
      targetPort: 8000
      nodePort: 30080
```

#### 应用服务配置
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: rayservice-external
  namespace: hu
spec:
  type: NodePort
  selector:
    ray.io/node-type: head
    ray.io/cluster-name: wan22-rayservice-raycluster-lwqd5
  ports:
    - name: serve
      port: 8000
      targetPort: 8000
      nodePort: 30080
EOF
```

#### 通过 NodePort 测试
```bash
# 获取节点 IP
kubectl -n hu get nodes -o wide

# 测试 API (替换 NODE_IP 为实际节点 IP)
curl -X POST http://NODE_IP:30080/generate \
  -H "Content-Type: application/json" \
  -d '{"task_id": "external_test"}'
```

## 视频生成测试用例

### 基础测试用例

#### 1. 简单文本到视频
```json
{
  "task_id": "t2v_test_001",
  "prompt": "A cat playing with a ball of yarn",
  "duration": 3,
  "resolution": "512x512"
}
```

#### 2. 图像到视频
```json
{
  "task_id": "i2v_test_001",
  "image_url": "https://example.com/input_image.jpg",
  "prompt": "Make the image come to life",
  "duration": 4
}
```

#### 3. 语音到视频
```json
{
  "task_id": "s2v_test_001",
  "audio_url": "https://example.com/input_audio.wav",
  "duration": 5
}
```

### 高级测试用例

#### 4. 复杂场景视频
```json
{
  "task_id": "complex_test_001",
  "prompt": "A futuristic city with flying cars and neon lights, cinematic quality",
  "duration": 8,
  "resolution": "768x768",
  "style": "cinematic",
  "fps": 24
}
```

#### 5. 批量生成测试
```json
{
  "task_id": "batch_test_001",
  "prompts": [
    "A peaceful forest with flowing river",
    "A bustling city street at night",
    "Underwater coral reef with colorful fish"
  ],
  "duration": 4,
  "batch_size": 3
}
```

## 监控和调试

### 检查服务状态
```bash
# 检查 RayService 状态
kubectl -n hu get rayservice

# 检查 Pod 状态
kubectl -n hu get pods | grep ray

# 检查 Serve 应用状态
kubectl -n hu exec -i -t wan22-rayservice-raycluster-lwqd5-head-vqln2 -- serve status

# 检查 Ray 集群状态
kubectl -n hu exec -i -t wan22-rayservice-raycluster-lwqd5-head-vqln2 -- ray status
```

### 查看日志
```bash
# 查看 Head Pod 日志
kubectl -n hu logs wan22-rayservice-raycluster-lwqd5-head-vqln2

# 查看 Worker Pod 日志
kubectl -n hu logs wan22-rayservice-raycluster-lwqd5-gpu-workers-worker-c9tqt

# 实时查看日志
kubectl -n hu logs -f wan22-rayservice-raycluster-lwqd5-head-vqln2
```

### 性能监控
```bash
# 检查 GPU 使用情况
kubectl -n hu exec -i -t wan22-rayservice-raycluster-lwqd5-head-vqln2 -- nvidia-smi

# 检查内存使用
kubectl -n hu top pods | grep ray
```

## 故障排除

### 常见问题

#### 1. API 端点不可用
- 检查 Serve 应用是否部署成功
- 验证端口是否正确暴露
- 检查防火墙和网络策略

#### 2. 视频生成失败
- 检查 GPU 资源是否充足
- 验证模型文件是否存在
- 检查输入数据格式

#### 3. 性能问题
- 监控 GPU 使用率
- 检查内存使用情况
- 优化批处理大小

#### 4. 分布式协调问题
- 检查 Redis 连接
- 验证分布式环境变量
- 确认 NCCL 配置

### 调试命令
```bash
# 进入容器调试
kubectl -n hu exec -it wan22-rayservice-raycluster-lwqd5-head-vqln2 -- bash

# 检查环境变量
kubectl -n hu exec -i -t wan22-rayservice-raycluster-lwqd5-head-vqln2 -- env | grep -E "(RANK|WORLD|CUDA|NCCL)"

# 测试 Redis 连接
kubectl -n hu exec -i -t wan22-rayservice-raycluster-lwqd5-head-vqln2 -- python -c "import redis; r = redis.Redis(host='172.31.0.181', port=6379); print(r.ping())"
```

## 最佳实践

1. **逐步测试**: 从简单请求开始，逐步增加复杂度
2. **监控资源**: 实时监控 GPU 和内存使用
3. **错误处理**: 实现完善的错误处理和重试机制
4. **日志记录**: 详细记录请求和响应信息
5. **性能优化**: 根据硬件配置调整批处理大小和并发数

## 注意事项

- RayService 自动部署存在 YAML 解析问题，当前使用手动部署
- 生产环境建议使用 LoadBalancer 或 Ingress 进行外部访问
- 视频生成是计算密集型任务，确保有足够的 GPU 资源
- 定期监控服务状态和资源使用情况