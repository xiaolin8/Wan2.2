# 部署与测试指南：Ray Serve 分布式推理服务

本文档提供了一个分步指南，用于部署、测试和验证我们新设计的基于 Ray Serve 的视频生成服务。

## 0. 先决条件

在开始之前，请确保您已准备好以下环境：

1.  一个可用的、配置好 `kubectl` 的 Kubernetes 集群，且包含 GPU 节点。
2.  集群中已成功部署 `kuberay-operator:v1.2.1`。
3.  集群中已成功部署 `KEDA`。
4.  一个正在运行的 Redis 服务，并确保其地址可从 K8s 集群内部访问。
5.  本地已安装 `docker` 和 `kubectl` 命令行工具。

---

## 1. 构建并推送 Docker 镜像

我们的 `RayCluster` 将使用一个包含所有最新代码和依赖的镜像来启动 Pod。 

```bash
# 1. 进入您的项目根目录 (wan2.2)
cd /path/to/your/wan2.2

# 2. 使用我们新创建的 Dockerfile.ray 来构建镜像
# 将 your-registry/wan22-ray:latest 替换为您自己的镜像仓库地址和标签
docker build -t your-registry/wan22-ray:latest -f Dockerfile.ray .

# 3. 推送镜像到您的镜像仓库
docker push your-registry/wan22-ray:latest
```

**重要**: 推送成功后，请打开 `k8s/raycluster.yaml` 文件，将其中 `image` 字段的值，从占位符 `your-registry/wan22-ray:latest` 修改为您刚刚推送的实际镜像地址。

---

## 2. 部署 Kubernetes 核心资源

现在，我们将创建 Ray 集群运行所需的基础服务和集群本身。

```bash
# 1. 应用 Headless Service，为 Ray Worker 提供服务发现
kubectl apply -f k8s/wan22-worker-headless-service.yaml

# 2. 应用 RayCluster 定义
# kuberay-operator 将会监听到这个操作，并开始创建 Ray Head Pod
kubectl apply -f k8s/raycluster.yaml
```

**验证**: 
执行 `kubectl get pods`，您应该能看到一个名为 `wan22-ray-cluster-head-...` 的 Pod 被创建并进入 `Running` 状态。此时，由于 `raycluster.yaml` 中 worker 的 `replicas` 设置为 0，不应该有任何 worker pod 被创建。

---

## 3. 部署 Ray Serve 应用

Ray 集群的“操作系统”已经启动，现在我们需要把我们的 Python 应用（`serve_app.py`）部署上去。

```bash
# 1. 找到您的 Ray Head Pod 的完整名称
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o=jsonpath='{.items[0].metadata.name}')

# 2. 在 Head Pod 中，使用 Ray Serve 的 CLI 来部署我们的应用
# 这会将 serve_app.py 中的代码部署到整个 Ray 集群
kubectl exec -it $HEAD_POD -- python -m serve run serve_app:app
```

**验证**: 
部署命令成功后，您可以访问 Ray Dashboard（通过 `kubectl port-forward $HEAD_POD 8265:8265`），在 "Serve" 标签页下，您应该能看到名为 `APIEntrypoint` 和 `VideoGenerator` 的两个 deployment，并且 `VideoGenerator` 应该有我们定义的多个副本（replicas）。

---

## 4. 激活 KEDA 自动伸缩

万事俱备，现在我们部署“哨兵” KEDA，让它开始监控任务队列并自动管理我们的 Ray Worker 数量。

**注意**: 在应用前，请确保 `k8s/keda-scaledobject.yaml` 文件中的 Redis 地址 (`metadata.address`) 是正确的。

```bash
# 应用 KEDA ScaledObject
kubectl apply -f k8s/keda-scaledobject.yaml
```

此时，整个系统已经“上线”并处于待命状态。由于 Redis 任务队列中没有任务，KEDA 会确保 Ray Worker 的数量为 0。

---

## 5. 端到端触发与测试

现在，我们将模拟您的 Go 控制平面，向 Ray Serve 发送一个 HTTP 请求来触发整个流程。

```bash
# 1. 为了方便从本地访问，将 Ray Serve 的 HTTP 端口转发到本地
# Ray Serve 会自动创建一个名为 serve-api-entrypoint-head 的 Service
# (注意：Service 名称可能因您的 serve_app.py 中的部署名称而异)
kubectl port-forward service/wan22-ray-cluster-head-svc 8000:8000

# 2. 在另一个终端中，使用 curl 模拟 Go 控制平面发送一个任务请求
curl -X POST -H "Content-Type: application/json" \
-d '{
  "task": "i2v-A14B",
  "prompt": "a beautiful cat",
  "ckpt_dir": "/data/Wan2.2-I2V-A14B",
  "image": "/path/in/pod/to/image.jpg",
  "size": "832*480"
}' \
http://localhost:8000/generate
```

**观察与验证**: 
在 `curl` 命令发送后，预期的行为将会依次发生：

1.  `curl` 命令会**立即**收到一个包含 `task_id` 的 JSON 响应。
2.  几乎在同时，Go 控制平面（如果已连接到 Redis）会收到一个入队消息。
3.  KEDA 检测到队列中有任务，会立即将 `RayCluster` 的 worker 副本数从 0 调整为 N。
4.  `kubectl get pods` 会看到新的 `wan22-ray-cluster-worker-...` Pod 开始被创建。
5.  Pod 启动并加载模型后，开始执行任务。
6.  您的 Redis 客户端如果订阅了 `wan22-progress-stream`，将会开始收到实时的进度消息。
7.  任务完成后，您会在 S3 存储桶中看到生成的视频文件。
8.  一段时间后，如果您不再提交新任务，KEDA 会将 Worker Pod 的数量重新缩减回 0。

至此，您已成功部署并验证了整套动态伸缩的分布式推理服务架构。
