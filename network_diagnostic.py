#!/usr/bin/env python3
"""
网络连接诊断脚本
用于检测和诊断 RayJob 环境中分布式 PyTorch 的网络通信问题
"""

import os
import sys
import socket
import subprocess
import json
import time
from pathlib import Path

def check_network_interfaces():
    """检查网络接口"""
    print("=== 网络接口检查 ===")
    try:
        result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'inet ' in line and '127.0.0.1' not in line:
                    print(f"网络接口: {line.strip()}")
        else:
            print("无法获取网络接口信息")
    except Exception as e:
        print(f"网络接口检查失败: {e}")

def check_hostname_resolution():
    """检查主机名解析"""
    print("\n=== 主机名解析检查 ===")
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        print(f"主机名: {hostname}")
        print(f"IP 地址: {ip}")
    except Exception as e:
        print(f"主机名解析失败: {e}")

def check_port_connectivity():
    """检查端口连通性"""
    print("\n=== 端口连通性检查 ===")
    test_ports = [29500, 29501, 29502, 6379, 8265]
    
    for port in test_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print(f"端口 {port}: 开放")
            else:
                print(f"端口 {port}: 关闭")
        except Exception as e:
            print(f"端口 {port}: 检查失败 - {e}")

def check_environment_variables():
    """检查环境变量"""
    print("\n=== 环境变量检查 ===")
    env_vars = [
        'NCCL_SOCKET_IFNAME',
        'NCCL_IB_DISABLE',
        'NCCL_DEBUG',
        'MASTER_ADDR',
        'MASTER_PORT',
        'WORLD_SIZE',
        'RANK',
        'LOCAL_RANK'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        print(f"{var}: {value}")

def test_pytorch_distributed():
    """测试 PyTorch 分布式初始化"""
    print("\n=== PyTorch 分布式测试 ===")
    try:
        import torch
        import torch.distributed as dist
        
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU 数量: {torch.cuda.device_count()}")
        
        # 测试 NCCL 初始化
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0,eno1,ens,ib'
        
        # 尝试初始化分布式
        try:
            dist.init_process_group(backend='nccl', init_method='env://', timeout=10)
            print("NCCL 分布式初始化: 成功")
            
            # 获取分布式信息
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(f"Rank: {rank}, World Size: {world_size}")
            
            dist.destroy_process_group()
            print("进程组清理: 成功")
            
        except Exception as e:
            print(f"NCCL 分布式初始化失败: {e}")
            
            # 尝试使用 Gloo 后端
            try:
                dist.init_process_group(backend='gloo', init_method='env://', timeout=5)
                print("Gloo 分布式初始化: 成功")
                dist.destroy_process_group()
            except Exception as e2:
                print(f"Gloo 分布式初始化也失败: {e2}")
                
    except ImportError:
        print("PyTorch 未安装或导入失败")

def check_accelerate_config():
    """检查 Accelerate 配置"""
    print("\n=== Accelerate 配置检查 ===")
    config_files = [
        'accelerate_config.yaml',
        'accelerate_config_k8s.yaml'
    ]
    
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"找到配置文件: {config_file}")
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    print(f"  分布式类型: {config.get('distributed_type')}")
                    print(f"  计算环境: {config.get('compute_environment')}")
                    print(f"  节点数量: {config.get('num_machines')}")
                    print(f"  进程数量: {config.get('num_processes')}")
            except Exception as e:
                print(f"  配置文件解析失败: {e}")
        else:
            print(f"配置文件不存在: {config_file}")

def main():
    """主函数"""
    print("RayJob 网络连接诊断工具")
    print("=" * 50)
    
    # 记录时间
    print(f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 执行各项检查
    check_network_interfaces()
    check_hostname_resolution()
    check_port_connectivity()
    check_environment_variables()
    check_accelerate_config()
    test_pytorch_distributed()
    
    print("\n" + "=" * 50)
    print("诊断完成")
    
    # 生成诊断报告
    report = {
        'timestamp': time.time(),
        'hostname': socket.gethostname(),
        'network_interfaces': subprocess.run(['hostname', '-I'], capture_output=True, text=True).stdout.strip() if subprocess.run(['which', 'hostname'], capture_output=True).returncode == 0 else 'N/A'
    }
    
    with open('network_diagnostic_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("诊断报告已保存到: network_diagnostic_report.json")

if __name__ == "__main__":
    main()