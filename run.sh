docker run -d --gpus all --name wan22ti2v \
     -v /data/Wan-AI/Wan2.2-TI2V-5B:/Wan2.2-TI2V-5B \
     -v /data/Wan-AI/output:/workspace/output \
     172.31.0.182/system_containers/wan2-2:1014 \
     tail -f /dev/null