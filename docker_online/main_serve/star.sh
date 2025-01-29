#!/bin/bash
# 加载 conda 初始化脚本
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_remote

# 导航到项目目录
cd /root/myProject/learning/06-项目/01-AI医生/docker_online/main_serve || exit

# 在后台启动Gunicorn服务器并记录输出
gunicorn -w 1 -b 0.0.0.0:5000 app:app > logs/gunicorn.log 2>&1 &
echo "Gunicorn server started on port 5000"
