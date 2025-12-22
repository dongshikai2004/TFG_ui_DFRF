#!/bin/bash
set -e

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: docker run --rm -v \$(pwd):/data video-quality-assessment /data/original.mp4 /data/generated.mp4 [--max-frames 100] [--output results.json]"
    exit 1
fi
# 执行评估
python evaluate_video.py --original "$1" --generated "$2" "${@:3}"
