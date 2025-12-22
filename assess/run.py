#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse

def run_docker_assessment(
    container_name="video-assess",
    image_name="assess:latest",
    original_video="french640.mp4",
    generated_video="out.mp4",
    output_json="french640_results.json"
):
    # 宿主机当前目录（assess/）
    current_dir = os.path.abspath(os.getcwd())
    videos_host = os.path.join(current_dir, "videos")
    results_host = os.path.join(current_dir, "results")

    # 确保 results 目录存在
    os.makedirs(results_host, exist_ok=True)

    # Step 1: 启动后台容器（如未运行）
    try:
        # 检查容器是否已在运行
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            capture_output=True, text=True, check=False
        )
        if "true" not in result.stdout:
            print(f"[+] 启动容器 {container_name}...")
            cmd_start = [
                "docker", "run", "-d",
                "--name", container_name,
                "-v", f"{videos_host}:/data/videos",
                "-v", f"{results_host}:/data/results",
                image_name,
                "tail", "-f", "/dev/null"
            ]
            subprocess.run(cmd_start, check=True)
        else:
            print(f"[✓] 容器 {container_name} 已在运行")
    except subprocess.CalledProcessError:
        print(f"[+] 启动新容器 {container_name}...")
        cmd_start = [
            "docker", "run", "-d",
            "--name", container_name,
            "-v", f"{videos_host}:/data/videos",
            "-v", f"{results_host}:/data/results",
            image_name,
            "tail", "-f", "/dev/null"
        ]
        subprocess.run(cmd_start, check=True)

    # Step 2: 执行评估
    print("[+] 容器内视频文件列表：")
    try:
        print("\n📁 /assess/videos/original/")
        subprocess.run(["docker", "exec", container_name, "ls", "-1", "/assess/videos/original/"])
        
        print("\n📁 /assess/videos/generated/")
        subprocess.run(["docker", "exec", container_name, "ls", "-1", "/assess/videos/generated/"])
    except subprocess.CalledProcessError as e:
        print(f"[!] 无法列出文件（可能路径错误）: {e}")
    print("[+] 开始评估...")
    cmd_exec = [
        "docker", "exec", "-i", container_name,
        "python", "/assess/evaluate_video.py",
        "--original", f"/assess/videos/original/{original_video}",
        "--generated", f"/assess/videos/generated/{generated_video}",
        "--max-frames","300",
        "--output", f"/assess/results/{output_json}"
    ]
    
    # 使用 subprocess.run 并实时输出（避免卡住）
    subprocess.run(cmd_exec)

    # Step 3: （可选）自动清理容器
    print("[✓] 评估完成！结果已保存。")
    cleanup = input("是否停止并删除容器？(y/N): ").strip().lower()
    if cleanup == 'y':
        subprocess.run(["docker", "stop", container_name], stdout=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", container_name], stdout=subprocess.DEVNULL)
        print(f"[✓] 容器 {container_name} 已清理")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通过 Docker 容器评估视频质量")
    parser.add_argument("--container", default="video-assess", help="容器名")
    parser.add_argument("--image", default="assess:latest", help="镜像名")
    parser.add_argument("--original", default="french640.mp4", help="原始视频文件名")
    parser.add_argument("--generated", default="out.mp4", help="生成视频文件名")
    parser.add_argument("--output", default="french640_results.json", help="输出 JSON 文件名")

    args = parser.parse_args()

    run_docker_assessment(
        container_name=args.container,
        image_name=args.image,
        original_video=args.original,
        generated_video=args.generated,
        output_json=args.output
    )