#!/usr/bin/env python3
import subprocess
import sys
import os

def run_step(step, video_id, background=False):
    """运行单个步骤"""
    cmd = ["python", "data_util/process_data.py", f"--id={video_id}", f"--step={step}"]
    print(f"执行步骤 {step}...")
    
    if background:
        # 后台执行
        process = subprocess.Popen(cmd)
        return process
    else:
        # 前台执行，等待完成
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"步骤 {step} 执行失败!")
            sys.exit(1)
        return None

def main():
    if len(sys.argv) < 2:
        print("用法: python script.py <video_id>")
        sys.exit(1)
    
    video_id = sys.argv[1]
    print(f"开始处理视频ID: {video_id}")
    
    # 步骤0 - 后台执行
    process0 = run_step(0, video_id, background=True)
    
    # 步骤1-2 - 顺序执行
    run_step(1, video_id)
    run_step(2, video_id)
    
    # 步骤6 - 后台执行
    process6 = run_step(6, video_id, background=True)
    
    # 步骤3-5 - 顺序执行
    run_step(3, video_id)
    run_step(4, video_id)
    run_step(5, video_id)
    
    # 等待所有后台进程完成
    print("等待后台任务完成...")
    if process0:
        process0.wait()
        print("步骤0 完成")
    if process6:
        process6.wait()
        print("步骤6 完成")
    
    # 最后执行步骤7
    run_step(7, video_id)
    
    print("所有步骤完成!")

if __name__ == "__main__":
    main()