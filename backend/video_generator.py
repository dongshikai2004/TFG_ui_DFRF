import os
import subprocess
import requests
from .utils import upload_file_and_get_url,check_and_download_video

def generate_video(data):
    """
    模拟视频生成逻辑：接收来自前端的参数，并返回一个视频路径。
    """

    print("[backend.video_generator] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")
    video_file_name = os.path.splitext(os.path.basename(data['ref_video']))[0]
    audio_file = os.path.basename(data['ref_audio'])
    if data['model_name'] == "VideoRetalk":
        # 上传音频
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise Exception("请设置DASHSCOPE_API_KEY环境变量")
        model_name="videoretalk"
        try:
            video_url = upload_file_and_get_url(api_key, model_name, data['ref_video'])
            audio_url = upload_file_and_get_url(api_key, model_name, data['ref_audio'])
        except Exception as e:
            print(f"Error: {str(e)}")
            return
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis/"
        headers = {
            'X-DashScope-Async': 'enable',
            'X-DashScope-OssResourceResolve': 'enable',
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": "videoretalk",
            "input": {
                "video_url": video_url,
                "audio_url": audio_url,
                "ref_image_url": ""
            },
            "parameters": {
                "video_extension": False
            }
        }
        response = requests.post(url, headers=headers, json=data)
        task_id = response.json()["output"]["task_id"]
        print(task_id)
        # 开始生成
        video_path = os.path.join("static", "videos", "out.mp4")
        success = check_and_download_video(task_id,api_key)
        if success:
            print(f"[backend.video_generator] 视频生成完成，路径：{video_path}")
        else :
            print(f"[backend.video_generator] 视频生成失败")
        return video_path

    video_path = os.path.join("static", "videos", "out.mp4")
    print(f"[backend.video_generator] 视频生成完成，路径：{video_path}")
    return video_path
