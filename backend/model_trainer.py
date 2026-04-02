import subprocess
import os
import time
import shutil
import re
from .utils import upload_file_and_get_url
from datetime import datetime, timedelta
def train_model(data):
    """
    模拟模型训练逻辑。
    """
    print("[backend.model_trainer] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    video_path = data['ref_video']
    epoch = data['epoch']
    print(f"输入视频：{video_path}")

    if data['model_choice'] == "VideoRetalk":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise Exception("请设置DASHSCOPE_API_KEY环境变量")
        model_name="videoretalk"
        file_path = data['ref_video']
        try:
            public_url = upload_file_and_get_url(api_key, model_name, file_path)
            # with open("tmp/video_url.txt",'w') as f:
            #     f.write(public_url)
            expire_time = datetime.now() + timedelta(hours=48)
            print(f"文件上传成功，有效期为48小时，过期时间: {expire_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"临时URL: {public_url}")
        except Exception as e:
            print(f"Error: {str(e)}")
    return video_path
