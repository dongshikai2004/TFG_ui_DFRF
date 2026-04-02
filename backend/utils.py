from pathlib import Path
import requests
import time
from github import Github
from github.GithubException import GithubException
import os
import base64

def get_upload_policy(api_key, model_name):
    """获取文件上传凭证"""
    url = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    params = {
        "action": "getPolicy",
        "model": model_name
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to get upload policy: {response.text}")

    return response.json()['data']

def upload_file_to_oss(policy_data, file_path):
    """将文件上传到临时存储OSS"""
    file_name = Path(file_path).name
    key = f"{policy_data['upload_dir']}/{file_name}"

    with open(file_path, 'rb') as file:
        files = {
            'OSSAccessKeyId': (None, policy_data['oss_access_key_id']),
            'Signature': (None, policy_data['signature']),
            'policy': (None, policy_data['policy']),
            'x-oss-object-acl': (None, policy_data['x_oss_object_acl']),
            'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
            'key': (None, key),
            'success_action_status': (None, '200'),
            'file': (file_name, file)
        }

        response = requests.post(policy_data['upload_host'], files=files)
        if response.status_code != 200:
            raise Exception(f"Failed to upload file: {response.text}")

    return f"oss://{key}"

def upload_file_and_get_url(api_key, model_name, file_path):
    """上传文件并获取URL"""
    # 1. 获取上传凭证，上传凭证接口有限流，超出限流将导致请求失败
    policy_data = get_upload_policy(api_key, model_name)
    # 2. 上传文件到OSS
    oss_url = upload_file_to_oss(policy_data, file_path)
    print(oss_url)
    print("===============================")
    return oss_url

def get_task_status(task_id, api_key):
    """
    查询DashScope任务状态
    """
    url = f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}'
    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None

def download_video(video_url, local_filename):
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"视频已成功下载到: {local_filename}")
        return True
    except Exception as e:
        print(f"下载视频时出错: {e}")
        return False

def check_and_download_video(task_id, api_key, poll_interval=10, timeout=600):
    """
    轮询检查任务状态并在成功后下载视频
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        # 查询任务状态
        result = get_task_status(task_id, api_key)
        if not result:
            print("获取任务状态失败，稍后重试...")
            time.sleep(poll_interval)
            continue
        # 提取任务信息
        task_status = result['output']['task_status']
        print(f"任务状态: {task_status}")
        if task_status == 'SUCCEEDED':
            # 任务成功，下载视频
            video_url = result['output']['video_url']
            print(f"任务成功，开始下载视频...")
            # 生成本地文件名
            local_filename = f"static/videos/out.mp4"
            if download_video(video_url, local_filename):
                print("视频处理完成！")
                return True
            else:
                print("视频下载失败")
                return False

        elif task_status in ['FAILED', 'CANCELLED']:
            print(f"任务失败或取消，状态: {task_status}")
            print(result)
            return False
        elif task_status in ['PENDING', 'RUNNING']:
            print("任务仍在处理中，稍后重试...")
            time.sleep(poll_interval)
        else:
            print(f"未知任务状态: {task_status}")
            time.sleep(poll_interval)

    print("任务检查超时")
    return False

def upload_mp3_to_github(mp3_file_path, repo_name='dongshikai2004/tfg-dataset', file_path_in_repo='tmp/audio.wav', access_token=os.getenv("GITHUB_TOKEN"), branch='main'):
    # 读取 MP3 文件内容（二进制模式）
    try:
        with open(mp3_file_path, 'rb') as mp3_file:
            mp3_content = mp3_file.read()
    except FileNotFoundError:
        print(f"错误：找不到文件 {mp3_file_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

    # 连接到 GitHub
    g = Github(access_token,verify=False)

    try:
        repo = g.get_repo(repo_name)
    except GithubException as e:
        print(f"获取仓库失败: {e}")
        return None

    # 上传文件到 GitHub
    try:
        # 尝试获取文件当前内容（如果存在）
        file_contents = repo.get_contents(file_path_in_repo, ref=branch)
        # 文件存在，则更新
        update_result = repo.update_file(
            file_path_in_repo,
            f"更新音频文件 {file_path_in_repo}",
            mp3_content,
            file_contents.sha, #type:ignore
            branch=branch
        )
        print(f"✅ MP3 文件 {file_path_in_repo} 更新成功")
    except GithubException as e:
        if e.status == 404:
            # 文件不存在，则创建
            create_result = repo.create_file(
                file_path_in_repo,
                f"上传音频文件 {file_path_in_repo}",
                mp3_content,
                branch=branch
            )
            print(f"✅ MP3 文件 {file_path_in_repo} 创建成功")
        else:
            print(f"❌ 操作失败: {e}")
            return None

    # 构建文件的原始直链
    raw_url = f"https://raw.githubusercontent.com/{repo_name}/{branch}/{file_path_in_repo}"
    print(f"🎵 MP3 文件直链已生成\n{raw_url}")

    return raw_url
