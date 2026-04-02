import os
import dashscope
import time
import requests
import wave
from dashscope.audio.tts_v2 import VoiceEnrollmentService, SpeechSynthesizer

from .utils import upload_mp3_to_github
from .video_generator import generate_video
from .config import repo_name
def chat_response(data):
    """
    模拟实时对话系统视频生成逻辑。
    """
    print("[backend.chat_engine] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    # 语音转文字
    input_audio = "./static/audios/input.wav"
    input_text = "./static/text/input.txt"
    audio_to_text(input_audio, input_text, api_key=os.getenv("DASHSCOPE_API_KEY"))
    # 大模型回答
    output_text = "./static/text/output.txt"
    get_ai_response(input_text, output_text, model=data["api_choice"], api_key=os.getenv("DASHSCOPE_API_KEY"))

    # 语音克隆
    voice_clone_mode = data.get("voice_clone", "CosyVoice API")
    output_audio = './static/audios/output.wav'

    text_to_audio(output_text, output_audio, data["ref_audio"], api_key=os.getenv("DASHSCOPE_API_KEY"), mode=voice_clone_mode)

    # 生成视频
    video_path = os.path.join("static", "videos", "out.mp4")
    d={
        'model_name':data["model_name"],
        'iter':data["iter"],
        'ref_video':data["ref_video"],
        'ref_audio':output_audio,
    }
    generate_video(d)
    print(f"[backend.chat_engine] 生成视频路径：{video_path}")
    return video_path

def audio_to_text(input_audio, input_text, api_key):
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
    audio_file_path = f"file://{input_audio}"
    messages = [
        {"role": "system", "content": [{"text": ""}]},
        {"role": "user", "content": [{"audio": audio_file_path}]}
    ]
    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model="qwen3-asr-flash",
        messages=messages,
        result_format="message",
        asr_options={
            "language": "en",
            "enable_itn":True
        }
    )
    # print(response)
    text=response.output.choices[0].message.content[0]["text"] #type:ignore
    print(text)
    with open(input_text,'w', encoding='utf-8') as f:
        f.write(text)
    return text

def get_ai_response(input_text, output_text, model, api_key):
    with open(input_text, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
    messages = [
        {"role": "system", "content": "You are a helpful assistant.It's a try in talking face generate.Response in few words."},
        {"role": "user", "content": content},
    ]
    response = dashscope.Generation.call(
        api_key=api_key,
        model=model,
        messages=messages,#type:ignore
        result_format="message",
        enable_thinking=False,
    )
    output = response.output.choices[0].message.content#type:ignore
    print(output)
    with open(output_text, 'w', encoding='utf-8') as file:
        file.write(output)#type:ignore

    print(f"答复已保存到: {output_text}")
    return output

def text_to_audio(output_text, output_audio, ref_audio, api_key, mode="CosyVoice API"):
    with open(output_text,'r', encoding='utf-8') as f:
        content = f.read()

    dashscope.api_key = api_key
    if not dashscope.api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable not set.")

    # 2. 定义复刻参数
    TARGET_MODEL = "cosyvoice-v3-flash"
    VOICE_PREFIX = "myvoice"

    # 3. 创建音色 (异步任务)
    print("--- Step 0: Uploading voice ---")
    ref_audio_url = upload_mp3_to_github(ref_audio,repo_name=repo_name)
    print("--- Step 1: Creating voice enrollment ---")
    service = VoiceEnrollmentService()
    try:
        voice_id = service.create_voice(
            target_model=TARGET_MODEL,
            prefix=VOICE_PREFIX,
            url=ref_audio_url#type:ignore
        )
        print(f"Voice enrollment submitted successfully. Request ID: {service.get_last_request_id()}")
        print(f"Generated Voice ID: {voice_id}")
    except Exception as e:
        print(f"Error during voice creation: {e}")
        raise e
    # 4. 轮询查询音色状态
    print("\n--- Step 2: Polling for voice status ---")
    max_attempts = 30
    poll_interval = 10 # 秒
    for attempt in range(max_attempts):
        try:
            voice_info = service.query_voice(voice_id=voice_id)
            status = voice_info.get("status")#type:ignore
            print(f"Attempt {attempt + 1}/{max_attempts}: Voice status is '{status}'")
            if status == "OK":
                print("Voice is ready for synthesis.")
                break
            elif status == "UNDEPLOYED":
                print(f"Voice processing failed with status: {status}. Please check audio quality or contact support.")
                raise RuntimeError(f"Voice processing failed with status: {status}")
            # 对于 "DEPLOYING" 等中间状态，继续等待
            time.sleep(poll_interval)
        except Exception as e:
            print(f"Error during status polling: {e}")
            time.sleep(poll_interval)
    else:
        print("Polling timed out. The voice is not ready after several attempts.")
        raise RuntimeError("Polling timed out. The voice is not ready after several attempts.")

    # 5. 使用复刻音色进行语音合成
    print("\n--- Step 3: Synthesizing speech with the new voice ---")
    try:
        synthesizer = SpeechSynthesizer(model=TARGET_MODEL, voice=voice_id)
        # call()方法返回二进制音频数据
        audio_data = synthesizer.call(content)
        print(f"Speech synthesis successful. Request ID: {synthesizer.get_last_request_id()}")

        # 6. 保存音频文件
        with open(output_audio, "wb") as f:
            f.write(audio_data)#type:ignore
        print(f"Audio saved to {output_audio}")

    except Exception as e:
        print(f"Error during speech synthesis: {e}")
