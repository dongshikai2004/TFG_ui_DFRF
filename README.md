# TFG_ui 项目使用指南

本项目包含核心功能后端（DFRF）与视频评测工具（Assess）两部分。

## 1. 环境准备

### 1.1 Python 环境

建议使用 Conda 创建独立的 Python 环境：
```bash
conda create -n tfg python=3.8.0
conda activate tfg
```

### 1.2 依赖安装

在项目根目录下安装必要的依赖：
```bash
pip install -r requirements.txt
```

---

按照`DFRF\methods.md`准备好相应文件，完成后运行

```bash
python data_util/face_tracking/convert_BFM.py
```

完成依赖安装

### 1.3 API配置

在 `.env`中配置

```bash
DASHSCOPE_API_KEY =
GITHUB_TOKEN =
```

用于api服务

在`backend/config.py`中配置

```py
CONTAINER_NAME = ""
repo_name = ""
```

## 2. 核心项目运行 (DFRF)

### 2.1 构建镜像

进入 `DFRF` 目录，构建 Docker 镜像：
```bash
cd DFRF
docker build -t tfgui:latest .
```

### 2.2 启动后台容器
运行以下命令启动容器。
> **注意**：请将 `D:\GithubDoc\TFG_ui\DFRF` 替换为你本地项目的**绝对路径**。

```bash
docker run -d --name dfrf-container -v D:\GithubDoc\TFG_ui\DFRF:/DFRF tfgui:latest sleep infinity
```

### 2.3 启动后端服务
返回项目根目录，运行后端程序：
```bash
cd ..
python app.py
```

---

## 3. 视频评测使用 (Assess)

评测工具用于对比原始视频与生成视频的质量。

### 3.1 构建评测镜像
进入 `assess` 目录，构建评测镜像：
```bash
cd assess
docker build -t assess:latest .
```

### 3.2 启动评测容器
运行以下命令启动后台评估容器。
> **注意**：请确保将挂载路径中的 `D:/GithubDoc/TFG_ui/assess/...` 替换为你本地的**绝对路径**。

```bash
docker run -d --name video-assess \
  -v "D:/GithubDoc/TFG_ui/assess/videos:/assess/videos" \
  -v "D:/GithubDoc/TFG_ui/assess/results:/assess/results" \
  -v "D:/GithubDoc/TFG_ui/assess/evaluate_video.py:/assess/evaluate_video.py" \
  assess:latest tail -f /dev/null
```

### 3.3 执行评估

#### 方法 A：自动评估（推荐）
在 `assess` 目录下直接运行 Python 脚本。该脚本会自动检测容器状态并输出结果到 `results/` 目录：
```bash
python run.py
```

#### 方法 B：手动单次评估
如果你需要手动指定文件进行评估，可运行：
```bash
docker exec -it video-assess python /assess/evaluate_video.py \
  --original /assess/videos/original/french640.mp4 \
  --generated /assess/videos/generated/out.mp4 \
  --output /assess/results/french640_results.json \
  --max-frames 30
```

---

## 4. 清理与维护

评估完成后，如需停止并删除评测容器，请执行：
```bash
docker stop video-assess && docker rm video-assess
```
如需停止核心项目容器：
```bash
docker stop dfrf-container && docker rm dfrf-container
```

---

## 目录结构参考

- `DFRF/`: 核心模型与镜像构建目录
- `assess/`: 评测脚本与视频存放目录
  - `videos/`: 存放待评估视频（original/generated）
  - `results/`: 评估结果输出目录
  - `run.py`: 评测自动化触发脚本
- `app.py`: 项目后端入口