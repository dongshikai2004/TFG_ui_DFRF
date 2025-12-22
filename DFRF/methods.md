# 下载数据

| name                          | url                                                          | location                        |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------- |
| 01_MorphableModel.mat         | https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads | data_util/face_tracking/3DMM/   |
| 79999_iter.pth                | https://github.com/sstzal/DFRF/releases/download/file/79999_iter.pth | data_util/face_parsing/         |
| exp_info.npy                  | https://github.com/sstzal/DFRF/releases/download/file/exp_info.npy | data_util/face_tracking/3DMM/   |
| 0.mp4                         | https://github.com/sstzal/DFRF/releases/download/Base_Videos/0.mp4 | dataset/vids/                   |
| 1.mp4                         | https://github.com/sstzal/DFRF/releases/download/Base_Videos/1.mp4 | dataset/vids/                   |
| 2.mp4                         | https://github.com/sstzal/DFRF/releases/download/Base_Videos/2.mp4 | dataset/vids/                   |
| 500000_head.tar<br />(rename) | https://github.com/sstzal/DFRF/releases/download/Pretrained_Models/french_500000_head.tar | dataset/finetune_models/french/ |

# 预处理模型

```bash
python data_util/face_tracking/convert_BFM.py
```

# 处理数据集

```bash
python process_data.py 0
```

```bash
python process_data.py french
```

# 训练

```bash
sh run.sh french
```

可跳过,已下载模型可跳过

# 推理

```bash
python rendering.py
```
