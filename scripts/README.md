# Game Super-Resolution Data Pipeline

第一阶段脚本：检查1080p视频、抽取HR帧、生成LR、按完整片段划分数据、运行Bicubic基线并计算PSNR/SSIM。

## 环境

- Python 3.10+；
- 安装 FFmpeg，并确保 `ffprobe` 已加入 PATH；

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 数据目录

```text
data/
├── raw_videos/
└── frames/
    ├── HR/
    └── LR_bicubic_X2/
```

视频文件名应包含游戏与片段信息，例如：`genshin_city_001.mp4`。不要让两个视频使用相同文件名。

## 依次运行

所有命令都在仓库根目录执行：

```powershell
python scripts/01_check_videos.py --input data/raw_videos --output metadata/videos.csv

python scripts/02_extract_frames.py --input data/raw_videos --output data/frames/HR --sample-fps 0.5

python scripts/03_generate_lr.py --hr-dir data/frames/HR --lr-dir data/frames/LR_bicubic_X2 --scale 2

python scripts/04_split_dataset.py --hr-dir data/frames/HR --lr-dir data/frames/LR_bicubic_X2 --output metadata/splits.csv --seed 42

python scripts/05_run_bicubic.py --splits metadata/splits.csv --split test --output results/images/bicubic_X2 --timings results/metrics/bicubic_timings.csv

python scripts/06_evaluate.py --splits metadata/splits.csv --split test --sr-dir results/images/bicubic_X2 --method bicubic --output results/metrics/bicubic_quality.csv

python scripts/07_make_comparison.py --splits metadata/splits.csv --split test --sr-dir results/images/bicubic_X2 --output results/comparisons --limit 10
```

## 说明

- `02_extract_frames.py` 默认每2秒抽1帧，并保存无损PNG。
- `03_generate_lr.py` 用Bicubic从HR生成LR；×2时，1920×1080会变成960×540。
- `04_split_dataset.py` 按完整视频片段划分，避免相邻帧同时进入训练集和测试集。
- 只有一个或两个视频片段时无法形成完整的训练/验证/测试划分，建议至少准备6个不同片段。
- 运行时间只应在相同设备和相同输入尺寸下比较。
- 当前PSNR和SSIM在RGB三个通道上计算。论文复现时需要按照对应论文设置调整颜色空间和边界裁剪。

