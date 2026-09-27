# 轻量化游戏画面超分模型研究

## 1. 项目简介

本项目面向游戏画面超分辨率任务，研究如何在较低计算开销下，将低分辨率游戏画面恢复到更高分辨率，并兼顾画面质量、推理速度和移动端部署需求。

项目首先录制原始 1920×1080 游戏视频，将其作为高清参考（HR）；随后通过统一下采样生成低分辨率图像（LR），构建配对数据集。在此基础上，以 Bicubic 插值作为基础基线，并计划进一步接入 FSR 1.0、SR-LUT 等轻量化超分方法，使用 PSNR、SSIM、处理时间等指标进行对比。

### 主要研究内容

- 构建包含不同游戏类型和场景的游戏画面超分数据集；
- 建立视频检查、抽帧、下采样、数据划分和评测流水线；
- 比较 Bicubic、FSR 1.0、SR-LUT 等方法的画质与运行效率；
- 分析 UI 文字、角色边缘、高频纹理、远处小物体及运动场景中的超分效果；
- 探索轻量化模型在移动设备上的部署与运行性能。

## 2. 仓库地址

```text
https://github.com/STRAWBERRYCAKE-GIT/game-super-resolution.git
```

## 3. 当前进度

| 模块 | 状态 |
| --- | --- |
| 1080p 游戏视频录制 | 进行中 |
| 视频信息检查 | 已完成 |
| HR 帧抽取 | 已完成 |
| Bicubic 下采样生成 LR | 已完成 |
| 按视频片段划分数据集 | 已完成 |
| Bicubic 超分基线 | 已完成 |
| PSNR、SSIM 评测 | 已完成 |
| FSR 1.0 接入 | 待完成 |
| SR-LUT 复现与训练 | 待完成 |
| 其他轻量化模型实验 | 待完成 |
| 手机端部署与测试 | 待完成 |

## 4. 项目流程

```text
原始 1080p 游戏视频
        ↓
视频信息检查与数据登记
        ↓
抽取无损 PNG 高清帧（HR）
        ↓
Bicubic 下采样生成低分辨率帧（LR）
        ↓
按完整视频片段划分训练集、验证集和测试集
        ↓
运行 Bicubic / FSR 1.0 / SR-LUT / ShiftLUT 等方法
        ↓
计算 PSNR、SSIM、处理时间等指标
        ↓
生成定量结果与画面对比图
```

当前默认进行 ×2 超分实验：

```text
HR：1920×1080
       ↓ Bicubic 下采样
LR：960×540
       ↓ 超分方法
SR：1920×1080
```

## 5. 目录结构

```text
game-super-resolution/
├── README.md                     # 项目说明文档
├── requirements.txt              # Python 依赖
├── .gitignore                    # Git 忽略规则
│
├── configs/                      # 实验配置（待完善）
│   └── experiment_x2.yaml
│
├── scripts/                      # 数据处理与评测脚本
│   ├── common.py                 # 公共工具函数
│   ├── 01_check_videos.py        # 检查视频信息
│   ├── 02_extract_frames.py      # 从视频抽取 HR 帧
│   ├── 03_generate_lr.py         # 生成 Bicubic LR 图像
│   ├── 04_split_dataset.py       # 按视频片段划分数据集
│   ├── 05_run_bicubic.py         # 运行 Bicubic 基线
│   ├── 06_evaluate.py            # 计算 PSNR 和 SSIM
│   └── 07_make_comparison.py     # 生成可视化对比图
│
├── methods/                      # 超分方法封装（待完善）
│   ├── bicubic/
│   ├── fsr1/
│   └── sr_lut/
│
├── third_party/                  # 第三方方法源码（按其许可证使用）
├── metadata/                     # 视频信息与数据划分清单
│   ├── videos.csv
│   └── splits.csv
│
├── results/                      # 实验结果
│   ├── metrics/                  # 指标和耗时 CSV
│   ├── images/                   # 超分结果图像
│   ├── comparisons/              # 可视化对比图
│   └── logs/                     # 训练或测试日志
│
└── data/                         # 本地数据，不上传 GitHub
    ├── raw_videos/               # 原始游戏录屏
    └── frames/
        ├── HR/                   # 高清参考帧
        └── LR_bicubic_X2/        # ×2 Bicubic 低分辨率帧
```

> `data/`、原始视频、完整图像数据集和模型权重体积较大，通过团队网盘共享，并在 `.gitignore` 中排除。

## 6. 环境要求

### 基础环境

- 操作系统：Windows 10/11 或 Linux；
- Python：3.10 及以上版本；
- FFmpeg：用于读取视频元数据；
- Git：用于代码版本管理；
- GPU：数据处理和 Bicubic 基线不作强制要求；后续训练模型时建议使用支持 CUDA 的 NVIDIA GPU。

### Python 依赖

```text
numpy
opencv-python
Pillow
scikit-image
tqdm
```

完整依赖见 `requirements.txt`。

## 7. 本地部署

### 7.1 克隆仓库

```bash
git clone <项目仓库地址>
cd <仓库目录>
```

### 7.2 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 7.3 安装依赖

```bash
pip install -r requirements.txt
```

### 7.4 安装并检查 FFmpeg

安装 FFmpeg 后，确保以下命令可以正常运行：

```bash
ffmpeg -version
ffprobe -version
```

### 7.5 准备数据

将原始游戏录屏放入：

```text
data/raw_videos/
```

推荐的视频文件名格式：

```text
游戏名_场景编号_分辨率.mp4
```

例如：

```text
Roco_scene01_1080p.mp4
```

## 8. 基本使用示例

以下命令均在项目根目录运行。

### 8.1 检查视频信息

```bash
python scripts/01_check_videos.py \
  --input data/raw_videos \
  --output metadata/videos.csv
```

### 8.2 抽取 HR 帧

```bash
python scripts/02_extract_frames.py \
  --input data/raw_videos \
  --output data/frames/HR \
  --sample-fps 2
```

`--sample-fps 2` 表示每秒抽取约 2 帧，并保存为无损 PNG 图像。

### 8.3 生成 ×2 LR 图像

```bash
python scripts/03_generate_lr.py \
  --hr-dir data/frames/HR \
  --lr-dir data/frames/LR_bicubic_X2 \
  --scale 2
```

该步骤将 1920×1080 的 HR 图像下采样为 960×540 的 LR 图像。

### 8.4 划分数据集

```bash
python scripts/04_split_dataset.py \
  --hr-dir data/frames/HR \
  --lr-dir data/frames/LR_bicubic_X2 \
  --output metadata/splits.csv \
  --seed 42
```

数据按照完整视频片段划分，避免相邻帧同时进入训练集和测试集造成数据泄漏。建议准备至少 6 个不同视频片段。

### 8.5 运行 Bicubic 基线

```bash
python scripts/05_run_bicubic.py \
  --splits metadata/splits.csv \
  --split test \
  --output results/images/bicubic_X2 \
  --timings results/metrics/bicubic_timings.csv
```

### 8.6 计算 PSNR 和 SSIM

```bash
python scripts/06_evaluate.py \
  --splits metadata/splits.csv \
  --split test \
  --sr-dir results/images/bicubic_X2 \
  --method bicubic \
  --output results/metrics/bicubic_quality.csv
```

### 8.7 生成对比图

```bash
python scripts/07_make_comparison.py \
  --splits metadata/splits.csv \
  --split test \
  --sr-dir results/images/bicubic_X2 \
  --output results/comparisons \
  --limit 10
```

## 9. 模型训练

> 本部分将在 SR-LUT 等模型训练流程确定后补充。

### 9.1 训练数据

待补充。

### 9.2 训练命令

```bash
# TODO：补充模型训练命令
```

### 9.3 训练配置

待补充：训练轮数、批大小、学习率、损失函数、随机种子及硬件环境等。

## 10. 模型测试

> 本部分将在模型测试代码接入后补充。

### 10.1 测试命令

```bash
# TODO：补充模型测试命令
```

### 10.2 评价指标

- PSNR：衡量超分结果与高清参考图像之间的像素误差；
- SSIM：衡量图像结构相似性；
- 单帧处理时间与 FPS：衡量推理速度；
- 模型或查找表大小：衡量存储开销；
- 峰值内存占用：衡量运行资源需求；
- 主观视觉效果：观察 UI 文字、细线、高频纹理及运动场景中的伪影。

## 11. 移动端部署

> 当前尚未完成，后续计划在移动设备上验证轻量化超分方法。

待补充内容：

- 模型转换或查找表导出方式；
- Android/HarmonyOS 工程配置；
- 推理接口和图像输入输出；
- 目标手机型号与软硬件环境；
- 移动端耗时、FPS、内存和功耗测试结果。

## 12. 实验规范

- 原始 1080p 视频作为高清参考，不进行反复转码；
- HR 与 LR 图像必须保持严格一一对应；
- 训练集、验证集和测试集按照完整视频片段划分；
- 不同方法使用相同测试集、缩放倍率和输出尺寸；
- 运行效率必须在相同设备和运行环境下比较；
- 保存实验参数、随机种子、运行日志和逐图指标；
- 论文复现时应遵循原论文的颜色空间、边界裁剪及评价设置。

## 13. 团队协作规范

- `main` 分支保存可运行的稳定版本；
- 每项任务建立独立分支，例如 `feature/dataset`、`feature/sr-lut`；
- 提交信息应清楚说明修改内容；
- 提交代码前先执行 `git pull`，减少合并冲突；
- 不向 GitHub 提交原始视频、完整数据集、大型模型权重和虚拟环境；
- 数据文件通过团队网盘共享，代码、配置和实验记录通过 GitHub 管理。

## 14. 参考方法

- Bicubic Interpolation；
- AMD FidelityFX Super Resolution 1.0；
- SR-LUT: *Practical Single-Image Super-Resolution Using Look-Up Table*, CVPR 2021；
- ShiftLUT: *Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration*, CVPR 2026。
