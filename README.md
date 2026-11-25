# 轻量化游戏画面超分模型研究

本项目面向游戏画面超分辨率重建任务，建立了涵盖游戏录屏、帧清洗、低分辨率（LR）与高分辨率（HR）图像配对、模型微调及统一评测的实验流程，分析重建质量、查表推理时间与存储开销。

正式实验采用 ×4 放大倍率，设置 Bicubic、官方预训练 SR-LUT 和游戏数据微调 SR-LUT 三组对照。三组使用相同的 LR 输入、HR 参考和测试清单，分别统计实验结果。

仓库地址：[STRAWBERRYCAKE-GIT/game-super-resolution](https://github.com/STRAWBERRYCAKE-GIT/game-super-resolution)

## 1. 项目概况与实验流程

| 模块 | 实现内容 |
| --- | --- |
| 视频检查、抽帧与 Bicubic 下采样 | 完成视频元数据检查、HR 帧提取和 ×4 LR 构建 |
| 数据集划分 | 在每款游戏内部按完整原始视频划分训练集、验证集和测试集 |
| 纯色帧清洗及关联 CSV 清理 | 同步清理配对图像及关联记录，清洗报告与备份保存在 `results/cleanup_runs/` |
| ×4 LR/HR 尺寸对齐 | 保存对齐参考图，正式实验使用 `metadata/splits_X4_aligned.csv` |
| Bicubic ×4 基线 | 完成对齐测试集上的重建、逐图质量评价和耗时统计 |
| 官方预训练 SR-LUT ×4 | 完成统一 LUT 导出、查表推理、质量评价和耗时统计 |
| 游戏数据微调与 LUT 导出 | 完成 ×4 网络微调、LUT 导出及测试，支持检查点保存与断点恢复 |

实验依次完成数据准备与尺寸对齐、预训练参数和微调参数的 LUT 导出、三组图像重建及统一评价。两组 SR-LUT 共用同一查表程序，三组方法共用同一质量评价程序。

## 2. 项目结构

```text
game-super-resolution/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_check_videos.py          # 视频元数据检查
│   ├── 02_extract_frames.py        # 抽取 HR 帧
│   ├── 03_generate_lr.py           # Pillow Bicubic 下采样
│   ├── 04_split_dataset.py         # 每游戏按原始视频划分数据
│   ├── 05_run_bicubic.py           # Bicubic 基线
│   ├── 06_evaluate.py              # 逐图 PSNR / SSIM
│   ├── 07_make_comparison.py       # LR / Bicubic / HR 对比面板
│   ├── 09_run_srlut.py             # 预训练或微调 LUT 的统一推理入口
│   ├── 10_finetune_srlut.py        # ×4 网络微调与恢复训练
│   ├── 11_export_srlut.py          # 网络参数导出 int8 LUT
│   ├── 12_align_x4_pairs.py        # 保存对齐 HR，修复 ×4 配对
│   ├── clean_solid_frames.py       # 纯色帧与关联数据清洗
│   ├── common.py                  # 公共读写与命名工具
│   ├── srlut_numpy.py             # NumPy 四维插值与旋转融合
│   └── srlut_training.py          # SRNet、配对读取与分块验证
├── results/
│   ├── metrics/
│   │   ├── bicubic_X4_aligned_quality.csv
│   │   ├── bicubic_X4_aligned_timings.csv
│   │   ├── srlut_pretrained_exported_X4_quality.csv
│   │   ├── srlut_pretrained_exported_X4_timings.csv
│   │   ├── srlut_pretrained_exported_X4_timings.json
│   │   ├── srlut_finetuned_X4_quality.csv
│   │   ├── srlut_finetuned_X4_timings.csv
│   │   └── srlut_finetuned_X4_timings.json
│   └── cleanup_runs/              # 清洗报告、备份图像与 CSV
└── tests/
    ├── test_align_x4.py
    ├── test_srlut.py
    ├── test_srlut_training.py
    └── reference/                 # 作者示例与插值参考代码
```

数据、权重及生成文件的目录约定如下。

| 路径 | 内容 |
| --- | --- |
| `data/raw_videos/` | 原始游戏录屏 |
| `data/frames/HR/` | 原始 HR 帧，保留实际分辨率 |
| `data/frames/LR_bicubic_X4/` | 最初生成的 ×4 LR 帧 |
| `data/frames/aligned_X4/HR/` | 需要裁齐的 HR 新副本 |
| `data/frames/aligned_X4/LR_bicubic_X4/` | 尺寸修复时需要重新生成的 LR；正确 LR 会直接复用 |
| `metadata/videos.csv` | 视频信息 |
| `metadata/splits_X4.csv` | 原始划分清单 |
| `metadata/splits_X4_aligned.csv` | 正式 ×4 实验统一使用的清单 |
| `weights/srlut_official/` | 官方 LUT、原始网络参数及来源记录 |
| `weights/srlut_game_X4/` | 正式微调检查点、配置和训练日志 |
| `results/images/` | 三组方法分别保存的 SR 图片 |
| `results/comparisons/` | 可视化对比图 |

`aligned_X4/` 仅保存需要修复的图像，对齐清单同时引用尺寸已正确的原始路径。数据配对与样本统计均以 CSV 清单为依据。

## 3. 环境配置

### 3.1 基础要求

- 操作系统：Windows 10/11 或 Linux；Python 环境：3.10 或 3.11。
- FFmpeg / ffprobe：用于视频元数据读取。
- 基础依赖：NumPy、OpenCV、Pillow、scikit-image、tqdm，见 `requirements.txt`。
- 网络微调与 LUT 导出依赖 PyTorch，以下配置命令采用 `2.5.1+cu121`。
- 数据处理、Bicubic 和当前 NumPy LUT 推理使用 CPU；网络微调可使用 CUDA GPU。

### 3.2 获取代码并创建虚拟环境

```bash
git clone https://github.com/STRAWBERRYCAKE-GIT/game-super-resolution.git
cd game-super-resolution
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.3 安装依赖并检查环境

```powershell
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --timeout 120
```

解释器路径、PyTorch 版本及 CUDA 可用性通过以下命令检查：

```powershell
python -c "import sys, torch; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 4. 数据命名与划分规则

原始视频命名为 `person01_clip001.mp4`，抽帧后对应：

```text
person01_clip001__f00001320.png
```

| 字段 | 含义 |
| --- | --- |
| `person01` | 游戏编号，相同编号表示同一游戏 |
| `clip001` | 该游戏中的原始录制视频编号 |
| `person01_clip001` | 跨游戏唯一的原始视频 ID，即 CSV 的 `clip_id` |
| `f00001320` | 原始视频中的帧编号 |

划分脚本在**每款游戏内部**以完整原始视频为单位随机分配 train、val、test，默认目标比例为 70% / 15% / 15%，随机种子为 42。比例针对原始视频数量；整数分配、视频时长和抽帧数量会使帧数比例不同。同一视频的所有帧只进入一个集合，避免相邻帧泄漏。

训练、验证和测试三路划分要求每款游戏至少包含 3 段独立原始视频。初始划分确定后，尺寸对齐、清洗和微调均保留原有视频归属。

CSV 主要字段：

```csv
split,game_id,clip_id,lr_path,hr_path
```

| 集合 | 用途 |
| --- | --- |
| train | 随机裁剪训练样本，更新网络参数 |
| val | 选择最佳网络检查点 |
| test | 最终三组方法比较，不参与参数更新或模型选择 |

## 5. 数据准备与 ×4 尺寸对齐

### 5.1 视频检查与抽帧

原始录屏保存在 `data/raw_videos/`。HR 参考帧由录屏提取，其分辨率按实际文件记录。

```powershell
python scripts/01_check_videos.py --input data/raw_videos --output metadata/videos.csv
python scripts/02_extract_frames.py --input data/raw_videos --output data/frames/HR --sample-fps 2
```

`--sample-fps 2` 表示每秒约抽取 2 帧。视频文件名应全局唯一，并符合前述游戏/视频命名规则。

### 5.2 生成 LR 并建立初始划分

```powershell
python scripts/03_generate_lr.py --hr-dir data/frames/HR --lr-dir data/frames/LR_bicubic_X4 --scale 4
python scripts/04_split_dataset.py --hr-dir data/frames/HR --lr-dir data/frames/LR_bicubic_X4 --output metadata/splits_X4.csv --train-ratio 0.70 --val-ratio 0.15 --seed 42
```

LR 图像由 Pillow 的 Bicubic 下采样算法生成。

### 5.3 清洗纯色帧

纯色帧清洗位于正式尺寸对齐和三组方法测试之前。`clean_solid_frames.py` 默认预览清理范围，使用 `--apply` 执行清理：

```powershell
python scripts/clean_solid_frames.py
python scripts/clean_solid_frames.py --apply
```

该脚本以全部像素颜色完全一致为判定条件识别 HR 纯色帧，同步处理默认目标目录中的配对图像、基线结果及可识别的逐帧 CSV 记录，备份与报告保存在 `results/cleanup_runs/`。近似纯色帧不在此判定范围内。清洗后的样本数量和评价指标以更新后的清单为依据。

默认清洗范围未包含全部后续生成的对齐图像和 SR-LUT 结果目录。测试集发生清洗变更时，对应重建结果及统计文件须依据更新后的清单重新生成，以保持三组样本一致。

### 5.4 保存对齐 HR，生成正式清单

`03_generate_lr.py` 在内存中裁去 HR 右侧和底部不足倍率整数倍的部分，再生成 LR；原始 HR 文件保持原尺寸。因此，当 HR 宽或高不能被 4 整除时，由尺寸对齐脚本另存裁剪后的参考图，并更新配对路径。

例如，原始 HR 为 `2880×1610` 时，对齐 HR 为 `2880×1608`，对应 LR 为 `720×402`。

```powershell
python scripts/12_align_x4_pairs.py --splits metadata/splits_X4.csv --output metadata/splits_X4_aligned.csv
```

尺寸对齐包含以下处理：

- 检查全部 train/val/test 配对，并保留原有集合和元数据；
- 将需要裁齐的 HR 新副本写入 `data/frames/aligned_X4/HR/`；
- 核对现有 LR 是否逐像素匹配裁后 HR 的 Bicubic 下采样结果，一致则复用，不一致才另存新的 LR；
- 保留原始图片和原 CSV，生成新清单及 `metadata/splits_X4_aligned.alignment.json` 记录。

为保护已有数据，对齐 CSV 或目标目录已存在时程序停止执行。生成的对齐清单可直接供后续实验使用，尺寸已正确的配对保留原路径。

所有 ×4 正式实验统一使用 `metadata/splits_X4_aligned.csv`。原 HR 尺寸对应的历史重建结果不纳入对齐实验；三组结果均由统一清单重新生成。

## 6. Bicubic 基线

### 6.1 重建与计时

```powershell
python scripts/05_run_bicubic.py --splits metadata/splits_X4_aligned.csv --split test --output results/images/bicubic_X4_aligned --timings results/metrics/bicubic_X4_aligned_timings.csv
```

该方法将 CSV 指定的 LR 放大至对应 HR 尺寸。计时范围是 Pillow 的 resize 操作，不包括图片读取和 PNG 保存。

### 6.2 质量评价

```powershell
python scripts/06_evaluate.py --splits metadata/splits_X4_aligned.csv --split test --sr-dir results/images/bicubic_X4_aligned --method bicubic --output results/metrics/bicubic_X4_aligned_quality.csv --shave 0
```

### 6.3 对比图

```powershell
python scripts/07_make_comparison.py --splits metadata/splits_X4_aligned.csv --split test --sr-dir results/images/bicubic_X4_aligned --output results/comparisons/bicubic_X4_aligned --limit 10
```

`07_make_comparison.py` 按清单顺序选取样本，生成 LR 最近邻放大预览、Bicubic 重建和 HR 参考三列图。LR 列用于展示输入像素结构。脚本中间列的默认标题为 Bicubic，SR-LUT 对比图的标题需与实际重建方法对应。

## 7. 官方预训练 SR-LUT ×4

### 7.1 准备初始权重

官方模型参数、原始查找表及来源记录保存在 `weights/srlut_official/`，主要文件如下：

| 文件 | 用途 |
| --- | --- |
| `Model_S_x4_original_weights.npz` | 从作者提供的 ×4 检查点提取的六层网络参数，作为微调初始值 |
| `Model_S_x4_4bit_int8.npy` | 作者导出目录提供的原始 ×4 LUT，可用于初步查表测试 |
| `SOURCE.json` | 文件来源、哈希与检查点/LUT 对应关系记录 |

作者的 `Model_S_x4_original.pth` 采用整对象保存方式。本项目通过 NPZ 加载提取后的网络参数，新检查点采用 state_dict 保存。作者测试目录与导出目录中存在数值不同的 ×4 LUT，实验通过 `SOURCE.json` 和文件哈希区分其来源。

### 7.2 统一导出原始组 LUT

为控制导出设备和浮点舍入差异，原始网络参数与微调参数均通过 `11_export_srlut.py` 在 CPU 上导出 LUT。网络微调与 LUT 导出为独立阶段，训练阶段可使用 GPU。

```powershell
python scripts/11_export_srlut.py --checkpoint weights/srlut_official/Model_S_x4_original_weights.npz --output weights/srlut_pretrained_X4_exported.npy --sampling-bits 4 --device cpu
```

`--sampling-bits 4` 对应采样步长 `2^4=16`，最后一个采样值为 255。输出 LUT 的形状为 `(83521, 1, 4, 4)`，数据类型为 int8，有效载荷为 1,336,336 字节，约 1.274 MiB。`.npy` 文件另包含少量格式头信息，文件大小按实际字节数记录。

导出程序采用禁止覆盖策略。同一配置的导出文件可复用，不同网络参数或导出配置采用独立文件名，并与推理记录中的 LUT 路径对应。

### 7.3 查表推理

```powershell
python scripts/09_run_srlut.py --splits metadata/splits_X4_aligned.csv --split test --scale 4 --lut weights/srlut_pretrained_X4_exported.npy --method srlut_pretrained --output results/images/srlut_pretrained_X4_aligned --timings results/metrics/srlut_pretrained_exported_X4_timings.csv
```

`--limit 1` 用于单帧流程检查，检查结果保存于独立目录。正式测试覆盖完整测试清单，不设置样本数量限制。

`09_run_srlut.py` 采用 CPU NumPy 实现，执行 RGB 四方向查表及四维单纯形插值，推理阶段不执行神经网络前向计算。计时范围排除 LUT 加载、图片解码、默认首图预热和 PNG 保存。对应的 timings JSON 记录 LUT 哈希、倍率、NumPy 版本及计时范围。

预训练组质量文件和耗时文件对应同一次 LUT 导出。LUT 来源与哈希由对应 JSON 记录；LUT 变更后，该组重建、质量和耗时文件同步更新。

### 7.4 质量评价

```powershell
python scripts/06_evaluate.py --splits metadata/splits_X4_aligned.csv --split test --sr-dir results/images/srlut_pretrained_X4_aligned --method srlut_pretrained --output results/metrics/srlut_pretrained_exported_X4_quality.csv --shave 0
```

## 8. 游戏数据微调

微调流程为：加载官方 ×4 网络参数，使用游戏 train 数据更新参数，使用 val 选择网络检查点，再导出新的 LUT 用于最终测试。原始 `.npy` 查找表不作为神经网络检查点加载。

### 8.1 两步流程检查

```powershell
python scripts/10_finetune_srlut.py --splits metadata/splits_X4_aligned.csv --steps 2 --batch-size 2 --patch-size 16 --val-every 2 --log-every 1 --val-limit 2 --output weights/srlut_game_X4_check_aligned
```

两步检查使用小批量样本和两张验证帧，验证数据读取、网络更新、验证及检查点保存流程。检查任务使用独立输出目录，其权重与指标不纳入正式实验。

### 8.2 正式微调

```powershell
python scripts/10_finetune_srlut.py --splits metadata/splits_X4_aligned.csv --init weights/srlut_official/Model_S_x4_original_weights.npz --steps 5000 --batch-size 4 --patch-size 48 --lr 1e-5 --val-every 500 --output weights/srlut_game_X4 --device cuda --shave 0
```

| 配置项 | 参数设置 |
| --- | --- |
| 倍率 | ×4 |
| 训练数据 | `splits_X4_aligned.csv` 的 train 行 |
| LR / HR 裁剪大小 | `48×48` / `192×192` |
| Batch size | 4 |
| 优化器 / 学习率 | Adam / `1e-5` |
| 损失函数 | MSE |
| 总更新数 | 5000 步 |
| 验证 | 初始第 0 步、每 500 步及结束时 |
| 随机种子 | 默认 42 |
| 数据加载进程 | 默认 `--workers 0`，兼容 Windows |

上述命令给出微调复现配置，具体运行参数由 `config.json` 保存。六层网络参数均参与更新，LR/HR 裁剪、翻转和旋转保持同步，并保留原网络的四方向融合、反射填充及归一化关系。微调效果由验证记录和最终 LUT 测试结果评价。

正式命令不使用 `--val-limit`。第 0 步的初始网络也参与最佳模型比较，`best.pt` 按平均逐帧 RGB 验证 PSNR 选择；验证采用分块预测，默认 `--val-tile 64`。测试集像素不用于训练和模型选择。

训练显存占用主要受 batch-size 影响，分块验证显存占用受 val-tile 影响。不同训练配置采用独立实验目录，以保留参数、日志与检查点的对应关系。

### 8.3 输出文件与训练恢复

| 文件 | 含义 |
| --- | --- |
| `weights/srlut_game_X4/config.json` | 参数、清单哈希、初始权重哈希及实际设备 |
| `weights/srlut_game_X4/training.csv` | 训练 MSE、验证 PSNR 与最佳步骤 |
| `weights/srlut_game_X4/best.pt` | 验证集选择的最佳检查点 |
| `weights/srlut_game_X4/last.pt` | 最近保存的训练/优化器状态 |
| `weights/srlut_game_X4/summary.json` | 已完成更新数及最佳步骤 |
| `weights/srlut_game_X4/val_*.csv` | 相应验证轮次的逐图记录 |

断点恢复命令如下：

```powershell
python scripts/10_finetune_srlut.py --splits metadata/splits_X4_aligned.csv --steps 5000 --batch-size 4 --patch-size 48 --lr 1e-5 --val-every 500 --output weights/srlut_game_X4 --resume weights/srlut_game_X4/last.pt --device cuda --shave 0
```

`--steps` 表示包含已完成步骤在内的总更新数。恢复训练沿用原目录的 `config.json`、`best.pt` 和 `last.pt`，保持数据清单、batch-size、patch-size、学习率、seed 和 shave 等关键配置一致。默认设置下，`last.pt` 在每 50 步日志检查点及验证结束后保存；恢复位置以最近一次已保存状态为准。

`summary.json` 的 `best_step` 记录验证集最优步骤。该值为 0 时，表示训练期间的验证 PSNR 未超过初始网络，`best.pt` 对应初始参数。

### 8.4 导出微调 LUT

```powershell
python scripts/11_export_srlut.py --checkpoint weights/srlut_game_X4/best.pt --output weights/srlut_game_X4/Model_S_x4_4bit_int8.npy --sampling-bits 4 --device cpu
```

导出规则与原始组相同：枚举四维采样网格，计算单方向网络输出，进行截断、乘 127、舍入和 int8 存储。LUT 旁的 JSON 保存检查点与导出文件的哈希。

验证日志中的 PSNR 对应 CNN 输出；量化和插值可能改变最终结果。因此报告中的最终质量以导出 LUT 的测试指标为准。

## 9. 微调 LUT 测试与三组汇总

### 9.1 测试与评价

```powershell
python scripts/09_run_srlut.py --splits metadata/splits_X4_aligned.csv --split test --scale 4 --lut weights/srlut_game_X4/Model_S_x4_4bit_int8.npy --method srlut_finetuned --output results/images/srlut_finetuned_X4_aligned --timings results/metrics/srlut_finetuned_X4_timings.csv
```

```powershell
python scripts/06_evaluate.py --splits metadata/splits_X4_aligned.csv --split test --sr-dir results/images/srlut_finetuned_X4_aligned --method srlut_finetuned --output results/metrics/srlut_finetuned_X4_quality.csv --shave 0
```

### 9.2 文件对应关系

三组正式实验的质量与耗时文件均位于 `results/metrics/`，文件对应关系如下。SR-LUT 耗时 CSV 另附同名 JSON，记录模型来源与运行配置。

| 实验组 | 逐图质量 CSV | 逐图耗时 CSV |
| --- | --- | --- |
| Bicubic | `bicubic_X4_aligned_quality.csv` | `bicubic_X4_aligned_timings.csv` |
| 原始 SR-LUT | `srlut_pretrained_exported_X4_quality.csv` | `srlut_pretrained_exported_X4_timings.csv` |
| 微调 SR-LUT | `srlut_finetuned_X4_quality.csv` | `srlut_finetuned_X4_timings.csv` |

### 9.3 评价口径

- **画质：** `06_evaluate.py` 使用 RGB uint8 图像、`data_range=255`，逐帧计算 PSNR / SSIM 后取算术平均。三组统一采用 `--shave 0`，评价区域为完整图像。
- **推理时间：** 对逐帧 `time_ms` 取算术平均，串行等效 FPS 定义为 `1000 / 平均 time_ms`。该值反映图像重建阶段的处理能力，不包含完整视频解码与显示开销。
- **存储：** LUT 记录实际字节数；Bicubic 的 LUT 大小为不适用。微调不改变网络结构和 LUT 形状，主要考察画质变化。
- **分组统计：** 总体指标按帧平均；分游戏指标通过 `image` / `clip_id` 与划分清单关联游戏编号后汇总。`06_evaluate.py` 直接输出逐帧结果和总体均值，分游戏汇总另行整理。
- **视觉评价：** 主要观察 UI 文字、角色轮廓、细线、纹理和小物体。当前算法独立处理各帧，单帧指标不反映连续帧闪烁等时序表现。

本项目的 SR-LUT 采用 NumPy 参考实现，查表过程包含排序、索引、加权及数组搬运；Bicubic 采用 Pillow 底层实现。因此，两者耗时反映当前软件实现下的运行表现，与论文中的手机原生实现存在差异。预训练与微调 SR-LUT 使用相同的查表脚本及计时设置。

## 10. 程序检查

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

| 测试文件 | 主要覆盖内容 |
| --- | --- |
| `test_align_x4.py` | 不整除尺寸、正确 LR 复用、错误 LR 重建、原始文件保留、失败清理和新 CSV 对接 |
| `test_srlut.py` | 24 种余数排序、边界、旋转、分块、CSV 接入和作者示例逐像素一致性 |
| `test_srlut_training.py` | 初始参数加载、独立 NumPy 前向核对、裁剪增强对齐、分块验证、训练更新、恢复和 LUT 导出 |

训练相关测试依赖 PyTorch，缺少该依赖时标记为跳过。单元测试使用合成小样本及作者参考图片，验证程序行为与参考输出的一致性；游戏数据上的重建质量由正式实验评价。

为保持与作者参考输出一致，插值代码保留原实现 `fc>fd>fa>fb` 分支的行为，具体说明见 `srlut_numpy.py` 注释。插值分支或推理内核发生变更时，预训练组和微调组采用同一实现版本重新评价。

## 11. 实验记录与版本管理

- 原始录屏与抽帧作为数据源保留，实际分辨率写入元数据记录。参考数据以原始解码结果为依据，避免重复转码引入额外失真。
- 最终配对清单及视频集合归属固定，三组共用相同 LR/HR 配对。跨机器迁移仅更新路径，保留原划分结果。
- 训练配置、初始权重来源、检查点、导出 JSON、逐图质量与耗时分别归档，实验结果与具体 LUT 来源对应。
- 运行记录包括设备型号、库版本、图像尺寸、预热设置及计时范围。GPU 训练时间与 CPU LUT 推理时间分别统计。
- 代码与实验说明通过 Git 管理。原始视频、完整帧数据、训练权重及批量结果图通过团队共享目录保存；虚拟环境在各运行设备本地建立，依赖通过环境配置记录。
- `main` 分支维护可运行版本，功能开发使用独立分支，合并前完成相应检查。

## 12. 参考方法

- Bicubic Interpolation：本项目使用 Pillow 的 Bicubic 缩放实现。
- Jo, Younghyun; Kim, Seon Joo. *Practical Single-Image Super-Resolution Using Look-Up Table*. CVPR 2021. [官方代码](https://github.com/yhjo09/SR-LUT)。本项目采用 Ours-S。
- Zeng et al. *ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration*. CVPR 2026. [官方代码](https://github.com/Sailor-t/ShiftLUT)。仅作为相关工作参考。

第三方代码、权重和示例素材的来源与使用条件以各自发布仓库为准。
