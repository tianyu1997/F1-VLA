# F1-VLA 快速入门指南

本指南帮助您快速开始 F1-VLA 的训练和推理。

---

## 目录

1. [环境安装](#1-环境安装)
2. [数据准备](#2-数据准备)
3. [模型训练](#3-模型训练)
4. [常见问题](#4-常见问题)

---

## 1. 环境安装

### 1.1 系统要求

- Python ≥ 3.10
- PyTorch ≥ 2.6.0
- CUDA ≥ 12.4
- GPU显存 ≥ 24GB（训练推荐48GB）

### 1.2 安装步骤

```bash
# 克隆仓库
git clone https://github.com/InternRobotics/F1-VLA.git
cd F1-VLA

# 创建conda环境
conda create -n f1_vla python=3.10
conda activate f1_vla

# 安装PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    torchcodec==0.2.1 --index-url https://download.pytorch.org/whl/cu124

# 安装F1-VLA
cd f1_vla
pip install -e .

# 修复numpy版本
pip install numpy==1.26.4
```

### 1.3 验证安装

```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "from f1_vla.src.models.memory import KVMemoryBank; print('Memory模块OK')"
```

---

## 2. 数据准备

### 2.1 下载预训练模型

| 模型 | 下载链接 | 说明 |
|------|---------|------|
| F1_pretrain | [HuggingFace](https://huggingface.co/InternRobotics/F1-VLA) | 预训练F1-VLA |
| PaliGemma | [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) | 语言骨干网络 |
| VAE | [vae_ch160v4096z32.pth](https://huggingface.co/FoundationVision/var) | VAR分词器 |

### 2.2 数据格式

```
data/
├── clean/
│   ├── episode_000/
│   │   ├── head/          # 头部相机图像
│   │   ├── wrist/         # 腕部相机图像
│   │   ├── actions.npy    # 动作序列
│   │   └── states.npy     # 状态序列
│   └── ...
└── noisy/                 # 可选：带噪声数据
```

---

## 3. 模型训练

### 3.1 World Model训练（带Memory）

```bash
# 自动检测空闲GPU
./train.sh -a -c f1_vla/config/memory_from_f1pretrain.yaml

# 指定GPU
./train.sh -g 0,1,2,3 -c f1_vla/config/memory_from_f1pretrain.yaml

# 从checkpoint恢复
./train.sh -g 0,1,2,3 -c config.yaml -r outputs/checkpoint-episode-50000
```

### 3.2 Teacher-Student蒸馏

```bash
# 需要2张GPU
./train.sh -g 5,6 -c f1_vla/config/teacher_student_config.yaml -r ""
```

### 3.3 Explorer RL训练

```bash
# Phase 1: 冻结World Model，训练Explorer
./train_explorer.sh -g 5 -p 1

# Phase 2: 对抗训练
./train_explorer.sh -g 5 -p 2
```

### 3.4 监控训练

```bash
# 实时查看日志
tail -f logs/latest_log.log

# 查看GPU使用
watch -n 1 nvidia-smi

# 停止训练
kill $(cat logs/train_pid.txt)
```

---

## 4. 常见问题

### 4.1 显存不足 (OOM)

```yaml
# 解决方案：减小batch_size，增大gradient_accumulation
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# 或减小memory_len
memory_config:
  memory_len: 16  # 从32减到16
```

### 4.2 Loss出现NaN

```yaml
# 检查memory初始化
memory_config:
  init_std: 0.02  # 不要太大

# 降低学习率
learning_rate: 1.0e-5

# 梯度裁剪
max_grad_norm: 1.0
```

### 4.3 数据加载慢

```bash
# 安装视频加速
pip install torchcodec ffmpeg-python

# 增加worker数
dataloader_num_workers: 8
```

---

## 关键配置说明

### Memory配置

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `use_memory` | `False` | 启用KV Memory |
| `memory_len` | `32` | 每层memory槽位数 |
| `bptt_steps` | `4` | BPTT截断步数 |
| `init_std` | `0.02` | 初始化标准差 |

### 训练参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `learning_rate` | `3e-5` | 学习率 |
| `batch_size` | `1` | 每GPU batch大小 |
| `gradient_accumulation_steps` | `8` | 梯度累积步数 |
| `save_episodes` | `240` | 保存checkpoint间隔（episodes） |

---

## 训练时间参考

| 阶段 | GPU配置 | 预计时间 |
|-----|--------|---------|
| World Model | 4× A6000 | 3-5天 |
| Teacher-Student | 2× A6000 | 1-2天 |
| Explorer Phase 1 | 1× A6000 | 1天 |
| Explorer Phase 2 | 1× A6000 | 1-2天 |

---

## 更多资源

- 详细训练指南：[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
- API参考：[docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- Memory实现：[docs/MEMORY_IMPLEMENTATION.md](docs/MEMORY_IMPLEMENTATION.md)

---

*更新日期：2026年1月*
