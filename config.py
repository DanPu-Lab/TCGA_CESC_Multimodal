import os

# ---------------------- 路径配置 ----------------------
# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 预处理后的数据存放目录（需要包含 svs/, tsv/, labels/, splits/ 等子文件夹）
DATA_DIR = os.path.join(BASE_DIR, "data/processed")

# 原始数据存放目录（SVS和TSV文件的原始位置，预处理脚本会从此处读取）
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")

# 所有结果输出目录（模型权重、评估结果、分析结果等）
RESULT_DIR = os.path.join(BASE_DIR, "result")

# ---------------------- 预处理参数 ----------------------
PATCH_SIZE = 224               # 图像块尺寸（像素）
MIN_PATCHES = 100              # 每个样本最少需要的patch数量
NUM_PATCHES_SAMPLE = 50        # 训练时每个样本随机采样的patch数量

# ---------------------- 训练参数 ----------------------
BATCH_SIZE = 16                # 批次大小（可根据GPU内存调整）
NUM_EPOCHS = 50                # 最大训练轮数
LEARNING_RATE = 1e-4           # 初始学习率
WEIGHT_DECAY = 1e-5            # 权重衰减
PATIENCE = 10                  # 早停耐心值

# ---------------------- 设备配置 ----------------------
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- 其他 ----------------------
RANDOM_SEED = 42               # 随机种子，保证可复现性