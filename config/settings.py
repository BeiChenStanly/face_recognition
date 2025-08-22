import torch
import os
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径
DATA_ROOT = './aligned_images_DB'

# 训练超参数
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3 # 初始学习率
WEIGHT_DECAY = 1e-4


# 模型参数
EMBEDDING_SIZE = 512
ARCFACE_S = 30.0
ARCFACE_M = 0.50
EASY_MARGIN = False

# 图像预处理参数
IMAGE_SIZE = 224

# 识别阈值
SIMILARITY_THRESHOLD = 0.5

# 日志和检查点配置
LOG_DIR = 'logs'
CHECKPOINT_DIR = 'checkpoints'
RESULTS_DIR = 'results'

# 创建必要的目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs('weights', exist_ok=True)

# 当前时间戳（用于区分不同运行）
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 检查点保存频率（每隔多少epoch保存一次）
CHECKPOINT_FREQUENCY = 10

# 学习率调度器配置
LR_SCHEDULER = 'step'  # 选项: 'step', 'cosine', 'plateau'
STEP_SIZE = 1          # step调度器的步长
GAMMA = 0.9            # 学习率衰减因子
LR_PATIENCE = 5        # plateau调度器的耐心值
LR_FACTOR = 0.5        # 学习率衰减因子
LR_MIN = 1e-6          # 最小学习率

TEST_DATA_ROOT = './testdata'
FEATURE_EXTRACTOR_WEIGHTS = 'weights/backbone.pth'
DLIB_FILE = 'files/shape_predictor_68_face_landmarks.dat'

IMAGE_PER_SUBJECT = 8 # 每个subject的图像数量

PRELOAD_TO_GPU = False  # 是否预加载到GPU
PRELOAD_TO_CPU = not PRELOAD_TO_GPU  # 是否预加载到CPU
PRELOAD_BATCH_SIZE = 512  # 预加载批次大小

EARLY_STOPPING_PATIENCE = 10  # 早停耐心值