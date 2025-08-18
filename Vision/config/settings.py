import torch
import os
import time

# ############################# 基础配置 #############################
# PROJECT_NAME = "SwinTransformer_CornerMask"  # 项目名称，用于wandb等
PROJECT_NAME = "SwinTransformer_CornerMask"  # 项目名称，用于wandb等
EXPERIMENT_NAME = "SwinTransformer_CornerMask_standard_training_new"  # 实验名称，标准训练
DEBUG_MODE = False  # 调试模式开关，True时输出详细日志
RANDOM_SEED = 42  # 随机种子，保证实验可复现
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')  # 设备配置，优先使用GPU

# ############################# 日志配置 #############################
# 日志级别: 10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), 50 (CRITICAL)
LOG_LEVEL = 10 if DEBUG_MODE else 20  # 调试模式时使用DEBUG级别，否则INFO级别
LOG_TO_CONSOLE = True  # 是否将日志输出到控制台
LOG_TO_FILE = True  # 是否将日志输出到文件

# ############################# 数据集路径 #############################
DATASET_ROOT_DIR = "/home/visllm/program/KGC/tactile/dataset_cornerpoint3"  # 修改为相对路径
TRAIN_DIR = "train"  # 训练集子目录名
TEST_DIR = "test"  # 测试集子目录名 (通常用于最终评估，验证集从训练集中划分)
VAL_SPLIT_RATIO = 0.1  # 验证集比例

# 数据加载器配置
NUM_WORKERS = 4  # 数据加载器使用的工作线程数
PIN_MEMORY = True  # 是否使用固定内存（对GPU训练有加速效果）

# ############################# 模型与训练参数 #############################
# ## 模型参数 ##
NUM_CORNERS = 4  # 预测的角点数量 (x1, y1, x2, y2) -> 对应 target_corners.shape[1]
PRETRAINED_BACKBONE = True  # 是否加载预训练的 EfficientNet 权重
PRETRAINED_MODEL_PATH = '/home/visllm/program/KGC/tactile/Tactile-Vision/model/pretrained/pytorch_model.bin'  # 本地预训练模型路径

# ## timm 模型配置 - 使用EfficientNet-B0 ##
MODEL_NAME = 'swin_base_patch4_window7_224.ms_in22k_ft_in1k'  # 使用 EfficientNet-B0 模型
FEATURE_FUSION = True  # 简化模型，关闭复杂的特征融合
LIGHTWEIGHT_FUSION = False  # 使用轻量化处理
USE_CHECKPOINT = False  # 标准训练不需要梯度检查点，如果显存不足可以启用

# ## 训练超参数 ##
BATCH_SIZE = 16  # 降低批量大小以适应更复杂的模型
LEARNING_RATE = 1e-4  # 降低学习率提高训练稳定性
NUM_EPOCHS = 100  # 标准训练轮数
START_EPOCH = 0  # 用于恢复训练时的起始 epoch
WEIGHT_DECAY = 1e-4  # 降低权重衰减
OPTIMIZER_TYPE = 'AdamW'  # 使用AdamW优化器，适合 Transformer 架构
SCHEDULER_TYPE = 'CosineAnnealingLR'  # 使用余弦退火学习率调度
SCHEDULER_STEP_SIZE = 30  # 对于 StepLR，学习率下降的epoch间隔
SCHEDULER_GAMMA = 0.1  # 对于 StepLR，学习率下降的乘法因子
GRAD_CLIP_MAX_NORM = 1.0  # 标准梯度裁剪值

# 标准正则化设置
DROPOUT_RATE = 0.1  # 标准Dropout比率

# ## 损失函数权重 ##
MASK_LOSS_WEIGHT = 1.0  # 掩码损失的权重

# ############################# 图像参数 #############################
IMAGE_WIDTH = 224  # 输入图像宽度，与Swin Transformer预训练尺寸匹配
IMAGE_HEIGHT = 224  # 输入图像高度，与Swin Transformer预训练尺寸匹配

# 图像归一化参数 (使用ImageNet的均值和标准差)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ############################# 输出与日志 #############################
BASE_OUTPUT_DIR = "outputs"  # 所有输出的基础目录
MODEL_SAVE_DIR = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT_NAME,
                              "checkpoints")  # 模型权重保存目录
LOG_DIR = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT_NAME, "logs")  # 训练日志保存目录
VISUALIZATION_DIR = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT_NAME,
                                 "visualizations")  # 可视化结果保存目录

# 新增: 日志文件名
LOG_FILE_NAME = "training_session.log"  # 将保存在 LOG_DIR 下

# 每 N 个 epoch 保存一次模型
SAVE_MODEL_EVERY_N_EPOCHS = 10  # 标准保存频率
VALIDATE_EVERY_N_EPOCHS = 1  # 每个epoch都验证

# ## 保存最佳模型的配置 ##
SAVE_BEST_MODEL_ONLY = True  # 只保存最佳模型
BEST_MODEL_METRIC = "val_mask_iou"  # 使用IoU作为最佳模型指标
BEST_MODEL_METRIC_MODE = "max"  # IoU越大越好

# ## 恢复训练配置 ##
RESUME_TRAINING = False  # 是否从检查点恢复训练
CHECKPOINT_PATH = None  # 恢复训练时加载的检查点路径

# ## 早停配置 ##
EARLY_STOPPING_PATIENCE = 15  # 早停耐心值
EARLY_STOPPING_MIN_DELTA = 0.001  # 最小改善量

# ## WandB 配置 ##
USE_WANDB = True  # 是否使用WandB进行实验跟踪
WANDB_PROJECT = PROJECT_NAME  # WandB 项目名称
WANDB_ENTITY = None  # WandB 实体
WANDB_RUN_NAME = f"{EXPERIMENT_NAME}_{time.strftime('%Y%m%d_%H%M%S')}" if EXPERIMENT_NAME else None
WANDB_LOG_IMAGES = True  # 在 WandB 中记录图像可视化
WANDB_LOG_MODEL = "best"  # 记录最佳模型到WandB

# ## 可视化配置 ##
VISUALIZE_DATALOADER_SAMPLES = True
NUM_DATALOADER_SAMPLES_TO_VISUALIZE = 4  # 标准可视化样本数量

# 在验证集上，每 N 个 epoch 可视化一次预测结果
VISUALIZE_PREDICTIONS_EVERY_N_EPOCHS = 5  # 标准可视化频率
NUM_VALIDATION_PREDICTIONS_TO_VISUALIZE = 4  # 标准验证可视化样本数量

# 在训练过程中记录的指标（只包含掩码相关指标）
METRICS_TO_TRACK = [
    "train_total_loss", "train_mask_loss", "val_total_loss", "val_mask_loss",
    "val_mask_iou", "val_mask_dice", "val_pixel_accuracy", "learning_rate"
]

# ############################# 推理设置 #############################
INFERENCE_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT_NAME,
                                    "checkpoints", "best_model.pth")
INFERENCE_INPUT_DIR = "/home/visllm/program/KGC/tactile/Tactile-Vision/light_scene"
INFERENCE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT_NAME,
                                    "inference_results", "light_scene")
INFERENCE_IMG_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
INFERENCE_MAX_OBJECTS = 20
INFERENCE_MIN_CONTOUR_AREA = 20

# 推理时亮度调整设置 - 提升阴暗环境下的性能
INFERENCE_BRIGHTNESS_ENHANCEMENT = True  # 是否启用亮度增强
INFERENCE_BRIGHTNESS_FACTOR = 1.1  # 亮度增强因子 (1.0=原始亮度, >1.0=增亮, <1.0=变暗)
INFERENCE_CONTRAST_FACTOR = 1.3  # 对比度增强因子 (1.0=原始对比度)
INFERENCE_GAMMA_CORRECTION = 0.8  # 伽马校正值 (<1.0=增亮, >1.0=变暗, 1.0=不变)

# ############################# 掩码后处理设置 #############################
MASK_SMOOTH_EDGES = True  # 启用掩码边缘平滑
MASK_SMOOTH_TYPE = 'gentle'  # 平滑类型: 'gentle'(温和), 'moderate'(中等), 'aggressive'(激进)

# 详细的平滑参数配置
MASK_GAUSSIAN_SIGMA = 1.0  # 高斯模糊标准差，用于平滑边缘（0表示禁用）
MASK_MORPH_KERNEL_SIZE = 3  # 形态学操作的核大小（奇数，0表示禁用）
MASK_MEDIAN_KERNEL_SIZE = 3  # 中值滤波的核大小（奇数，0表示禁用）

# 精细控制各种平滑操作的启用
MASK_ENABLE_GAUSSIAN_BLUR = True  # 是否启用高斯模糊
MASK_ENABLE_MORPHOLOGY = True  # 是否启用形态学操作（开运算和闭运算）
MASK_ENABLE_MEDIAN_FILTER = True  # 是否启用中值滤波

# 预设平滑配置（根据MASK_SMOOTH_TYPE自动设置上述参数）
MASK_SMOOTH_PRESETS = {
    'gentle': {
        'gaussian_sigma': 0.8,
        'morph_kernel_size': 3,
        'median_kernel_size': 3,
        'enable_gaussian': True,
        'enable_morphology': True,
        'enable_median': False
    },
    'moderate': {
        'gaussian_sigma': 1.2,
        'morph_kernel_size': 5,
        'median_kernel_size': 3,
        'enable_gaussian': True,
        'enable_morphology': True,
        'enable_median': True
    },
    'aggressive': {
        'gaussian_sigma': 1.5,
        'morph_kernel_size': 7,
        'median_kernel_size': 5,
        'enable_gaussian': True,
        'enable_morphology': True,
        'enable_median': True
    }
}

# ############################# 数据增强设置 #############################
# 基础开关
ENABLE_DATA_AUGMENTATION = True  # 启用数据增强
AUGMENTATION_PROBABILITY = 0.8  # 降低整体增强概率

# 几何变换 - 触觉图像对几何变换相对不敏感
ROTATION_RANGE = (-10, 10)  # 减小旋转范围，避免过度变形
SCALE_RANGE = (0.9, 1.1)  # 减小缩放范围，保持原始比例
TRANSLATE_RANGE = (0.05, 0.05)  # 减小平移范围
HORIZONTAL_FLIP_PROB = 0.3  # 降低翻转概率
VERTICAL_FLIP_PROB = 0.05  # 大幅降低垂直翻转概率
ELASTIC_TRANSFORM_PROB = 0.1  # 降低弹性变形概率

# 高级几何变换 - 谨慎使用
GRID_DISTORTION_PROB = 0.1  # 降低网格失真概率
OPTICAL_DISTORTION_PROB = 0.1  # 降低光学失真概率
PERSPECTIVE_PROB = 0.15  # 降低透视变换概率

# 颜色增强 - 触觉图像对颜色变化敏感，要温和
BRIGHTNESS_RANGE = (0.9, 1.1)  # 减小亮度变化范围
CONTRAST_RANGE = (0.9, 1.1)  # 减小对比度变化范围
SATURATION_RANGE = (0.95, 1.05)  # 大幅减小饱和度变化范围
HUE_RANGE = (-0.05, 0.05)  # 减小色调变化范围

# 噪声和模糊 - 适度模拟环境干扰
GAUSSIAN_NOISE_PROB = 0.15  # 降低高斯噪声概率
GAUSSIAN_NOISE_VAR_RANGE = (3, 20)  # 减小噪声方差范围
MOTION_BLUR_PROB = 0.08  # 降低运动模糊概率
GAUSSIAN_BLUR_PROB = 0.08  # 降低高斯模糊概率
BLUR_KERNEL_SIZE_RANGE = (3, 5)  # 减小模糊核大小范围

# 遮挡增强 - 大幅降低，避免遮挡关键特征
CUTOUT_PROB = 0.1  # 大幅降低Cutout概率
CUTOUT_NUM_HOLES = (1, 1)  # 只允许一个孔
CUTOUT_HOLE_SIZE_RANGE = (0.01, 0.05)  # 大幅减小孔大小范围
COARSE_DROPOUT_PROB = 0.08  # 降低粗糙遮挡概率
SHADOW_PROB = 0.05  # 大幅降低阴影概率

# 触觉专用增强 - 保持温和
TACTILE_PRESSURE_RANGE = (0.95, 1.05)  # 减小触觉压力范围
TACTILE_SURFACE_ROUGHNESS = 0.15  # 降低表面粗糙度概率
TACTILE_EDGE_ENHANCEMENT = 0.2  # 降低边缘增强概率

# 高级增强策略
ADAPTIVE_AUGMENTATION = True  # 启用自适应增强
MULTI_OBJECT_AUGMENTATION = True  # 多目标增强

# 测试时增强 (TTA) - 推理时可以尝试
USE_TEST_TIME_AUGMENTATION = True  # 推理时启用TTA
TTA_SCALES = [0.95, 1.0, 1.05]  # 减小TTA尺度范围
TTA_FLIPS = [False, True]  # TTA翻转

# 可视化配置
VISUALIZE_AUGMENTATION_STEPS = True  # 可视化增强步骤
NUM_AUGMENTATION_EXAMPLES = 4  # 标准增强示例数量
