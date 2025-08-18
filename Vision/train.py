import os
import sys
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from datetime import datetime

# 导入项目模块
from config import settings
from data.dataset import CornerPointDataset
from data.transforms import Preprocessing
from model.network import CornerMaskModel
from utils.losses import CombinedLoss, MaskLoss  # 导入更新后的损失函数
from utils.metrics import intersection_over_union, dice_coefficient, pixel_accuracy  # 只导入掩码相关指标
from utils.visualize import visualize_validation_masks_with_postprocessing  # 验证阶段掩码可视化

os.environ["WANDB_MODE"] = "offline"


def setup_logger():
    """
    设置日志记录器，支持同时输出到控制台和文件
    """
    # 创建日志目录（如果不存在）
    if settings.LOG_TO_FILE and not os.path.exists(settings.LOG_DIR):
        os.makedirs(settings.LOG_DIR, exist_ok=True)

    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(settings.LOG_LEVEL)

    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 设置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # 添加控制台处理器
    if settings.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(settings.LOG_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 添加文件处理器
    if settings.LOG_TO_FILE:
        log_file = os.path.join(settings.LOG_DIR, settings.LOG_FILE_NAME)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(settings.LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed):
    """
    设置随机种子以确保实验可复现性
    
    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_directories():
    """
    创建必要的输出目录
    """
    os.makedirs(settings.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    os.makedirs(settings.VISUALIZATION_DIR, exist_ok=True)
    logging.info(f"已创建输出目录: {settings.BASE_OUTPUT_DIR}")


def get_optimizer(model_parameters):
    """
    根据设置创建优化器
    
    Args:
        model_parameters: 模型参数
        
    Returns:
        torch.optim.Optimizer: 配置好的优化器
    """
    if settings.OPTIMIZER_TYPE == 'Adam':
        return optim.Adam(model_parameters,
                          lr=settings.LEARNING_RATE,
                          weight_decay=settings.WEIGHT_DECAY)
    elif settings.OPTIMIZER_TYPE == 'AdamW':
        return optim.AdamW(model_parameters,
                           lr=settings.LEARNING_RATE,
                           weight_decay=settings.WEIGHT_DECAY)
    elif settings.OPTIMIZER_TYPE == 'SGD':
        return optim.SGD(model_parameters,
                         lr=settings.LEARNING_RATE,
                         momentum=0.9,
                         weight_decay=settings.WEIGHT_DECAY)
    else:
        raise ValueError(f"不支持的优化器类型: {settings.OPTIMIZER_TYPE}")


def get_scheduler(optimizer):
    """
    根据设置创建学习率调度器
    
    Args:
        optimizer: 优化器
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: 配置好的学习率调度器，或者None
    """
    if settings.SCHEDULER_TYPE == 'StepLR':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=settings.SCHEDULER_STEP_SIZE,
            gamma=settings.SCHEDULER_GAMMA)
    elif settings.SCHEDULER_TYPE == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=settings.NUM_EPOCHS)
    elif not settings.SCHEDULER_TYPE:
        return None
    else:
        raise ValueError(f"不支持的学习率调度器类型: {settings.SCHEDULER_TYPE}")


def save_checkpoint(model,
                    optimizer,
                    scheduler,
                    epoch,
                    metrics,
                    is_best=False):
    """
    保存模型检查点
    
    Args:
        model (nn.Module): 模型
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): 学习率调度器
        epoch (int): 当前训练轮次
        metrics (dict): 评估指标
        is_best (bool): 是否为最佳模型
    """
    # 创建检查点字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # 保存最新检查点
    latest_path = os.path.join(settings.MODEL_SAVE_DIR, 'latest_model.pth')
    torch.save(checkpoint, latest_path)
    logging.info(f"已保存最新检查点到 {latest_path}")

    # 如果是最佳模型，保存到特定文件
    if is_best:
        best_path = os.path.join(settings.MODEL_SAVE_DIR, 'best_model.pth')
        torch.save(checkpoint, best_path)
        logging.info(f"已保存最佳模型到 {best_path}")

    # 如果配置了每N个epoch保存一次，则保存epoch检查点
    if settings.SAVE_MODEL_EVERY_N_EPOCHS > 0 and epoch % settings.SAVE_MODEL_EVERY_N_EPOCHS == 0:
        epoch_path = os.path.join(settings.MODEL_SAVE_DIR,
                                  f'model_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        logging.info(f"已保存第 {epoch} 轮模型到 {epoch_path}")


def load_checkpoint(model, optimizer, scheduler):
    """
    从检查点恢复训练状态
    
    Args:
        model (nn.Module): 模型
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): 学习率调度器
        
    Returns:
        int: 恢复的起始epoch
        dict: 恢复的指标
    """
    # 确定要加载的检查点路径
    checkpoint_path = settings.CHECKPOINT_PATH

    # 如果未指定检查点路径但启用了恢复训练，尝试加载最新检查点
    if checkpoint_path is None and settings.RESUME_TRAINING:
        checkpoint_path = os.path.join(settings.MODEL_SAVE_DIR,
                                       'latest_model.pth')

    if not os.path.exists(checkpoint_path):
        logging.warning(f"检查点文件 {checkpoint_path} 不存在，将从头开始训练")
        return 0, {}

    # 加载检查点
    logging.info(f"正在加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=settings.DEVICE)

    # 恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])

    # 恢复优化器状态
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 恢复调度器状态（如果存在）
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 恢复epoch和指标
    start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
    metrics = checkpoint.get('metrics', {})

    logging.info(f"已恢复到第 {start_epoch-1} 轮的训练状态")
    return start_epoch, metrics


# visualize_samples函数已移除，只保留验证阶段的掩码可视化


def setup_wandb():
    """
    设置并初始化wandb
    
    Returns:
        dict: wandb配置
    """
    if not settings.USE_WANDB:
        logging.info("未启用wandb，将不会记录实验")
        return {}

    # 准备wandb配置
    wandb_config = {
        "batch_size": settings.BATCH_SIZE,
        "learning_rate": settings.LEARNING_RATE,
        "epochs": settings.NUM_EPOCHS,
        "optimizer": settings.OPTIMIZER_TYPE,
        "scheduler": settings.SCHEDULER_TYPE,
        "mask_loss_weight": settings.MASK_LOSS_WEIGHT,
        "image_size": f"{settings.IMAGE_WIDTH}x{settings.IMAGE_HEIGHT}",
        "seed": settings.RANDOM_SEED,
        "task": "mask_segmentation"  # 标记为掩码分割任务
    }

    # 初始化wandb
    try:
        wandb.init(project=settings.WANDB_PROJECT,
                   entity=settings.WANDB_ENTITY,
                   name=settings.WANDB_RUN_NAME,
                   config=wandb_config)
        logging.info(
            f"已初始化wandb，项目: {settings.WANDB_PROJECT}, 运行名: {settings.WANDB_RUN_NAME}"
        )
    except Exception as e:
        logging.error(f"初始化wandb失败: {e}")
        return {}

    return wandb_config


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    
    Args:
        model (nn.Module): 模型
        train_loader (DataLoader): 训练数据加载器
        criterion (nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        device (torch.device): 设备
        epoch (int): 当前epoch
        
    Returns:
        dict: 训练指标
    """
    model.train()
    epoch_losses = {"total": 0.0, "mask_loss": 0.0}

    # 使用tqdm创建进度条
    progress_bar = tqdm(train_loader,
                        desc=f"训练 Epoch {epoch+1}/{settings.NUM_EPOCHS}")

    for batch_idx, batch in enumerate(progress_bar):
        # 将数据移到设备
        images = batch['image'].to(device)
        target_masks = batch['mask'].to(device)

        # 清除梯度
        optimizer.zero_grad()

        # 前向传播 - 现在模型只输出 pred_mask_logits
        pred_mask_logits = model(images)

        # 计算损失 - 只有掩码损失
        losses = criterion(pred_mask_logits, target_masks)
        total_loss = losses['total']

        # 反向传播
        total_loss.backward()

        # 梯度裁剪（如果配置）
        if settings.GRAD_CLIP_MAX_NORM is not None:
            nn.utils.clip_grad_norm_(model.parameters(),
                                     settings.GRAD_CLIP_MAX_NORM)

        # 优化步骤
        optimizer.step()

        # 更新累积损失
        epoch_losses["total"] += total_loss.item()
        epoch_losses["mask_loss"] += losses["mask_loss"].item()

        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        postfix_dict = {
            'total_loss': total_loss.item(),
            'mask_loss': losses["mask_loss"].item(),
            'lr': current_lr
        }
        progress_bar.set_postfix(postfix_dict)

        # 详细日志记录（调试模式）
        if settings.DEBUG_MODE and batch_idx % 10 == 0:
            log_message = (f"训练批次 {batch_idx}/{len(train_loader)}, "
                           f"损失: {total_loss.item():.4f}, "
                           f"掩码损失: {losses['mask_loss'].item():.4f}")
            logging.debug(log_message)

    # 计算平均损失
    num_batches = len(train_loader)
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

    # 记录训练指标
    train_metrics = {
        "train_total_loss": avg_losses["total"],
        "train_mask_loss": avg_losses["mask_loss"],
    }

    # 日志记录训练结果
    log_message_epoch = (f"训练 Epoch {epoch+1}/{settings.NUM_EPOCHS} 完成: "
                         f"总损失: {train_metrics['train_total_loss']:.4f}, "
                         f"掩码损失: {train_metrics['train_mask_loss']:.4f}")
    logging.info(log_message_epoch)

    return train_metrics


def validate(model, val_loader, criterion, device, epoch):
    """
    在验证集上评估模型
    
    Args:
        model (nn.Module): 模型
        val_loader (DataLoader): 验证数据加载器
        criterion (nn.Module): 损失函数
        device (torch.device): 设备
        epoch (int): 当前epoch
        
    Returns:
        tuple: (验证指标字典, 用于可视化的批次数据)
    """
    model.eval()
    val_losses = {"total": 0.0, "mask_loss": 0.0}

    # 掩码评估指标累积
    total_iou = 0.0
    total_dice = 0.0
    total_accuracy = 0.0
    num_samples = 0

    visualization_batch_data = None

    with torch.no_grad():
        progress_bar = tqdm(val_loader,
                            desc=f"验证 Epoch {epoch+1}/{settings.NUM_EPOCHS}")

        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备
            images = batch['image'].to(device)
            target_masks = batch['mask'].to(device)

            # 前向传播 - 只返回掩码 logits
            pred_mask_logits = model(images)

            # 计算损失 - 只计算掩码损失
            losses = criterion(pred_mask_logits, target_masks)

            # 更新累积损失
            val_losses["total"] += losses["total"].item()
            val_losses["mask_loss"] += losses["mask_loss"].item()

            # 计算掩码评估指标
            pred_mask_probs = torch.sigmoid(pred_mask_logits)

            # 计算IoU
            batch_iou = intersection_over_union(pred_mask_probs, target_masks)
            total_iou += batch_iou.sum().item()

            # 计算Dice系数
            batch_dice = dice_coefficient(pred_mask_probs, target_masks)
            total_dice += batch_dice.sum().item()

            # 计算像素准确率
            batch_accuracy = pixel_accuracy(pred_mask_probs, target_masks)
            total_accuracy += batch_accuracy.sum().item()

            num_samples += images.size(0)

            # 更新进度条
            postfix_dict_val = {
                'val_loss': losses["total"].item(),
                'iou': batch_iou.mean().item(),
                'dice': batch_dice.mean().item()
            }
            progress_bar.set_postfix(postfix_dict_val)

            # 保存第一个批次用于可视化
            if batch_idx == 0:
                visualization_batch_data = (images.cpu(), target_masks.cpu(),
                                            pred_mask_logits.cpu())

            # 详细日志记录（调试模式）
            if settings.DEBUG_MODE and batch_idx % 10 == 0:
                log_message_val = (f"验证批次 {batch_idx}/{len(val_loader)}, "
                                   f"损失: {losses['total'].item():.4f}, "
                                   f"IoU: {batch_iou.mean().item():.4f}, "
                                   f"Dice: {batch_dice.mean().item():.4f}")
                logging.debug(log_message_val)

    # 计算平均损失和指标
    num_batches = len(val_loader)
    avg_losses = {k: v / num_batches for k, v in val_losses.items()}

    # 计算平均掩码指标
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_accuracy = total_accuracy / num_samples

    # 组合指标字典
    val_metrics_dict = {
        "val_total_loss": avg_losses["total"],
        "val_mask_loss": avg_losses["mask_loss"],
        "val_mask_iou": avg_iou,
        "val_mask_dice": avg_dice,
        "val_pixel_accuracy": avg_accuracy
    }

    # 日志记录验证结果
    log_message_epoch_val = (
        f"验证 Epoch {epoch+1}/{settings.NUM_EPOCHS} 完成: "
        f"总损失: {val_metrics_dict['val_total_loss']:.4f}, "
        f"掩码损失: {val_metrics_dict['val_mask_loss']:.4f}, "
        f"掩码IoU: {val_metrics_dict['val_mask_iou']:.4f}, "
        f"掩码Dice: {val_metrics_dict['val_mask_dice']:.4f}")
    logging.info(log_message_epoch_val)

    return val_metrics_dict, visualization_batch_data


def visualize_validation_results(visualization_batch, epoch):
    """
    可视化验证结果，使用掩码预测和后处理提取的角点
    
    Args:
        visualization_batch (tuple): 包含图像、目标掩码和预测掩码logits的元组
        epoch (int): 当前epoch
    """
    if visualization_batch is None:
        logging.warning("无可视化数据可用")
        return

    images, true_masks, pred_mask_logits = visualization_batch

    # 创建可视化图像并保存
    viz_path = os.path.join(settings.VISUALIZATION_DIR,
                            f'val_predictions_epoch_{epoch+1}.png')

    # 使用新的验证阶段可视化函数
    try:
        visualize_validation_masks_with_postprocessing(
            images=images[:settings.NUM_VALIDATION_PREDICTIONS_TO_VISUALIZE],
            true_masks=true_masks[:settings.
                                  NUM_VALIDATION_PREDICTIONS_TO_VISUALIZE],
            pred_masks=
            pred_mask_logits[:settings.
                             NUM_VALIDATION_PREDICTIONS_TO_VISUALIZE],
            save_path=viz_path,
            show_corners=True  # 显示后处理提取的角点
        )
        logging.info(f"已保存验证预测可视化到 {viz_path}")
    except Exception as e:
        logging.error(f"验证可视化失败: {e}")
        # 如果角点提取失败，尝试只显示掩码
        try:
            visualize_validation_masks_with_postprocessing(
                images=images[:settings.
                              NUM_VALIDATION_PREDICTIONS_TO_VISUALIZE],
                true_masks=true_masks[:settings.
                                      NUM_VALIDATION_PREDICTIONS_TO_VISUALIZE],
                pred_masks=
                pred_mask_logits[:settings.
                                 NUM_VALIDATION_PREDICTIONS_TO_VISUALIZE],
                save_path=viz_path,
                show_corners=False  # 只显示掩码
            )
            logging.info(f"已保存验证掩码可视化到 {viz_path} (跳过角点)")
        except Exception as e2:
            logging.error(f"掩码可视化也失败: {e2}")
            return

    # 如果启用wandb并配置了图像记录，则上传图像
    if settings.USE_WANDB and settings.WANDB_LOG_IMAGES:
        try:
            wandb_img = wandb.Image(
                viz_path, caption=f"Epoch {epoch+1} Validation Predictions")
            wandb.log({"validation_predictions": wandb_img}, step=epoch + 1)
        except Exception as e:
            logging.error(f"上传图像到wandb失败: {e}")


def is_best_model(current_metrics, best_metrics):
    """
    检查当前模型是否优于之前的最佳模型
    
    Args:
        current_metrics (dict): 当前模型的指标
        best_metrics (dict): 最佳模型的指标
        
    Returns:
        bool: 如果当前模型更好则返回True
    """
    # 如果没有最佳指标记录，当前模型就是最佳的
    if not best_metrics:
        return True

    metric_key = settings.BEST_MODEL_METRIC

    # 检查指标是否存在
    if metric_key not in current_metrics:
        logging.warning(f"指定的指标 '{metric_key}' 不在当前指标中，无法比较")
        return False

    # 根据模式（越大越好或越小越好）比较指标值
    current_value = current_metrics[metric_key]
    best_value = best_metrics.get(
        metric_key,
        float('inf')
        if settings.BEST_MODEL_METRIC_MODE == "min" else float('-inf'))

    if settings.BEST_MODEL_METRIC_MODE == "max":
        return current_value > best_value
    else:  # "min" mode
        return current_value < best_value


def main():
    """
    主函数，包含整个训练流程
    """
    # 设置日志记录器
    logger = setup_logger()
    logging.info("=" * 50)
    logging.info(f"开始训练，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(
        f"项目: {settings.PROJECT_NAME}, 实验: {settings.EXPERIMENT_NAME}")
    logging.info(f"调试模式: {'开启' if settings.DEBUG_MODE else '关闭'}")
    logging.info(f"设备: {settings.DEVICE}")
    logging.info(f"任务: 掩码分割")

    # 设置随机种子
    set_seed(settings.RANDOM_SEED)
    logging.info(f"已设置随机种子: {settings.RANDOM_SEED}")

    # 准备输出目录
    prepare_directories()

    # 初始化数据预处理
    train_transform = Preprocessing(mode='train')
    val_transform = Preprocessing(mode='train')  # 验证也使用train模式，因为不需要角点处理
    logging.info("已初始化训练和验证的数据预处理器")

    # 加载数据集
    logging.info(f"正在加载数据集，路径: {settings.DATASET_ROOT_DIR}")
    full_train_dataset_raw = CornerPointDataset(mode='train', transform=None)
    test_dataset = CornerPointDataset(mode='test', transform=val_transform)

    # 从训练集中分割出验证集
    train_size = int(
        len(full_train_dataset_raw) * (1 - settings.VAL_SPLIT_RATIO))
    val_size = len(full_train_dataset_raw) - train_size

    train_indices, val_indices = random_split(
        range(len(full_train_dataset_raw)), [train_size, val_size],
        generator=torch.Generator().manual_seed(settings.RANDOM_SEED))

    train_dataset_transformed = CornerPointDataset(mode='train',
                                                   transform=train_transform)
    train_dataset_transformed.image_filenames = [
        full_train_dataset_raw.image_filenames[idx]
        for idx in train_indices.indices
    ]

    val_dataset_transformed = CornerPointDataset(mode='train',
                                                 transform=val_transform)
    val_dataset_transformed.image_filenames = [
        full_train_dataset_raw.image_filenames[idx]
        for idx in val_indices.indices
    ]

    logging.info(
        f"数据集大小 - 训练: {len(train_dataset_transformed)}, 验证: {len(val_dataset_transformed)}, 测试: {len(test_dataset)}"
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset_transformed,
                              batch_size=settings.BATCH_SIZE,
                              shuffle=True,
                              num_workers=settings.NUM_WORKERS,
                              pin_memory=settings.PIN_MEMORY)

    val_loader = DataLoader(val_dataset_transformed,
                            batch_size=settings.BATCH_SIZE,
                            shuffle=False,
                            num_workers=settings.NUM_WORKERS,
                            pin_memory=settings.PIN_MEMORY)

    # 训练前可视化已取消，只在验证阶段进行掩码可视化
    # 这样可以避免读取角点数据，专注于掩码分割任务
    if settings.VISUALIZE_DATALOADER_SAMPLES:
        logging.info("跳过训练前可视化，将在验证阶段进行掩码预测可视化")

    # 初始化模型
    model = CornerMaskModel().to(settings.DEVICE)

    # 初始化损失函数 - 只使用掩码损失
    criterion = CombinedLoss().to(settings.DEVICE)
    logging.info(
        f"已初始化损失函数 CombinedLoss (只包含 MaskLoss)，掩码权重: {settings.MASK_LOSS_WEIGHT}"
    )

    # 初始化优化器
    optimizer = get_optimizer(model.parameters())
    logging.info(
        f"已初始化优化器: {settings.OPTIMIZER_TYPE}, 学习率: {settings.LEARNING_RATE}")

    # 初始化学习率调度器
    scheduler = get_scheduler(optimizer)
    if scheduler:
        scheduler_name = settings.SCHEDULER_TYPE
        logging.info(f"已初始化学习率调度器: {scheduler_name}")
    else:
        logging.info("未使用学习率调度器")

    # 恢复训练（如果启用）
    start_epoch = settings.START_EPOCH
    best_metrics = {}
    if settings.RESUME_TRAINING:
        start_epoch, best_metrics = load_checkpoint(model, optimizer,
                                                    scheduler)

    # 设置wandb
    if settings.USE_WANDB:
        wandb_config = setup_wandb()
        # 记录模型图
        if settings.DEBUG_MODE:
            try:
                wandb.watch(model, log="all")
            except Exception as e:
                logging.error(f"wandb.watch失败: {e}")

    # 训练循环
    logging.info(f"开始训练，共 {settings.NUM_EPOCHS} 轮，从第 {start_epoch+1} 轮开始")
    for epoch in range(start_epoch, settings.NUM_EPOCHS):
        epoch_start_time = time.time()

        # 训练一个epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer,
                                    settings.DEVICE, epoch)

        # 验证
        if (epoch + 1) % settings.VALIDATE_EVERY_N_EPOCHS == 0:
            # 验证模型
            val_metrics, visualization_batch = validate(
                model, val_loader, criterion, settings.DEVICE, epoch)

            # 可视化预测结果
            if settings.VISUALIZE_PREDICTIONS_EVERY_N_EPOCHS > 0 and (
                    epoch +
                    1) % settings.VISUALIZE_PREDICTIONS_EVERY_N_EPOCHS == 0:
                visualize_validation_results(visualization_batch, epoch)

            # 合并训练和验证指标
            all_metrics = {**train_metrics, **val_metrics}

            # 检查是否为最佳模型
            current_is_best = is_best_model(val_metrics, best_metrics)
            if current_is_best:
                best_metrics = val_metrics.copy()
                logging.info(f"发现新的最佳模型！指标 '{settings.BEST_MODEL_METRIC}': "
                             f"{val_metrics[settings.BEST_MODEL_METRIC]:.4f}")

            # 记录当前学习率
            if scheduler:
                all_metrics["learning_rate"] = scheduler.get_last_lr()[0]
            else:
                all_metrics["learning_rate"] = settings.LEARNING_RATE

            # 保存检查点
            save_checkpoint(model,
                            optimizer,
                            scheduler,
                            epoch,
                            all_metrics,
                            is_best=current_is_best)

            # 记录指标到wandb
            if settings.USE_WANDB:
                # 只记录跟踪的指标
                metrics_to_log = {
                    k: v
                    for k, v in all_metrics.items()
                    if k in settings.METRICS_TO_TRACK
                }
                wandb.log(metrics_to_log, step=epoch + 1)

        # 更新学习率调度器
        if scheduler:
            scheduler.step()

        # 计算并记录本轮用时
        epoch_time = time.time() - epoch_start_time
        logging.info(f"第 {epoch+1} 轮用时: {epoch_time:.2f}秒")

    # 训练结束
    logging.info(f"训练完成，共 {settings.NUM_EPOCHS} 轮")

    # 保存最终模型
    final_model_path = os.path.join(settings.MODEL_SAVE_DIR, 'final_model.pth')
    torch.save(
        {
            'epoch': settings.NUM_EPOCHS - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': best_metrics
        }, final_model_path)
    logging.info(f"已保存最终模型到 {final_model_path}")

    # 记录最佳指标
    logging.info("训练过程中的最佳指标:")
    for k, v in best_metrics.items():
        logging.info(f"  {k}: {v:.4f}")

    # 关闭wandb
    if settings.USE_WANDB:
        try:
            # 如果启用了模型记录
            if settings.WANDB_LOG_MODEL == "all" or settings.WANDB_LOG_MODEL == "best":
                best_model_path = os.path.join(settings.MODEL_SAVE_DIR,
                                               'best_model.pth')
                artifact = wandb.Artifact(f"{settings.EXPERIMENT_NAME}_model",
                                          type="model")
                artifact.add_file(best_model_path)
                wandb.log_artifact(artifact)

            wandb.finish()
            logging.info("已关闭wandb")
        except Exception as e:
            logging.error(f"关闭wandb时出错: {e}")

    logging.info(
        f"整个训练流程已完成，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
