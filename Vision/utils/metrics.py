import torch
import numpy as np
import sys
import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 如果项目根目录不在sys.path中，则添加它
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import settings
# 导入 mask_to_longest_edge_corners
from .postprocessing import mask_to_longest_edge_corners


def corner_distance(pred_corners,
                    true_corners,
                    normalized=True,
                    image_width=None,
                    image_height=None):
    """
    计算预测角点和真实角点之间的欧几里得距离。
    
    Args:
        pred_corners (torch.Tensor): 预测角点坐标，形状为 [B, N*2]（例如 [B,4] for [x1,y1,x2,y2]）
        true_corners (torch.Tensor): 真实角点坐标，形状为 [B, N*2]
        normalized (bool): 如果为True，则假设坐标已归一化到 [0, 1] 范围。
        image_width (int, optional): 图像宽度，当 normalized=False 且需要从像素反归一化时使用，或 normalized=True 但距离要以像素为单位时。
                                     默认为 settings.IMAGE_WIDTH。
        image_height (int, optional): 图像高度，类似 image_width。默认为 settings.IMAGE_HEIGHT。
        
    Returns:
        torch.Tensor: 每个样本所有角点的平均欧几里得距离，形状为 [B]
    """
    batch_size = pred_corners.shape[0]
    num_coords = pred_corners.shape[1]
    if num_coords % 2 != 0:
        raise ValueError("角点坐标数量必须是偶数。")
    num_points = num_coords // 2

    # 确保 image_width 和 image_height 有值
    eff_image_width = image_width if image_width is not None else settings.IMAGE_WIDTH
    eff_image_height = image_height if image_height is not None else settings.IMAGE_HEIGHT

    # 重塑为 [B, num_points, 2] 以便计算点间距离
    pred_reshaped = pred_corners.reshape(batch_size, num_points, 2)
    true_reshaped = true_corners.reshape(batch_size, num_points, 2)

    # 计算每个角点的欧几里得距离的平方 (x-x')^2 + (y-y')^2
    # 得到 [B, num_points] 的张量
    squared_distances = torch.sum((pred_reshaped - true_reshaped)**2, dim=2)

    # 正确的做法应该是：
    # 1. 如果输入是归一化的，先转换为像素坐标。
    # 2. 然后计算像素空间中的欧式距离。
    if normalized:
        pred_pixel = pred_reshaped.clone()
        pred_pixel[..., 0] *= eff_image_width
        pred_pixel[..., 1] *= eff_image_height
        true_pixel = true_reshaped.clone()
        true_pixel[..., 0] *= eff_image_width
        true_pixel[..., 1] *= eff_image_height
        squared_distances = torch.sum((pred_pixel - true_pixel)**2, dim=2)

    point_distances = torch.sqrt(
        squared_distances)  # 现在 point_distances 是像素单位的距离，形状 [B, num_points]

    # 计算每个样本所有角点的平均距离
    mean_distances_per_sample = point_distances.mean(dim=1)  # 形状 [B]

    return mean_distances_per_sample


def percentage_correct_keypoints(pred_corners,
                                 true_corners,
                                 threshold_normalized=0.005,
                                 image_width=None,
                                 image_height=None):
    """
    计算正确预测的关键点百分比 (PCK)。
    如果预测角点与真实角点之间的距离小于某个阈值（该阈值通常相对于图像尺寸定义），则认为该角点预测正确。
    输入角点坐标假定是归一化的 [0,1]。
    
    Args:
        pred_corners (torch.Tensor): 预测角点坐标 (归一化)，形状为 [B, N*2]
        true_corners (torch.Tensor): 真实角点坐标 (归一化)，形状为 [B, N*2]
        threshold_normalized (float): 归一化距离阈值，通常是图像对角线长度的一个比例。
                                      但更常见的做法是定义一个像素阈值，或者一个相对于特征尺度（如包围框大小）的阈值。
                                      这里我们假设 threshold_normalized 是一个相对于对角线的比例，或者可以直接传入像素阈值。
                                      为了与原始 PCK 定义一致，我们使用基于对角线的阈值。
        image_width (int, optional): 图像宽度。默认为 settings.IMAGE_WIDTH。
        image_height (int, optional): 图像高度。默认为 settings.IMAGE_HEIGHT。
        
    Returns:
        float: PCK 值 (0.0-1.0)，批次中所有关键点的平均正确率。
    """
    batch_size = pred_corners.shape[0]
    num_coords = pred_corners.shape[1]
    if num_coords % 2 != 0:
        raise ValueError("角点坐标数量必须是偶数。")
    num_points = num_coords // 2

    eff_image_width = image_width if image_width is not None else settings.IMAGE_WIDTH
    eff_image_height = image_height if image_height is not None else settings.IMAGE_HEIGHT

    # 将归一化坐标转换为像素坐标
    pred_pixel = pred_corners.reshape(batch_size, num_points, 2).clone()
    pred_pixel[..., 0] *= eff_image_width
    pred_pixel[..., 1] *= eff_image_height

    true_pixel = true_corners.reshape(batch_size, num_points, 2).clone()
    true_pixel[..., 0] *= eff_image_width
    true_pixel[..., 1] *= eff_image_height

    # 计算像素空间中的欧几里得距离
    point_distances_pixel = torch.sqrt(
        torch.sum((pred_pixel - true_pixel)**2, dim=2))  # 形状 [B, num_points]

    # 计算像素阈值 (基于图像对角线长度)
    diag_pixel = torch.sqrt(
        torch.tensor([eff_image_width**2 + eff_image_height**2],
                     dtype=torch.float32,
                     device=pred_corners.device))
    threshold_pixel_val = threshold_normalized * diag_pixel

    # 计算正确预测的关键点 (距离小于像素阈值)
    correct_keypoints = (point_distances_pixel < threshold_pixel_val).float()

    # 计算批次总PCK
    pck = correct_keypoints.sum() / (batch_size * num_points)

    return pck.item()


def intersection_over_union(pred_masks, true_masks, threshold=0.5):
    """
    计算预测掩码和真实掩码之间的IoU（交并比）。
    
    Args:
        pred_masks (torch.Tensor): 预测掩码，形状为 [B, 1, H, W]，值为sigmoid输出 (0-1)
        true_masks (torch.Tensor): 真实掩码，形状为 [B, 1, H, W]，值为0或1
        threshold (float): 将预测掩码二值化的阈值
        
    Returns:
        torch.Tensor: 每个样本的IoU值，形状为 [B]
    """
    # 二值化预测掩码
    pred_binary = (pred_masks > threshold).float()

    # 计算交集和并集
    intersection = (pred_binary * true_masks).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + true_masks.sum(
        dim=(1, 2, 3)) - intersection

    # 处理特殊情况：并集为零
    iou = torch.zeros_like(intersection)
    valid_mask = union > 0
    iou[valid_mask] = intersection[valid_mask] / union[valid_mask]

    return iou


def dice_coefficient(pred_masks, true_masks, threshold=0.5):
    """
    计算预测掩码和真实掩码之间的Dice系数。
    
    Args:
        pred_masks (torch.Tensor): 预测掩码，形状为 [B, 1, H, W]，值为sigmoid输出 (0-1)
        true_masks (torch.Tensor): 真实掩码，形状为 [B, 1, H, W]，值为0或1
        threshold (float): 将预测掩码二值化的阈值
        
    Returns:
        torch.Tensor: 每个样本的Dice系数，形状为 [B]
    """
    # 二值化预测掩码
    pred_binary = (pred_masks > threshold).float()

    # 计算交集
    intersection = (pred_binary * true_masks).sum(dim=(1, 2, 3))

    # 计算和
    sum_pred_true = pred_binary.sum(dim=(1, 2, 3)) + true_masks.sum(dim=(1, 2,
                                                                         3))

    # 处理特殊情况：和为零
    dice = torch.zeros_like(intersection)
    valid_mask = sum_pred_true > 0
    dice[valid_mask] = 2.0 * intersection[valid_mask] / sum_pred_true[
        valid_mask]

    return dice


def pixel_accuracy(pred_masks, true_masks, threshold=0.5):
    """
    计算预测掩码和真实掩码之间的像素准确率。
    
    Args:
        pred_masks (torch.Tensor): 预测掩码，形状为 [B, 1, H, W]，值为sigmoid输出 (0-1)
        true_masks (torch.Tensor): 真实掩码，形状为 [B, 1, H, W]，值为0或1
        threshold (float): 将预测掩码二值化的阈值
        
    Returns:
        torch.Tensor: 每个样本的像素准确率，形状为 [B]
    """
    # 二值化预测掩码
    pred_binary = (pred_masks > threshold).float()

    # 计算正确分类的像素数量
    correct = (pred_binary == true_masks).float().sum(dim=(1, 2, 3))

    # 计算总像素数
    total = true_masks.shape[1] * true_masks.shape[2] * true_masks.shape[3]

    # 计算准确率
    accuracy = correct / total

    return accuracy


def evaluate_batch(pred_mask_logits,
                   true_sorted_corners_normalized,
                   true_masks,
                   original_image_width=None,
                   original_image_height=None,
                   pck_threshold_normalized=0.005,
                   postprocess_threshold_mask=0.5,
                   debug=False):
    """
    评估一个批次的预测结果，使用掩码派生角点。支持多目标识别评估。
    
    Args:
        pred_mask_logits (torch.Tensor): 预测的掩码 logits [B, 1, H_mask, W_mask]
        true_sorted_corners_normalized (torch.Tensor): 真实的、排序后的、归一化的角点坐标 [B, NUM_CORNERS]
        true_masks (torch.Tensor): 真实的掩码 [B, 1, H_mask, W_mask]
        original_image_width (int, optional): 原始图像宽度，默认使用settings.IMAGE_WIDTH
        original_image_height (int, optional): 原始图像高度，默认使用settings.IMAGE_HEIGHT
        pck_threshold_normalized (float): PCK计算中使用的归一化阈值
        postprocess_threshold_mask (float): 用于二值化掩码以提取角点的阈值
        debug (bool): 是否打印调试信息
        
    Returns:
        dict: 包含各项评估指标的字典
    """
    img_width = original_image_width if original_image_width is not None else settings.IMAGE_WIDTH
    img_height = original_image_height if original_image_height is not None else settings.IMAGE_HEIGHT
    batch_size = pred_mask_logits.shape[0]

    # 1. 从预测掩码 logits 中派生角点
    pred_mask_probs = torch.sigmoid(pred_mask_logits)
    pred_corners_tensor, num_objects_tensor = mask_to_longest_edge_corners(
        pred_mask_probs,
        original_image_width=img_width,
        original_image_height=img_height,
        threshold=postprocess_threshold_mask,
        max_objects=5)  # 最多检测5个物体

    # 2. 评估角点预测
    # 在多目标情况下，我们需要找到与真实标签最匹配的预测物体
    # 这里使用简单的策略：找距离最近的那个物体（如果数据集中都是单目标，就是第一个物体）

    # 初始化度量结果
    mean_pixel_distances = []
    pck_values = []
    best_matching_object_indices = []

    for b in range(batch_size):
        # 获取该样本的物体数量
        num_objects = num_objects_tensor[b].item()

        if num_objects == 0:
            # 如果没有检测到物体，计入最大距离（最差情况）
            mean_pixel_distances.append(
                torch.tensor(float('inf'), device=pred_mask_logits.device))
            pck_values.append(0.0)
            best_matching_object_indices.append(-1)  # 表示没有匹配
            continue

        # 获取真实角点（假设数据集每个样本只有一个目标物体）
        true_corners = true_sorted_corners_normalized[b].unsqueeze(0)  # [1, 4]

        # 为每个预测的物体计算距离
        best_distance = float('inf')
        best_pck = 0.0
        best_obj_idx = 0

        for obj_idx in range(num_objects):
            # 获取当前物体的角点
            pred_corners_pixel = pred_corners_tensor[b, obj_idx].unsqueeze(
                0)  # [1, 4]

            # 跳过包含NaN的角点
            if torch.isnan(pred_corners_pixel).any():
                continue

            # 将像素坐标转换为归一化坐标用于评估
            pred_corners_normalized = pred_corners_pixel.clone().float()
            pred_corners_normalized = pred_corners_normalized.view(-1, 2)
            pred_corners_normalized[:, 0] /= img_width
            pred_corners_normalized[:, 1] /= img_height
            pred_corners_normalized = torch.clamp(pred_corners_normalized, 0,
                                                  1)
            pred_corners_normalized = pred_corners_normalized.view(
                1, -1)  # [1, 4]

            # 计算角点距离
            curr_distance = corner_distance(pred_corners_normalized,
                                            true_corners,
                                            normalized=True,
                                            image_width=img_width,
                                            image_height=img_height).item()

            # 计算PCK
            curr_pck = percentage_correct_keypoints(
                pred_corners_normalized,
                true_corners,
                threshold_normalized=pck_threshold_normalized,
                image_width=img_width,
                image_height=img_height)

            # 如果这个物体的角点距离更小，更新最佳匹配
            if curr_distance < best_distance:
                best_distance = curr_distance
                best_pck = curr_pck
                best_obj_idx = obj_idx

        # 添加该样本的最佳匹配结果
        mean_pixel_distances.append(
            torch.tensor(best_distance, device=pred_mask_logits.device))
        pck_values.append(best_pck)
        best_matching_object_indices.append(best_obj_idx)

    # 转换为张量并计算平均值
    mean_pixel_distance = torch.stack(mean_pixel_distances).mean().item()
    pck = sum(pck_values) / len(pck_values)

    # 3. 评估掩码预测 (这部分保持不变)
    iou_val = intersection_over_union(pred_mask_probs,
                                      true_masks).mean().item()
    dice_val = dice_coefficient(pred_mask_probs, true_masks).mean().item()
    accuracy_val = pixel_accuracy(pred_mask_probs, true_masks).mean().item()

    # 返回评估结果，包括检测到的物体数量信息
    return {
        'corner_distance': mean_pixel_distance,
        'pck': pck,
        'iou': iou_val,
        'dice': dice_val,
        'accuracy': accuracy_val,
        'pred_corners_tensor': pred_corners_tensor,
        'num_objects_tensor': num_objects_tensor,
        'best_matching_indices': best_matching_object_indices
    }


if __name__ == '__main__':
    print("\n====== 指标计算测试 (使用掩码派生角点) ======")
    batch_size = 3
    num_coords = settings.NUM_CORNERS
    num_points = num_coords // 2

    img_h, img_w = settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH

    print(
        f"\n创建测试数据: batch_size={batch_size}, num_points={num_points} (from NUM_CORNERS={num_coords})"
    )
    print(f"图像尺寸: {img_w}x{img_h}")

    true_corners_normalized = torch.zeros(batch_size, num_coords)

    true_corners_normalized[:, 0] = 0.3
    true_corners_normalized[:, 1] = 0.4
    true_corners_normalized[:, 2] = 0.7
    true_corners_normalized[:, 3] = 0.6

    pred_mask_logits = torch.rand(batch_size, 1, img_h, img_w) * 5 - 2
    true_masks = (torch.rand(batch_size, 1, img_h, img_w) > 0.5).float()

    print("\n警告: mask_to_longest_edge_corners 未实现，将使用模拟的预测角点进行测试。")
    mock_pred_corners_pixel = torch.zeros(batch_size, num_coords)
    mock_pred_corners_pixel[0, 0] = 0.31 * img_w
    mock_pred_corners_pixel[0, 1] = 0.41 * img_h
    mock_pred_corners_pixel[0, 2] = 0.71 * img_w
    mock_pred_corners_pixel[0, 3] = 0.61 * img_h

    mock_pred_corners_pixel[1, 0] = 0.25 * img_w
    mock_pred_corners_pixel[1, 1] = 0.35 * img_h
    mock_pred_corners_pixel[1, 2] = 0.75 * img_w
    mock_pred_corners_pixel[1, 3] = 0.65 * img_h

    mock_pred_corners_pixel[2, 0] = 0.1 * img_w
    mock_pred_corners_pixel[2, 1] = 0.2 * img_h
    mock_pred_corners_pixel[2, 2] = 0.9 * img_w
    mock_pred_corners_pixel[2, 3] = 0.8 * img_h

    original_mask_to_corners = mask_to_longest_edge_corners

    def mock_mask_to_corners_func(pred_mask_probs, original_image_width,
                                  original_image_height, threshold):
        return mock_pred_corners_pixel.to(pred_mask_probs.device)

    import newmodel.utils.metrics
    newmodel.utils.metrics.mask_to_longest_edge_corners = mock_mask_to_corners_func

    print("\n执行评估 (使用模拟的掩码派生角点)...")
    metrics_output = evaluate_batch(pred_mask_logits,
                                    true_corners_normalized,
                                    true_masks,
                                    original_image_width=img_w,
                                    original_image_height=img_h,
                                    pck_threshold_normalized=0.05,
                                    postprocess_threshold_mask=0.5,
                                    debug=True)

    newmodel.utils.metrics.mask_to_longest_edge_corners = original_mask_to_corners

    print("\n评估结果:")
    print(f"角点平均距离: {metrics_output['corner_distance']:.2f} 像素")
    print(f"PCK@0.05: {metrics_output['pck']:.4f}")
    print(f"掩码IoU: {metrics_output['iou']:.4f}")
    print(f"掩码Dice系数: {metrics_output['dice']:.4f}")
    print(f"掩码像素准确率: {metrics_output['accuracy']:.4f}")

    print("\n预测的角点坐标 (来自模拟的掩码派生):")
    for b in range(min(batch_size, 3)):
        print(f"样本 {b}:")
        print(
            f"  模拟派生像素坐标: {metrics_output['pred_corners_tensor'][b].tolist()}")
        print(
            f"  模拟派生归一化坐标: {metrics_output['pred_corners_tensor'][b].tolist()}"
        )
        print(f"  真实归一化坐标: {true_corners_normalized[b].tolist()}")

    print("\n====== 指标计算测试完成 ======")
