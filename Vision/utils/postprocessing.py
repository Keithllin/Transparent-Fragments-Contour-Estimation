import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2  # Added OpenCV import

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (newmodel/ 感知为项目 utils/ 的上一级)
project_root = os.path.dirname(current_dir)
# 如果项目根目录不在sys.path中，则添加它
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Attempt to import settings, fallback to a mock for standalone testing or incomplete setup
try:
    from config import settings
except ModuleNotFoundError:
    print("无法导入 config.settings。将使用 MockSettings 进行 postprocessing.py 测试。")

    class MockSettings:
        IMAGE_WIDTH = 640
        IMAGE_HEIGHT = 480
        NUM_CORNERS = 4  # Assuming 2 points, (x1,y1,x2,y2)
        # HEATMAP_SIGMA and NUM_KEYPOINTS are no longer needed here

    settings = MockSettings()

# dsntnn related imports are removed as heatmaps_to_corners is removed
# try:
#     import dsntnn
# except ImportError:
# ... (dsntnn mock removed)

# Removed: visualize_heatmap function (was specific to heatmaps)

# Removed: find_max_location function (was specific to heatmaps)

# Removed: preprocess_heatmap function (was specific to heatmaps)

# Commented out or removed: heatmaps_to_corners function
# def heatmaps_to_corners(pred_heatmaps, ...):
#     ...


def mask_to_longest_edge_corners(
        pred_mask_probs_batch,
        original_image_width,
        original_image_height,
        threshold=0.5,
        max_objects=5,  # 最多处理的物体数量
        min_contour_area=100):  # 最小轮廓面积（像素）
    """
    从预测的分割掩码中为每个检测到的物体提取最远的两个顶点。
    
    Args:
        pred_mask_probs_batch (torch.Tensor): 预测的掩码概率 (sigmoid后), 形状 [B, 1, H, W]。
        original_image_width (int): 原始图像宽度。
        original_image_height (int): 原始图像高度。
        threshold (float): 用于二值化掩码的阈值。
        max_objects (int): 每张图像最多处理的物体数量。
        min_contour_area (int): 要考虑的最小轮廓面积。
        
    Returns:
        tuple: (corners_tensor, num_objects_tensor)
            corners_tensor: 形状为 [B, max_objects, 4] 的角点像素坐标 (x1,y1,x2,y2)
            num_objects_tensor: 形状为 [B] 的每个样本中检测到的物体数量
    """
    batch_size = pred_mask_probs_batch.shape[0]
    device = pred_mask_probs_batch.device

    # 初始化输出 - 现在是3D的 [B, max_objects, 4]
    output_corners = np.full((batch_size, max_objects, 4),
                             np.nan,
                             dtype=np.float32)
    num_objects_detected = np.zeros((batch_size, ), dtype=np.int32)

    # 处理每个批次样本
    pred_mask_probs_np_batch = pred_mask_probs_batch.squeeze(1).cpu().numpy()

    for i in range(batch_size):
        mask_probs_np = pred_mask_probs_np_batch[i]
        binary_mask = (mask_probs_np > threshold).astype(np.uint8)

        # 找到所有轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 过滤掉太小的轮廓并按面积排序（从大到小）
        valid_contours = [
            c for c in contours if cv2.contourArea(c) >= min_contour_area
        ]
        valid_contours.sort(key=cv2.contourArea, reverse=True)

        # 限制处理的轮廓数量
        valid_contours = valid_contours[:max_objects]
        num_objects = len(valid_contours)
        num_objects_detected[i] = num_objects

        # 处理每个有效轮廓
        for j, contour in enumerate(valid_contours):
            if j >= max_objects:
                break

            # 修改：直接使用原始轮廓点，不再进行多边形逼近
            # # 使用 Ramer-Douglas-Peucker 算法简化轮廓
            # epsilon = 0.01 * cv2.arcLength(contour, True)
            # approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            #
            # if approx_polygon.shape[0] < 2:
            #     continue  # 跳过没有足够顶点的轮廓
            #
            # # 找到最远的顶点对
            # polygon_points = approx_polygon.squeeze(1)

            polygon_points = contour.squeeze(1) # 直接使用原始轮廓点

            if len(polygon_points) < 2: # 确保轮廓至少有两个点
                continue

            max_dist_sq = -1
            farthest_pair = None

            for m in range(len(polygon_points)):
                for n in range(m + 1, len(polygon_points)):
                    p1 = polygon_points[m]
                    p2 = polygon_points[n]

                    dx = p1[0] - p2[0]
                    dy = p1[1] - p2[1]
                    current_dist_sq = dx * dx + dy * dy

                    if current_dist_sq > max_dist_sq:
                        max_dist_sq = current_dist_sq
                        # 确保顶点对的顺序一致
                        if p1[0] < p2[0] or (p1[0] == p2[0] and p1[1] < p2[1]):
                            farthest_pair = (p1, p2)
                        else:
                            farthest_pair = (p2, p1)

            if farthest_pair:
                # 保存这个物体的角点
                output_corners[i, j, :] = [
                    farthest_pair[0][0], farthest_pair[0][1],
                    farthest_pair[1][0], farthest_pair[1][1]
                ]

    # 转换为torch张量
    corners_tensor = torch.tensor(output_corners,
                                  dtype=torch.float32,
                                  device=device)
    num_objects_tensor = torch.tensor(num_objects_detected,
                                      dtype=torch.int32,
                                      device=device)

    # 为了兼容性，如果max_objects=1，则返回形状为[B,4]的张量
    if max_objects == 1:
        return corners_tensor.view(batch_size, 4)

    return corners_tensor, num_objects_tensor


# 为了兼容性，添加一个包装函数，用于获取单物体角点
def get_single_object_corners(pred_mask_probs_batch,
                              original_image_width,
                              original_image_height,
                              threshold=0.5):
    """
    从预测的分割掩码中提取最大轮廓的最远顶点对（兼容旧接口）。
    
    Args:
        pred_mask_probs_batch (torch.Tensor): 预测的掩码概率 (sigmoid后), 形状 [B, 1, H, W]。
        original_image_width (int): 原始图像宽度。
        original_image_height (int): 原始图像高度。
        threshold (float): 用于二值化掩码的阈值。
        
    Returns:
        torch.Tensor: 形状为 [B, 4] 的角点像素坐标 (x1,y1,x2,y2)
    """
    # 调用新的多物体函数，但只取最大的一个物体
    corners_tensor, _ = mask_to_longest_edge_corners(
        pred_mask_probs_batch,
                        original_image_width,
                        original_image_height,
        threshold=threshold,
        max_objects=1  # 只处理最大的一个物体
    )

    # 由于max_objects=1，corners_tensor形状为[B,1,4]，需要移除中间维度
    return corners_tensor.squeeze(1)


if __name__ == '__main__':
    print("====== Postprocessing: mask_to_longest_edge_corners 测试 ======")

    # 尝试导入数据处理和数据集相关的模块
    try:
        from data.dataset import CornerPointDataset
        from data.transforms import Preprocessing  # For loading original image correctly
        from utils.metrics import corner_distance  # For evaluation
        has_deps = True
        print("成功导入测试依赖 (CornerPointDataset, Preprocessing, corner_distance)。")
    except ImportError as e:
        has_deps = False
        print(f"未能导入测试依赖: {e}。测试将受限。")

    if not has_deps:
        print("由于缺少依赖，无法执行完整的 mask_to_longest_edge_corners 测试。")
        sys.exit(1)

    # 初始化参数
    batch_size_to_test = 2
    img_h_settings = settings.IMAGE_HEIGHT
    img_w_settings = settings.IMAGE_WIDTH
    max_objects = 5  # 每个样本最多检测的物体数

    print(f"\n测试参数: Batch size={batch_size_to_test}, 最大物体数={max_objects}")
    print(
        f"图像尺寸 (from settings for Preprocessing): H={img_h_settings}, W={img_w_settings}"
    )

    # 1. 使用真实数据集加载样本，并使用其真实掩码进行测试
    print("\n使用真实数据集的真实掩码进行测试多物体检测...")
    # Load dataset without transforms to get original image and mask
    raw_dataset = CornerPointDataset(mode='train', transform=None)
    # Preprocessor for consistent image loading if needed for visualization later
    # but mask_to_longest_edge_corners takes mask probabilities, so original mask is fine.

    if len(raw_dataset) < batch_size_to_test:
        print(f"数据集样本数量不足 ({len(raw_dataset)} < {batch_size_to_test})，无法完成测试。")
        sys.exit(1)

    all_derived_corners_pixel = []
    all_num_objects = []
    all_gt_corners_pixel = []  # For metrics
    all_gt_corners_normalized = []  # For metrics, original normalized format

    # 为不同物体准备不同的颜色
    colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 不同物体的颜色

    fig, axes = plt.subplots(batch_size_to_test,
                             3,
                             figsize=(15, 5 * batch_size_to_test))
    if batch_size_to_test == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size_to_test):
        sample = raw_dataset[i]
        original_pil_image = sample['image']  # PIL Image
        gt_anno_mask_np = sample['mask']  # Numpy array (H, W), 0 or 1
        gt_corners_orig_pixel_np = sample[
            'corner']  # Numpy array (2,2) [[x1,y1],[x2,y2]] (original pixel coords)

        original_w, original_h = original_pil_image.size

        # 将真实掩码转换为 mask_to_longest_edge_corners 所需的格式 [1, 1, H, W] tensor
        # 这里假设真实掩码是完美的概率（0或1）
        gt_anno_mask_tensor = torch.from_numpy(
            gt_anno_mask_np).float().unsqueeze(0).unsqueeze(0)

        # 调用函数进行测试 - 现在返回两个值：角点和物体数量
        derived_corners_pixel_tensor, num_objects_tensor = mask_to_longest_edge_corners(
            gt_anno_mask_tensor,
            original_image_width=original_w,
            original_image_height=original_h,
            threshold=0.5,  # Threshold for binarization
            max_objects=max_objects,  # 最多处理的物体数量
            min_contour_area=100  # 最小轮廓面积（像素）
        )
        derived_corners_pixel_np = derived_corners_pixel_tensor.squeeze(
            0).cpu().numpy()  # Get [max_objects, 4]
        num_objects = num_objects_tensor[0].item()  # 获取第一个样本中检测到的物体数量

        all_derived_corners_pixel.append(
            derived_corners_pixel_tensor
        )  # Store tensor for batch metric calculation
        all_num_objects.append(num_objects_tensor)  # Store number of objects

        # 准备真实角点用于比较 (扁平化并排序，如果需要)
        # 原始真实角点已经是 [[x1,y1],[x2,y2]]，需要排序和扁平化
        sort_indices_gt = np.lexsort(
            (gt_corners_orig_pixel_np[:, 1], gt_corners_orig_pixel_np[:, 0]))
        sorted_gt_corners_pixel_np = gt_corners_orig_pixel_np[
            sort_indices_gt].flatten()
        all_gt_corners_pixel.append(
            torch.from_numpy(sorted_gt_corners_pixel_np).float().unsqueeze(0))

        # Also prepare normalized GT corners for metric function if it expects normalized
        # The corner_distance function can handle both, but PCK typically uses normalized inputs
        # The `corner` field from Preprocessing is sorted & normalized & flattened
        # Here we construct a similar one from raw data for testing metrics
        sorted_gt_normalized = gt_corners_orig_pixel_np[sort_indices_gt].copy(
        ).astype(np.float32)
        sorted_gt_normalized[:, 0] /= original_w
        sorted_gt_normalized[:, 1] /= original_h
        all_gt_corners_normalized.append(
            torch.from_numpy(
                sorted_gt_normalized.flatten()).float().unsqueeze(0))

        print(f"\n样本 {i}: Original Dims ({original_w}x{original_h})")
        print(f"  检测到的物体数量: {num_objects}")
        print(f"  真实角点 (原始像素,排序后扁平): {sorted_gt_corners_pixel_np}")

        # 打印每个检测到的物体的角点
        for obj_idx in range(num_objects):
            obj_corners = derived_corners_pixel_np[obj_idx]
            if not np.isnan(obj_corners).any():  # 跳过无效角点
                print(f"  物体 {obj_idx+1} 从真实掩码派生的角点 (像素): {obj_corners}")

        # 可视化
        # 原始图像
        axes[i, 0].imshow(original_pil_image)
        axes[i, 0].set_title(f'Sample {i}: Original Image')
        axes[i, 0].scatter(sorted_gt_corners_pixel_np[0::2],
                           sorted_gt_corners_pixel_np[1::2],
                           c='lime',
                           marker='o',
                           s=50,
                           label='GT Corners')
        axes[i, 0].legend()

        # 真实掩码 + 轮廓
        axes[i, 1].imshow(gt_anno_mask_np, cmap='gray')
        axes[i, 1].set_title(f'Sample {i}: GT Mask & Contour')
        # 重新找到轮廓以便绘制
        contours_viz, _ = cv2.findContours(gt_anno_mask_np.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        if contours_viz:
            # 绘制所有轮廓
            for contour_idx, contour in enumerate(contours_viz):
                color_idx = contour_idx % len(colors)
                axes[i, 1].plot(contour[:, 0, 0],
                                contour[:, 0, 1],
                                f'{colors[color_idx]}-',
                                linewidth=1,
                                label=f'Contour {contour_idx+1}' if contour_idx
                                == 0 else f'Contour {contour_idx+1}')

            if len(contours_viz) > 0:
                axes[i, 1].legend()

        # 原始图像 + 派生角点
        axes[i, 2].imshow(original_pil_image)
        axes[i, 2].set_title(f'Sample {i}: {num_objects} Detected Objects')

        # 绘制每个检测到的物体
        legend_added = False
        for obj_idx in range(num_objects):
            obj_corners = derived_corners_pixel_np[obj_idx]
            if not np.isnan(obj_corners).any():  # 跳过无效角点
                # 选择颜色
                color_idx = obj_idx % len(colors)
                color = colors[color_idx]

                # 绘制角点
                axes[i, 2].scatter([obj_corners[0], obj_corners[2]],
                                   [obj_corners[1], obj_corners[3]],
                                   c=color,
                                   marker='x',
                                   s=50,
                                   label=f'Obj {obj_idx+1}'
                                   if not legend_added else f'Obj {obj_idx+1}')

                # 绘制连接线
                axes[i, 2].plot([obj_corners[0], obj_corners[2]],
                                [obj_corners[1], obj_corners[3]],
                                f'{color}--',
                                linewidth=2)

                legend_added = True

        if legend_added:
            axes[i, 2].legend()

    plt.tight_layout()
    save_path_fig = "postprocessing_mask_to_corners_test.png"
    plt.savefig(save_path_fig)
    print(f"\n可视化结果已保存至: {save_path_fig}")

    # 计算指标
    if all_derived_corners_pixel and all_gt_corners_normalized:
        print("\n评估第一个物体与真实标签的匹配度...")
        # 简化的评估，只评估每个样本中的第一个物体
        batch_size = len(all_derived_corners_pixel)
        first_obj_corners = []

        for b in range(batch_size):
            # 获取该样本的物体数量
            num_objects = all_num_objects[b][0].item()

            if num_objects > 0:
                # 获取第一个物体的角点
                first_obj_corner = all_derived_corners_pixel[b][
                    0, 0].unsqueeze(0)  # [1, 4]
                first_obj_corners.append(first_obj_corner)
            else:
                # 如果没有物体，添加NaN
                first_obj_corners.append(
                    torch.full((1, 4),
                               float('nan'),
                               device=all_derived_corners_pixel[b].device))

        if first_obj_corners:
            # 连接所有第一个物体的角点
            batch_first_obj_corners = torch.cat(first_obj_corners,
                                                dim=0)  # [B, 4]
            batch_gt_corners_normalized = torch.cat(all_gt_corners_normalized,
                                                    dim=0)  # [B, 4]

            # 计算归一化坐标
            valid_mask = ~torch.isnan(batch_first_obj_corners).any(dim=1)
            if valid_mask.any():
                first_sample_w, first_sample_h = raw_dataset[0]['image'].size

                # 转换为归一化坐标
                batch_first_obj_normalized = batch_first_obj_corners[
                    valid_mask].clone()
                num_pts = batch_first_obj_normalized.shape[1] // 2
                batch_first_obj_normalized = batch_first_obj_normalized.view(
                    -1, num_pts, 2)
                batch_first_obj_normalized[..., 0] /= first_sample_w
                batch_first_obj_normalized[..., 1] /= first_sample_h
                batch_first_obj_normalized = batch_first_obj_normalized.view(
                    -1, settings.NUM_CORNERS)

                gt_corners_for_metric = batch_gt_corners_normalized[valid_mask]

                # 计算距离
                if batch_first_obj_normalized.shape == gt_corners_for_metric.shape:
                    avg_dist = corner_distance(
                        pred_corners=batch_first_obj_normalized,
                        true_corners=gt_corners_for_metric,
                        normalized=True,
                        image_width=first_sample_w,
                        image_height=first_sample_h).mean().item()
                    print(f"\n第一个物体角点的平均距离: {avg_dist:.2f} 像素")
                else:
                    print(
                        f"\n形状不匹配，无法计算角点距离。 First obj: {batch_first_obj_normalized.shape}, GT: {gt_corners_for_metric.shape}"
                    )
            else:
                print("\n没有有效的第一个物体角点。")

    print("\n====== Postprocessing 测试完成 ======")
