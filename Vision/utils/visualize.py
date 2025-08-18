import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import make_grid
import os
import sys

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 如果项目根目录不在sys.path中，则添加它
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import settings
from data.dataset import CornerPointDataset
from data.transforms import Preprocessing
from utils.postprocessing import mask_to_longest_edge_corners


def denormalize_image(tensor,
                      mean=settings.NORMALIZE_MEAN,
                      std=settings.NORMALIZE_STD):
    """
    将归一化的图像张量转换回原始RGB值范围，用于可视化。
    
    Args:
        tensor (torch.Tensor): 形状为 [C, H, W] 的归一化图像张量
        mean (list): 归一化时使用的均值
        std (list): 归一化时使用的标准差
        
    Returns:
        torch.Tensor: 形状为 [C, H, W] 的反归一化图像张量，值范围为 [0, 1]
    """
    tensor = tensor.clone()  # 避免修改原张量

    # 反归一化
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    # 裁剪到 [0, 1] 范围
    return torch.clamp(tensor, 0, 1)


def visualize_dataset_samples(dataset,
                              num_samples=3,
                              figsize=(15, 5),
                              save_path=None):
    """
    可视化数据集的原始样本。
    
    Args:
        dataset (CornerPointDataset): 数据集实例
        num_samples (int): 要可视化的样本数量
        figsize (tuple): 图像尺寸
        save_path (str, optional): 保存图像的路径，如果为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    num_samples = min(num_samples, len(dataset))
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # 获取样本
        sample_idx = np.random.randint(0, len(dataset))
        # 使用 get_raw_sample 获取未经转换的原始数据
        if hasattr(dataset, 'get_raw_sample'):
            sample = dataset.get_raw_sample(sample_idx)
        else:
            sample = dataset[sample_idx]

        # 获取原始图像、掩码，角点是可选的
        image = sample['image']
        corners = sample.get('corner', None)  # 角点可能不存在
        mask = sample['mask']

        # 如果是PyTorch张量，转换为NumPy数组
        if isinstance(image, torch.Tensor):
            image = denormalize_image(image).permute(1, 2, 0).cpu().numpy()
        else:  # 如果是PIL图像，转换为NumPy数组
            image = np.array(image)

        if isinstance(corners, torch.Tensor):
            corners = corners.cpu().numpy()

        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()

        # 显示原始图像
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Index: {sample_idx}: Original Image")
        axes[i, 0].axis('off')

        # 显示图像和角点（如果有的话）
        axes[i, 1].imshow(image)

        if corners is not None:
            axes[i, 1].set_title(f"Index: {sample_idx}: Image and Corners")

            # corners is sample['corner'], assumed (2,2) numpy array of pixel coordinates
            # Example: np.array([[x_diag1, y_diag1], [x_diag2, y_diag2]])
            if isinstance(corners, np.ndarray) and corners.shape == (2, 2):
                # Assuming raw corners are in pixel coordinates
                pixel_corners = corners
                x_coords_raw = pixel_corners[:,
                                             0]  # Extracts [x_diag1, x_diag2]
                y_coords_raw = pixel_corners[:,
                                             1]  # Extracts [y_diag1, y_diag2]

                axes[i, 1].scatter(x_coords_raw, y_coords_raw, c='red', s=40)
                axes[i, 1].axis('on')
                axes[i, 1].axis('image')
                axes[i, 1].set_xlim(0, image.shape[1])
                axes[i, 1].set_ylim(image.shape[0], 0)
            else:
                print(
                    f"Skipping corner plotting for raw sample {sample_idx} in visualize_dataset_samples due to unexpected format: {corners.shape if isinstance(corners, np.ndarray) else type(corners)}"
                )
        else:
            axes[i,
                 1].set_title(f"Index: {sample_idx}: Image Only (No Corners)")

        axes[i, 1].axis('off')  # Ensure axis is off after plotting

        # 显示掩码
        axes[i, 2].imshow(mask, cmap='gray')
        axes[i, 2].set_title(f"Index: {sample_idx}: Mask")
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    return fig


def visualize_transformed_samples(dataset,
                                  transform=None,
                                  num_samples=3,
                                  figsize=(18, 10),
                                  save_path=None):
    """
    可视化数据集样本经过变换前后的对比。
    移除了目标热图的显式可视化，因为其已从预处理中移除。
    
    Args:
        dataset (CornerPointDataset): 不包含变换的原始数据集
        transform (callable, optional): 要应用的变换函数
        num_samples (int): 要可视化的样本数量
        figsize (tuple): 图像尺寸
        save_path (str, optional): 保存图像的路径，如果为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    num_samples = min(num_samples, len(dataset))
    # Columns: Orig Img, Orig Corners, Transformed Img, Transformed Corners+Mask
    # No target_heatmap column anymore
    num_cols = 4
    fig, axes = plt.subplots(num_samples, num_cols, figsize=figsize)

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # 获取原始样本
        sample_idx = np.random.randint(0, len(dataset))
        # 使用 get_raw_sample 获取未经转换的原始数据
        if hasattr(dataset, 'get_raw_sample'):
            original_sample = dataset.get_raw_sample(sample_idx)
        else:
            original_sample = dataset[sample_idx]

        # 应用变换（如果提供）
        if transform:
            transformed_sample = transform(original_sample)
        else:
            transformed_sample = original_sample

        # 处理原始图像
        if isinstance(original_sample['image'], torch.Tensor):
            orig_img = original_sample['image'].permute(1, 2, 0).cpu().numpy()
        else:
            orig_img = np.array(original_sample['image']) / 255.0

        # 处理原始角点（可能不存在）
        orig_corners = original_sample.get('corner', None)
        if orig_corners is not None and isinstance(orig_corners, torch.Tensor):
            orig_corners = orig_corners.cpu().numpy()

        # 处理原始掩码 (not explicitly shown in its own column anymore, but used for context)
        # orig_mask = original_sample['mask']
        # if isinstance(orig_mask, torch.Tensor):
        #     orig_mask = orig_mask.squeeze().cpu().numpy()
        # else:
        #     orig_mask = np.array(orig_mask)

        # 处理变换后的图像
        if isinstance(transformed_sample['image'], torch.Tensor):
            trans_img = denormalize_image(transformed_sample['image']).permute(
                1, 2, 0).cpu().numpy()
        else:
            trans_img = np.array(transformed_sample['image']) / 255.0

        # 处理变换后的角点 (ground truth corners after transform)
        trans_corners_gt = transformed_sample.get('corner', None)
        if trans_corners_gt is not None and isinstance(trans_corners_gt,
                                                       torch.Tensor):
            trans_corners_gt = trans_corners_gt.cpu().numpy()

        # 处理变换后的掩码 (ground truth mask after transform)
        trans_mask_gt = transformed_sample['mask']
        if isinstance(trans_mask_gt, torch.Tensor):
            trans_mask_gt = trans_mask_gt.squeeze().cpu().numpy()
        else:
            trans_mask_gt = np.array(trans_mask_gt)

        # Column 0: Original Image
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Index: {sample_idx}: Original Image")
        axes[i, 0].axis('off')

        # Column 1: Original Corners on Original Image (if available)
        axes[i, 1].imshow(orig_img)
        if orig_corners is not None:
            axes[i, 1].set_title(f"Index: {sample_idx}: Original Corners")
            if isinstance(orig_corners,
                          np.ndarray) and orig_corners.shape == (2, 2):
                pixel_orig_corners = orig_corners
                x_coords_orig_raw = pixel_orig_corners[:, 0]
                y_coords_orig_raw = pixel_orig_corners[:, 1]
                axes[i, 1].scatter(x_coords_orig_raw,
                                   y_coords_orig_raw,
                                   c='red',
                                   s=40)
                axes[i, 1].axis('on')
                axes[i, 1].axis('image')
                axes[i, 1].set_xlim(0, orig_img.shape[1])
                axes[i, 1].set_ylim(orig_img.shape[0], 0)
            else:
                print(
                    f"Skipping original corner plotting for sample {sample_idx} in visualize_transformed_samples due to unexpected format."
                )
        else:
            axes[i, 1].set_title(
                f"Index: {sample_idx}: Original Image (No Corners)")
        axes[i, 1].axis('off')

        # Column 2: Transformed Image
        axes[i, 2].imshow(trans_img)
        axes[i, 2].set_title(f"Index: {sample_idx}: Transformed Image")
        axes[i, 2].axis('off')

        # Column 3: Transformed Ground Truth Corners and Mask on Transformed Image
        axes[i, 3].imshow(trans_img)
        h_trans, w_trans = trans_img.shape[:2]

        # 只有当角点数据存在时才绘制角点
        if trans_corners_gt is not None:
            # trans_corners_gt is a numpy array [x1_norm, y1_norm, x2_norm, y2_norm]
            scaled_transformed_gt_corners = trans_corners_gt.copy()
            scaled_transformed_gt_corners[0] *= w_trans
            scaled_transformed_gt_corners[1] *= h_trans
            scaled_transformed_gt_corners[2] *= w_trans
            scaled_transformed_gt_corners[3] *= h_trans
            x_coords_plot_gt = [
                scaled_transformed_gt_corners[0],
                scaled_transformed_gt_corners[2]
            ]
            y_coords_plot_gt = [
                scaled_transformed_gt_corners[1],
                scaled_transformed_gt_corners[3]
            ]
            axes[i, 3].scatter(x_coords_plot_gt,
                               y_coords_plot_gt,
                               c='lime',
                               marker='o',
                               s=40,
                               label='GT Corners (Transformed)')

        axes[i,
             3].imshow(trans_mask_gt, alpha=0.5,
                       cmap='Greens_r')  # Use a different colormap for GT mask

        title_suffix = " (Corners & Mask)" if trans_corners_gt is not None else " (Mask only)"
        axes[i,
             3].set_title(f"Index: {sample_idx}: Transformed GT{title_suffix}")
        axes[i, 3].axis('on')
        axes[i, 3].axis('image')
        axes[i, 3].set_xlim(0, trans_img.shape[1])
        axes[i, 3].set_ylim(trans_img.shape[0], 0)
        if trans_corners_gt is not None:
            axes[i, 3].legend()
        axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    return fig


def visualize_predictions(images,
                          true_corners=None,
                          true_masks=None,
                          pred_corners=None,
                          pred_masks=None,
                          num_samples=None,
                          figsize=(15, 10),
                          save_path=None):
    """
    可视化模型预测结果与真实标签的对比。
    如果未提供 pred_corners，则会尝试从 pred_masks (logits) 派生。
    支持可视化每个图像中的多个物体。
    
    Args:
        images (torch.Tensor): 批量图像张量，形状为 [B, C, H, W]
        true_corners (torch.Tensor, optional): 真实角点坐标 (归一化)，形状为 [B, 4]
        true_masks (torch.Tensor, optional): 真实掩码，形状为 [B, 1, H, W]
        pred_corners (tuple/torch.Tensor, optional): 
            如果是元组: (corners_tensor, num_objects_tensor)，其中
                corners_tensor 形状为 [B, max_objects, 4]
                num_objects_tensor 形状为 [B]
            如果是张量: 预测角点像素坐标，形状为 [B, 4]，表示单个物体
            如果为None且pred_masks提供，则会从中派生。
        pred_masks (torch.Tensor, optional): 预测掩码 logits，形状为 [B, 1, H, W]
        num_samples (int, optional): 要可视化的样本数量，默认为全部
        figsize (tuple): 图像尺寸
        save_path (str, optional): 保存图像的路径，如果为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    batch_size = images.shape[0]
    img_h, img_w = images.shape[2], images.shape[3]

    if num_samples is None:
        num_samples = batch_size
    else:
        num_samples = min(num_samples, batch_size)

    # 如果没有提供预测角点但提供了预测掩码，则从掩码派生角点
    if pred_corners is None and pred_masks is not None:
        pred_mask_probs = torch.sigmoid(
            pred_masks)  # Convert logits to probabilities
        # 多物体检测：返回 (corners_tensor, num_objects_tensor)
        pred_corners = mask_to_longest_edge_corners(
            pred_mask_probs,
            original_image_width=img_w,
            original_image_height=img_h,
            max_objects=5)  # 最多检测5个物体

    # 判断pred_corners是元组(多物体)还是张量(单物体)
    multi_object = isinstance(pred_corners, tuple)
    if multi_object:
        pred_corners_tensor, num_objects_tensor = pred_corners
    else:
        # 将单物体格式转换为多物体格式
        pred_corners_tensor = pred_corners.unsqueeze(1)  # [B, 4] -> [B, 1, 4]
        num_objects_tensor = torch.ones(batch_size,
                                        dtype=torch.int32,
                                        device=pred_corners.device)

    fig, axes = plt.subplots(num_samples, 2, figsize=figsize)

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # 为多个物体准备不同的颜色
    colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 不同物体的颜色

    for i in range(num_samples):
        img_idx_to_show = i  # Or some random index if you prefer for larger batches
        img = denormalize_image(images[img_idx_to_show]).permute(
            1, 2, 0).cpu().numpy()
        h_img_plot, w_img_plot = img.shape[:2]

        # --- 左侧图像：真实标签 ---
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {img_idx_to_show} - True Label")
        axes[i, 0].axis('off')

        if true_corners is not None:
            t_corners_norm = true_corners[img_idx_to_show].cpu().numpy()
            t_scaled_corners_pixel = t_corners_norm.copy()
            t_scaled_corners_pixel[0::2] *= w_img_plot  # x coords
            t_scaled_corners_pixel[1::2] *= h_img_plot  # y coords
            axes[i, 0].scatter(t_scaled_corners_pixel[0::2],
                               t_scaled_corners_pixel[1::2],
                               c='lime',
                               marker='o',
                               s=50,
                               label='True Corners')
            if len(t_scaled_corners_pixel) == 4:  # Draw line for 2 corners
                axes[i, 0].plot(
                    [t_scaled_corners_pixel[0], t_scaled_corners_pixel[2]],
                    [t_scaled_corners_pixel[1], t_scaled_corners_pixel[3]],
                    'g-',
                    linewidth=2)

        if true_masks is not None:
            t_mask = true_masks[img_idx_to_show].squeeze().cpu().numpy()
            axes[i, 0].imshow(t_mask, alpha=0.4, cmap='Greens_r')

        if true_corners is not None or true_masks is not None:
            axes[i, 0].legend()

        # --- 右侧图像：预测结果 ---
        axes[i, 1].imshow(img)
        axes[i, 1].set_title(f"Sample {img_idx_to_show} - Predicted")
        axes[i, 1].axis('off')

        if pred_corners_tensor is not None:
            # 处理该图像中预测的所有物体
            num_objects = num_objects_tensor[img_idx_to_show].item()

            for obj_idx in range(num_objects):
                # 获取当前物体的角点坐标
                p_corners_pixel = pred_corners_tensor[
                    img_idx_to_show, obj_idx].detach().cpu().numpy()

                # 跳过NaN值
                if np.isnan(p_corners_pixel).any():
                    continue

                # 选择物体的颜色
                obj_color = colors[obj_idx % len(colors)]

                # 绘制角点
                axes[i, 1].scatter([p_corners_pixel[0], p_corners_pixel[2]],
                                   [p_corners_pixel[1], p_corners_pixel[3]],
                                   c=obj_color,
                                   marker='x',
                                   s=50,
                                   label=f'Obj {obj_idx+1}'
                                   if obj_idx == 0 else f'Obj {obj_idx+1}')

                # 绘制连接线
                axes[i, 1].plot([p_corners_pixel[0], p_corners_pixel[2]],
                                [p_corners_pixel[1], p_corners_pixel[3]],
                                f'{obj_color}--',
                                linewidth=2)

        if pred_masks is not None:
            p_mask_probs = torch.sigmoid(
                pred_masks[img_idx_to_show].detach()).squeeze().cpu().numpy()
            axes[i, 1].imshow(p_mask_probs,
                              alpha=0.4,
                              cmap='Reds_r',
                              vmin=0,
                              vmax=1)

        # 如果有预测值，添加图例
        if pred_corners_tensor is not None or pred_masks is not None:
            axes[i, 1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig


def visualize_mask_predictions(images,
                               true_masks,
                               pred_masks,
                               num_samples=None,
                               figsize=(15, 10),
                               save_path=None):
    """
    可视化掩码分割预测结果与真实标签的对比。
    
    Args:
        images (torch.Tensor): 批量图像张量，形状为 [B, C, H, W]
        true_masks (torch.Tensor): 真实掩码，形状为 [B, 1, H, W]
        pred_masks (torch.Tensor): 预测掩码 logits，形状为 [B, 1, H, W]
        num_samples (int, optional): 要可视化的样本数量，默认为全部
        figsize (tuple): 图像尺寸
        save_path (str, optional): 保存图像的路径，如果为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    batch_size = images.shape[0]

    if num_samples is None:
        num_samples = batch_size
    else:
        num_samples = min(num_samples, batch_size)

    # 创建子图：每行3列（原图、真实掩码、预测掩码）
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        img_idx_to_show = i

        # 反归一化图像用于显示
        img = denormalize_image(images[img_idx_to_show]).permute(
            1, 2, 0).cpu().numpy()

        # 获取真实掩码和预测掩码
        true_mask = true_masks[img_idx_to_show].squeeze().cpu().numpy()
        pred_mask_logits = pred_masks[img_idx_to_show].squeeze().cpu().numpy()
        pred_mask_probs = torch.sigmoid(
            pred_masks[img_idx_to_show]).squeeze().cpu().numpy()

        # 第一列：原图
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {img_idx_to_show} - Original")
        axes[i, 0].axis('off')

        # 第二列：真实掩码叠加在原图上
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(true_mask, alpha=0.5, cmap='Greens')
        axes[i, 1].set_title(f"Sample {img_idx_to_show} - True Mask")
        axes[i, 1].axis('off')

        # 第三列：预测掩码叠加在原图上
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(pred_mask_probs,
                          alpha=0.5,
                          cmap='Reds',
                          vmin=0,
                          vmax=1)
        axes[i, 2].set_title(f"Sample {img_idx_to_show} - Predicted Mask")
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def visualize_validation_masks_with_postprocessing(images,
                                                   true_masks,
                                                   pred_masks,
                                                   num_samples=None,
                                                   figsize=(20, 10),
                                                   save_path=None,
                                                   show_corners=True):
    """
    可视化验证阶段的掩码分割结果，可选择显示通过后处理提取的角点。
    
    Args:
        images (torch.Tensor): 批量图像张量，形状为 [B, C, H, W]
        true_masks (torch.Tensor): 真实掩码，形状为 [B, 1, H, W]
        pred_masks (torch.Tensor): 预测掩码 logits，形状为 [B, 1, H, W]
        num_samples (int, optional): 要可视化的样本数量，默认为全部
        figsize (tuple): 图像尺寸
        save_path (str, optional): 保存图像的路径，如果为None则不保存
        show_corners (bool): 是否显示通过后处理提取的角点
        
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    batch_size = images.shape[0]
    img_h, img_w = images.shape[2], images.shape[3]

    if num_samples is None:
        num_samples = batch_size
    else:
        num_samples = min(num_samples, batch_size)

    # 创建子图：根据是否显示角点决定列数
    num_cols = 4 if show_corners else 3  # 原图、真实掩码、预测掩码、[角点对比]
    fig, axes = plt.subplots(num_samples, num_cols, figsize=figsize)

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # 如果需要显示角点，使用后处理函数提取
    pred_corners_data = None
    if show_corners:
        try:
            pred_mask_probs = torch.sigmoid(pred_masks)
            pred_corners_data = mask_to_longest_edge_corners(
                pred_mask_probs,
                original_image_width=img_w,
                original_image_height=img_h,
                max_objects=5)
        except Exception as e:
            print(f"角点后处理失败，将跳过角点显示: {e}")
            show_corners = False

    for i in range(num_samples):
        img_idx_to_show = i

        # 反归一化图像用于显示
        img = denormalize_image(images[img_idx_to_show]).permute(
            1, 2, 0).cpu().numpy()

        # 获取真实掩码和预测掩码
        true_mask = true_masks[img_idx_to_show].squeeze().cpu().numpy()
        pred_mask_probs = torch.sigmoid(
            pred_masks[img_idx_to_show]).squeeze().cpu().numpy()

        # 第一列：原图
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {img_idx_to_show} - Original")
        axes[i, 0].axis('off')

        # 第二列：真实掩码叠加在原图上
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(true_mask, alpha=0.5, cmap='Greens')
        axes[i, 1].set_title(f"Sample {img_idx_to_show} - True Mask")
        axes[i, 1].axis('off')

        # 第三列：预测掩码叠加在原图上
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(pred_mask_probs,
                          alpha=0.5,
                          cmap='Reds',
                          vmin=0,
                          vmax=1)
        axes[i, 2].set_title(f"Sample {img_idx_to_show} - Predicted Mask")
        axes[i, 2].axis('off')

        # 第四列（可选）：显示后处理提取的角点
        if show_corners and pred_corners_data is not None:
            corners_tensor, num_objects_tensor = pred_corners_data
            axes[i, 3].imshow(img)
            axes[i,
                 3].set_title(f"Sample {img_idx_to_show} - Extracted Corners")

            # 获取该样本的物体数量
            num_objects = num_objects_tensor[img_idx_to_show].item()

            # 为不同物体使用不同颜色
            colors = ['red', 'blue', 'green', 'cyan', 'magenta']

            for obj_idx in range(num_objects):
                # 获取当前物体的角点坐标（像素坐标）
                obj_corners = corners_tensor[img_idx_to_show,
                                             obj_idx].cpu().numpy()

                # 跳过NaN值
                if np.isnan(obj_corners).any():
                    continue

                # 选择颜色
                color = colors[obj_idx % len(colors)]

                # 绘制角点
                if len(obj_corners) >= 4:  # 确保有足够的坐标
                    x_coords = [obj_corners[0], obj_corners[2]]
                    y_coords = [obj_corners[1], obj_corners[3]]

                    axes[i, 3].scatter(x_coords,
                                       y_coords,
                                       c=color,
                                       s=50,
                                       alpha=0.8,
                                       label=f'Object {obj_idx+1}')
                    # 连接角点
                    axes[i, 3].plot(x_coords,
                                    y_coords,
                                    color=color,
                                    linewidth=2,
                                    alpha=0.7)

            # 叠加预测掩码以便对比
            axes[i, 3].imshow(pred_mask_probs,
                              alpha=0.3,
                              cmap='Reds',
                              vmin=0,
                              vmax=1)

            if num_objects > 0:
                axes[i, 3].legend(fontsize=8)
            axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def demo_visualize_dataset():
    """
    演示可视化数据集样本、变换和预测（从掩码派生角点）。
    """
    try:
        vis_dir = os.path.join(project_root, settings.VISUALIZATION_DIR,
                               "demo")
        os.makedirs(vis_dir, exist_ok=True)

        # 1. 可视化原始数据集样本
        print("正在可视化原始数据集样本...")
        raw_dataset = CornerPointDataset(mode='train', transform=None)
        if not len(raw_dataset):
            print("原始数据集为空，跳过可视化。")
            return False
        raw_samples_path = os.path.join(vis_dir,
                                        'demo_raw_dataset_samples.png')
        visualize_dataset_samples(raw_dataset,
                                  num_samples=min(3, len(raw_dataset)),
                                  save_path=raw_samples_path)
        print(f"原始样本已保存到: {raw_samples_path}")

        # 2. 可视化数据变换前后的对比
        print("\n正在可视化数据变换效果...")
        preprocessor = Preprocessing()
        # We pass the raw_dataset to visualize_transformed_samples, it will apply the transform internally for vis.
        transform_vis_path = os.path.join(vis_dir,
                                          'demo_transform_comparison.png')
        visualize_transformed_samples(raw_dataset,
                                      transform=preprocessor,
                                      num_samples=min(3, len(raw_dataset)),
                                      save_path=transform_vis_path)
        print(f"变换对比已保存到: {transform_vis_path}")

        # 3. 演示预测结果可视化 (角点从掩码派生)
        print("\n正在演示预测结果可视化 (角点从掩码派生)...")
        preprocessed_dataset = CornerPointDataset(mode='train',
                                                  transform=preprocessor)
        if len(preprocessed_dataset) < 2:
            print("预处理后的数据集样本不足 (少于2个)，跳过预测可视化。")
            # plt.show() # Show previous plots if any
            return True  # Return true if previous steps succeeded

        num_vis_preds = min(2, len(preprocessed_dataset))

        # 获取样本
        samples_for_pred_vis = [
            preprocessed_dataset[i] for i in range(num_vis_preds)
        ]

        images_tensor = torch.stack([s['image'] for s in samples_for_pred_vis])
        true_corners_tensor = torch.stack([
            s['corner'] for s in samples_for_pred_vis
        ])  # Normalized GT corners
        true_masks_tensor = torch.stack(
            [s['mask'] for s in samples_for_pred_vis])

        # 模拟模型输出的掩码 logits (例如，随机值，或者可以是一个简单的形状)
        # For a more meaningful demo, let's try to make mock_pred_mask_logits somewhat resemble true_masks
        mock_pred_mask_logits = torch.randn_like(
            true_masks_tensor)  # Random noise
        # Slightly bias towards the true mask to make visualization more interesting
        mock_pred_mask_logits = (
            true_masks_tensor * 2.0 -
            1.0) * 1.5 + torch.randn_like(true_masks_tensor) * 0.5

        pred_vis_path = os.path.join(vis_dir,
                                     'demo_predictions_visualization.png')
        visualize_predictions(
            images_tensor,
            true_corners=true_corners_tensor,
            true_masks=true_masks_tensor,
            pred_masks=
            mock_pred_mask_logits,  # Pass logits, corners will be derived
            pred_corners=None,  # Explicitly None to trigger derivation
            num_samples=num_vis_preds,
            save_path=pred_vis_path)
        print(f"预测结果可视化已保存到: {pred_vis_path}")

        # plt.show() # Uncomment to display plots if running interactively
        return True

    except Exception as e:
        print(f"可视化演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("开始可视化模块演示...")
    success = demo_visualize_dataset()
    if success:
        print("\n可视化模块演示成功完成!")
    else:
        print("\n可视化模块演示失败，请检查错误信息。")
