import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import sys
import math
import matplotlib.pyplot as plt

sys.path.append("..")
from config import settings
# from dataset import CornerPointDataset # 移除此处的全局导入
# from dsntnn import pixel_to_normalized_coordinates, make_gauss  # 导入 dsntnn 函数

# Removed: generate_gaussian_heatmap function as it's no longer needed
# def generate_gaussian_heatmap(image_height, image_width, corner_coords_pixel,
#                               sigma, num_keypoints):
#     ... (function content removed) ...


class Preprocessing:
    """
    数据预处理类，负责图像和掩码的转换。
    训练时只处理图像和掩码，角点处理是可选的。
    """

    def __init__(self,
                 mode='train',
                 use_augmentation=None,
                 load_corners=False):
        self.image_width = settings.IMAGE_WIDTH
        self.image_height = settings.IMAGE_HEIGHT
        self.normalize_mean = settings.NORMALIZE_MEAN
        self.normalize_std = settings.NORMALIZE_STD
        self.mode = mode
        self.load_corners = load_corners  # 控制是否处理角点数据

        # 智能设置数据增强
        if use_augmentation is None:
            # 训练模式默认启用数据增强，测试模式默认禁用
            use_augmentation = (
                mode == 'train') and settings.ENABLE_DATA_AUGMENTATION

        self.use_augmentation = use_augmentation

        # 初始化数据增强器
        if self.use_augmentation:
            try:
                from .augmentation import TactileAugmentation
                self.augmentation = TactileAugmentation(
                    mode=mode, use_augmentation=use_augmentation)
                print(f"数据增强已启用 (mode: {mode})")
            except ImportError as e:
                print(f"警告: 无法导入数据增强模块，将禁用数据增强。错误: {e}")
                self.use_augmentation = False
                self.augmentation = None
        else:
            self.augmentation = None
            print(f"数据增强已禁用 (mode: {mode})")

    def __call__(self, sample):
        """
        对输入的样本（包含图像、掩码，可选角点）进行处理。

        Args:
            sample (dict): 包含以下键值对的字典:
                'image' (PIL.Image.Image): 输入的RGB图像。
                'mask' (numpy.ndarray): 标注掩码，二维数组。
                'corner' (numpy.ndarray, optional): 角点坐标，形状类似 [[x1, y1], [x2, y2]]。
                                                   只在load_corners=True时处理。

        Returns:
            dict: 包含处理后的张量的字典:
                'image' (torch.Tensor): 形状为 (C, H, W) 的图像张量，已归一化。
                'mask' (torch.Tensor): 形状为 (1, H, W) 的掩码张量。
                'corner' (torch.Tensor, optional): 只在load_corners=True且输入包含角点时返回。
        """
        image_pil = sample['image']
        anno_mask_original = sample['mask']
        corner_coords_pixel_original = sample.get(
            'corner', None) if self.load_corners else None

        # 0. 获取原始图像尺寸
        original_pil_width, original_pil_height = image_pil.size

        # 1. 数据增强（在原始数据上进行）
        if self.use_augmentation and self.augmentation is not None:
            try:
                if self.load_corners and corner_coords_pixel_original is not None:
                    # 如果需要处理角点数据
                    augmented_sample = self.augmentation(sample)
                    image_pil = augmented_sample['image']
                    anno_mask_original = augmented_sample['mask']
                    corner_coords_pixel_original = augmented_sample.get(
                        'corner')
                else:
                    # 只处理图像和掩码
                    augment_sample = {
                        'image': image_pil,
                        'mask': anno_mask_original
                    }
                    augmented_sample = self.augmentation(augment_sample)
                    image_pil = augmented_sample['image']
                    anno_mask_original = augmented_sample['mask']

                # 更新图像尺寸（如果增强改变了尺寸）
                if isinstance(image_pil, Image.Image):
                    original_pil_width, original_pil_height = image_pil.size
                else:
                    # 如果返回的是numpy数组，转换为PIL
                    if isinstance(image_pil, np.ndarray):
                        image_pil = Image.fromarray(image_pil)
                        original_pil_width, original_pil_height = image_pil.size

            except Exception as e:
                print(f"数据增强失败，使用原始数据: {e}")
                # 保持原始数据不变

        # 2. 图像转换
        image_resized_pil = image_pil.resize(
            (self.image_width, self.image_height), Image.LANCZOS)
        image_tensor = TF.to_tensor(image_resized_pil)
        image_tensor = TF.normalize(image_tensor,
                                    mean=self.normalize_mean,
                                    std=self.normalize_std)

        # 3. 掩码转换
        if not isinstance(anno_mask_original, np.ndarray):
            anno_mask_np = np.array(anno_mask_original, dtype=np.float32)
        else:
            anno_mask_np = anno_mask_original.astype(np.float32)

        mask_pil = Image.fromarray(anno_mask_np)
        mask_resized_pil = mask_pil.resize(
            (self.image_width, self.image_height), Image.NEAREST)
        mask_resized_np = np.array(mask_resized_pil, dtype=np.float32)

        binary_mask = (mask_resized_np > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()

        # 4. 组织返回结果
        result = {'image': image_tensor, 'mask': mask_tensor}

        # 5. 角点处理（只在需要且有数据时进行）
        if self.load_corners and corner_coords_pixel_original is not None:
            if not isinstance(corner_coords_pixel_original, np.ndarray):
                corners_pixel_original_np = np.array(
                    corner_coords_pixel_original, dtype=np.float32)
            else:
                corners_pixel_original_np = corner_coords_pixel_original.astype(
                    np.float32)

            # 动态确定 num_keypoints，通常应为2个角点
            num_keypoints_from_data = corners_pixel_original_np.shape[0]
            if num_keypoints_from_data * 2 != settings.NUM_CORNERS:  # settings.NUM_CORNERS is 4 (x1,y1,x2,y2)
                raise ValueError(
                    f"输入角点数据形状 ({corners_pixel_original_np.shape}) "
                    f"与 settings.NUM_CORNERS ({settings.NUM_CORNERS}) 不兼容。期望得到 {settings.NUM_CORNERS // 2} 个角点。"
                )
            if corners_pixel_original_np.shape[1] != 2:
                raise ValueError(
                    f"每个角点应有2个坐标 (x,y)，但得到形状: {corners_pixel_original_np.shape}"
                )

            # 角点排序 (按 x 坐标，然后按 y 坐标)
            sort_indices = np.lexsort(
                (corners_pixel_original_np[:,
                                           1], corners_pixel_original_np[:,
                                                                         0]))
            sorted_corners_pixel_original_np = corners_pixel_original_np[
                sort_indices]

            # 角点归一化
            sorted_corners_normalized_for_eval = sorted_corners_pixel_original_np.copy(
            )
            sorted_corners_normalized_for_eval[:,
                                               0] /= float(original_pil_width)
            sorted_corners_normalized_for_eval[:,
                                               1] /= float(original_pil_height)
            sorted_corners_normalized_for_eval = np.clip(
                sorted_corners_normalized_for_eval, 0, 1)
            corner_tensor = torch.tensor(
                sorted_corners_normalized_for_eval.flatten(),
                dtype=torch.float32)

            result['corner'] = corner_tensor

        return result


if __name__ == '__main__':
    print("开始 Preprocessing 测试 (无热图)...")

    from data.dataset import CornerPointDataset

    try:
        # 在测试中，我们通常想验证包括角点在内的所有处理，因此使用 'val' 模式
        preprocessor = Preprocessing(mode='val')
        print(
            f"使用 'val' 模式进行测试。图像尺寸: {settings.IMAGE_WIDTH}x{settings.IMAGE_HEIGHT}, NUM_CORNERS: {settings.NUM_CORNERS}"
        )
    except Exception as e:
        print(f"初始化 Preprocessing 失败: {e}")
        exit()

    sample_to_process = None
    is_real_data = False
    original_pil_image_for_plot = None
    original_corners_input_for_plot = None

    try:
        print(f"尝试从真实数据集加载样本，路径: {settings.DATASET_ROOT_DIR}")
        real_dataset = CornerPointDataset(mode='train', transform=None)
        if len(real_dataset) > 0:
            sample_idx_to_test = 0
            sample_to_process = real_dataset[sample_idx_to_test]
            original_pil_image_for_plot = sample_to_process['image'].copy()

            # 修复：鲁棒地复制角点数据，无论它是 tensor 还是 numpy array
            corners_data = sample_to_process['corner']
            if isinstance(corners_data, torch.Tensor):
                # PyTorch Tensor 使用 .clone() 而不是 .copy()
                original_corners_input_for_plot = corners_data.clone().cpu(
                ).numpy()
            elif isinstance(corners_data, np.ndarray):
                original_corners_input_for_plot = corners_data.copy()
            else:
                # 对于其他类型，转换为 numpy 数组
                original_corners_input_for_plot = np.array(corners_data,
                                                           dtype=np.float32)

            print(f"成功从真实数据集中加载第 {sample_idx_to_test} 个样本。")
            print(f"  原始输入角点 (未排序):\\n{original_corners_input_for_plot}")
            is_real_data = True
        else:
            print("警告: 真实训练数据集中没有样本。")
    except FileNotFoundError as fnf_error:
        print(f"加载真实数据集时文件未找到: {fnf_error}。请检查 settings.DATASET_ROOT_DIR。")
    except Exception as e:
        print(f"加载真实数据集时出错: {e}。")

    if not is_real_data:
        print("测试将回退到使用虚拟数据。")
        original_width, original_height = 800, 600
        dummy_image_pil = Image.new('RGB', (original_width, original_height),
                                    color='blue')
        # settings.NUM_CORNERS is 4 (meaning 2 points)
        # Ensure dummy corners provide 2 points (4 coordinates)
        num_points_expected = settings.NUM_CORNERS // 2
        if num_points_expected == 2:
            dummy_corners_original_pixel = np.array(
                [[550.0, 450.0], [150.0, 200.0]], dtype=np.float32)  # 2 points
        elif num_points_expected == 1:  # Should not happen with NUM_CORNERS = 4
            dummy_corners_original_pixel = np.array([[150.0, 200.0]],
                                                    dtype=np.float32)
        else:
            dummy_corners_original_pixel = np.random.rand(
                num_points_expected, 2).astype(np.float32)
            dummy_corners_original_pixel[:, 0] *= original_width
            dummy_corners_original_pixel[:, 1] *= original_height

        dummy_mask_original = (np.random.rand(original_height, original_width)
                               > 0.7).astype(np.float32)

        sample_to_process = {
            'image': dummy_image_pil,
            'corner': dummy_corners_original_pixel,
            'mask': dummy_mask_original
        }
        original_pil_image_for_plot = dummy_image_pil.copy()
        original_corners_input_for_plot = dummy_corners_original_pixel.copy()
        print("已创建虚拟数据用于测试。")
        print(f"  虚拟输入角点 (未排序):\\n{original_corners_input_for_plot}")

    try:
        transformed_sample = preprocessor(sample_to_process)
        print("样本成功转换。")
    except Exception as e:
        print(f"转换样本时出错: {e}")
        import traceback
        traceback.print_exc()
        exit()

    print("\\n转换后样本信息:")
    print(
        f"  图像张量形状: {transformed_sample['image'].shape}, dtype: {transformed_sample['image'].dtype}"
    )
    # 修复：在访问 'corner' 之前检查它是否存在
    if 'corner' in transformed_sample:
        print(
            f"  排序归一化角点张量 (评估用): {transformed_sample['corner']}, shape: {transformed_sample['corner'].shape}"
        )
    else:
        print("  角点张量: 在当前模式下未生成。")
    print(
        f"  掩码张量形状: {transformed_sample['mask'].shape}, dtype: {transformed_sample['mask'].dtype}"
    )
    print(f"  掩码张量唯一值: {torch.unique(transformed_sample['mask'])}")
    # print( # Removed: No target_heatmap
    #     f"  目标热图形状: {transformed_sample['target_heatmap'].shape}, dtype: {transformed_sample['target_heatmap'].dtype}"
    # )

    # 只有在角点数据存在时才打印和可视化
    if 'corner' in transformed_sample:
        sorted_corners_pixel_original_torch = transformed_sample[
            '_sorted_corners_pixel_original']
        print(
            f"  内部: 排序后的原始像素角点:\\n{sorted_corners_pixel_original_torch.numpy()}"
        )
        # Renamed variable for clarity as it's no longer specifically for heatmaps
        sorted_corners_resized_pixel_torch = transformed_sample[
            '_sorted_corners_resized_pixel']
        print(
            f"  内部: 排序后并缩放到目标图像尺寸的像素角点:\\n{sorted_corners_resized_pixel_torch.numpy()}"
        )
    else:
        print("  内部角点调试信息: 在当前模式下未生成。")

    # 可视化对比
    try:
        # num_heatmaps_to_plot = settings.NUM_KEYPOINTS # Removed
        num_points_for_plot = settings.NUM_CORNERS // 2  # Assuming NUM_CORNERS is 4 for two (x,y) points

        # Adjusted number of columns as heatmaps are removed
        num_base_plots = 2  # 原始图+角点, Resize图
        # num_heatmaps_to_plot = 0 # No heatmaps
        num_columns = num_base_plots  # + num_heatmaps_to_plot

        fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
        if num_columns == 1:  # Should be 2 now
            axes = [axes]

        # 1. 显示原始图像和 *输入* (可能未排序) 角点
        ax_orig = axes[0]
        if original_pil_image_for_plot:
            orig_img_np = np.array(original_pil_image_for_plot)
            ax_orig.imshow(orig_img_np)
            title_orig = f"Original Image ({orig_img_np.shape[1]}x{orig_img_np.shape[0]})\\nInput Corners (Unsorted)"
            ax_orig.set_title(title_orig)
            ax_orig.axis('off')
            if original_corners_input_for_plot is not None and original_corners_input_for_plot.shape[
                    0] == num_points_for_plot:
                ax_orig.scatter(original_corners_input_for_plot[:, 0],
                                original_corners_input_for_plot[:, 1],
                                c=['red', 'blue', 'green',
                                   'purple'][:num_points_for_plot],
                                s=80,
                                marker='x',
                                label='Input GT (Unsorted)')
                for i in range(num_points_for_plot):
                    ax_orig.annotate(f"P{i}",
                                     (original_corners_input_for_plot[i, 0],
                                      original_corners_input_for_plot[i, 1]),
                                     color='yellow')

            sorted_orig_pixels_np = transformed_sample[
                '_sorted_corners_pixel_original'].numpy()
            ax_orig.scatter(sorted_orig_pixels_np[:, 0],
                            sorted_orig_pixels_np[:, 1],
                            c=['cyan', 'magenta', 'lime',
                               'black'][:num_points_for_plot],
                            s=50,
                            marker='o',
                            alpha=0.7,
                            label='Sorted GT (Original Pixels)')
            for i in range(num_points_for_plot):
                ax_orig.annotate(
                    f"S{i}",
                    (sorted_orig_pixels_np[i, 0], sorted_orig_pixels_np[i, 1]),
                    color='white')
            ax_orig.legend(fontsize='small')
        else:
            ax_orig.text(0.5,
                         0.5,
                         "Original Image N/A",
                         ha='center',
                         va='center')
            ax_orig.set_title("Original Image N/A")
            ax_orig.axis('off')

        # 2. 显示Resize后的图像
        ax_resized = axes[1]
        img_to_show_tensor = transformed_sample['image']
        mean_th = torch.tensor(settings.NORMALIZE_MEAN).view(3, 1, 1)
        std_th = torch.tensor(settings.NORMALIZE_STD).view(3, 1, 1)
        img_to_show_unnorm = img_to_show_tensor.cpu().clone() * std_th + mean_th
        img_to_show_unnorm = torch.clamp(img_to_show_unnorm, 0, 1)
        ax_resized.imshow(img_to_show_unnorm.permute(1, 2, 0).numpy())
        ax_resized.set_title(
            f"Resized Image ({settings.IMAGE_WIDTH}x{settings.IMAGE_HEIGHT})")
        ax_resized.axis('off')

        # 在resize后的图像上绘制 *排序后、缩放后* 的角点 (之前是用于热图的)
        corners_scaled_np = transformed_sample[
            '_sorted_corners_resized_pixel'].numpy()  # Renamed variable
        ax_resized.scatter(
            corners_scaled_np[:, 0],
            corners_scaled_np[:, 1],
            c=['cyan', 'magenta', 'lime', 'black'][:num_points_for_plot],
            s=80,
            marker='o',
            alpha=0.8,
            label='Sorted & Scaled GT (for Resized Img)')  # Label updated
        for i in range(num_points_for_plot):
            ax_resized.annotate(
                f"S{i}", (corners_scaled_np[i, 0], corners_scaled_np[i, 1]),
                color='yellow')
        ax_resized.legend(fontsize='small')

        # 3. 移除目标热图的显示
        # for i in range(num_heatmaps_to_plot):
        #     ... (heatmap plotting code removed) ...

        plt.suptitle(
            f"Preprocessing Test (No Heatmaps) - Real Data: {is_real_data}",  # Title updated
            fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = "preprocessing_test_no_heatmap_output.png"  # Filename updated
        plt.savefig(save_path)
        print(f"\\n测试可视化图像已保存到: {save_path}")

    except Exception as e:
        print(f"\\n可视化对比时出错: {e}")
        import traceback
        traceback.print_exc()

    print(
        "\\nPreprocessing (No Heatmaps) 测试完成。请检查输出形状、值以及图像 \'preprocessing_test_no_heatmap_output.png\' 文件。"
    )
