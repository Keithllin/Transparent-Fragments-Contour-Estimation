"""
数据增强模块 - 专为真实环境多目标识别设计

该模块使用albumentations库实现各种数据增强技术，包括：
- 几何变换：旋转、缩放、平移、翻转
- 颜色增强：亮度、对比度、饱和度调整
- 噪声模糊：高斯噪声、椒盐噪声、模糊效果
- 触觉专用：压力模拟、表面纹理增强
- 多目标增强：遮挡模拟、部分可见性处理
- 高级技术：弹性变形、Cutout增强

注意：训练时只处理图像和掩码，关键点仅在验证时使用。
"""

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random
import sys
import os

sys.path.append("..")
from config import settings


class TactileAugmentation:
    """
    触觉图像专用数据增强类
    
    只处理图像和掩码的同步变换，训练时不处理关键点。
    """

    def __init__(self, mode='train', use_augmentation=True):
        """
        初始化数据增强器
        
        Args:
            mode (str): 'train' 或 'test'
            use_augmentation (bool): 是否启用数据增强
        """
        self.mode = mode
        self.use_augmentation = use_augmentation and settings.ENABLE_DATA_AUGMENTATION

        # 只在训练模式且启用增强时创建增强管道
        if self.use_augmentation and mode == 'train':
            self.transform = self._create_augmentation_pipeline()
        else:
            self.transform = A.Compose([A.NoOp()])  # 无操作，保持原样

        print(
            f"TactileAugmentation initialized: mode={mode}, use_augmentation={self.use_augmentation}"
        )

    def _create_augmentation_pipeline(self):
        """
        创建数据增强管道 - 只处理图像和掩码
        """
        transforms = []

        # 1. 几何变换 - 概率较高，对触觉数据很重要
        geometric_transforms = [
            A.Rotate(limit=settings.ROTATION_RANGE,
                     p=0.6,
                     border_mode=cv2.BORDER_REFLECT_101),
            A.ShiftScaleRotate(
                shift_limit=settings.TRANSLATE_RANGE,
                scale_limit=(settings.SCALE_RANGE[0] - 1,
                             settings.SCALE_RANGE[1] - 1),
                rotate_limit=0,  # 旋转已经在上面处理
                p=0.5,
                border_mode=cv2.BORDER_REFLECT_101),
            A.HorizontalFlip(p=settings.HORIZONTAL_FLIP_PROB),
            A.VerticalFlip(p=settings.VERTICAL_FLIP_PROB),
        ]

        # 2. 颜色增强 - 模拟不同光照条件
        color_transforms = [
            A.RandomBrightnessContrast(
                brightness_limit=(settings.BRIGHTNESS_RANGE[0] - 1,
                                  settings.BRIGHTNESS_RANGE[1] - 1),
                contrast_limit=(settings.CONTRAST_RANGE[0] - 1,
                                settings.CONTRAST_RANGE[1] - 1),
                p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=int(settings.HUE_RANGE[1] *
                                    180),  # albumentations uses degree
                sat_shift_limit=int((settings.SATURATION_RANGE[1] - 1) * 100),
                val_shift_limit=0,  # 亮度已经在上面处理
                p=0.5),
            A.ChannelShuffle(p=0.1),  # 偶尔交换颜色通道
        ]

        # 3. 噪声和模糊 - 模拟真实环境干扰
        noise_blur_transforms = [
            A.GaussNoise(var_limit=settings.GAUSSIAN_NOISE_VAR_RANGE,
                         p=settings.GAUSSIAN_NOISE_PROB),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.MotionBlur(blur_limit=settings.BLUR_KERNEL_SIZE_RANGE,
                         p=settings.MOTION_BLUR_PROB),
            A.GaussianBlur(blur_limit=settings.BLUR_KERNEL_SIZE_RANGE,
                           p=settings.GAUSSIAN_BLUR_PROB),
            A.MedianBlur(blur_limit=5, p=0.1),
        ]

        # 4. 遮挡和缺失模拟
        occlusion_transforms = [
            A.CoarseDropout(max_holes=settings.CUTOUT_NUM_HOLES[1],
                            max_height=int(settings.CUTOUT_HOLE_SIZE_RANGE[1] *
                                           settings.IMAGE_HEIGHT),
                            max_width=int(settings.CUTOUT_HOLE_SIZE_RANGE[1] *
                                          settings.IMAGE_WIDTH),
                            min_holes=settings.CUTOUT_NUM_HOLES[0],
                            min_height=int(settings.CUTOUT_HOLE_SIZE_RANGE[0] *
                                           settings.IMAGE_HEIGHT),
                            min_width=int(settings.CUTOUT_HOLE_SIZE_RANGE[0] *
                                          settings.IMAGE_WIDTH),
                            fill_value=0,
                            p=settings.CUTOUT_PROB),
            A.GridDropout(ratio=0.3,
                          unit_size_min=10,
                          unit_size_max=20,
                          holes_number_x=5,
                          holes_number_y=5,
                          p=settings.COARSE_DROPOUT_PROB),
        ]

        # 5. 光照和阴影效果
        lighting_transforms = [
            A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                           num_shadows_lower=1,
                           num_shadows_upper=3,
                           shadow_dimension=5,
                           p=settings.SHADOW_PROB),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 1),
                             angle_lower=0,
                             angle_upper=1,
                             num_flare_circles_lower=1,
                             num_flare_circles_upper=3,
                             p=0.1),
        ]

        # 6. 触觉专用增强
        tactile_transforms = [
            A.Sharpen(alpha=(0.2, 0.5),
                      lightness=(0.5, 1.0),
                      p=settings.TACTILE_EDGE_ENHANCEMENT),
            A.Emboss(alpha=(0.2, 0.5),
                     strength=(0.2, 0.7),
                     p=settings.TACTILE_SURFACE_ROUGHNESS),
        ]

        # 使用OneOf来随机选择部分变换，避免过度增强
        final_transforms = [
            # 几何变换（大概率应用）
            A.OneOf(geometric_transforms[:2], p=0.8),  # 旋转和平移缩放
            A.OneOf(geometric_transforms[2:], p=0.6),  # 翻转

            # 颜色增强（中等概率）
            A.OneOf(color_transforms, p=0.7),

            # 噪声模糊（较低概率）
            A.OneOf(noise_blur_transforms, p=0.5),

            # 遮挡（低概率）
            A.OneOf(occlusion_transforms, p=0.3),

            # 光照效果（低概率）
            A.OneOf(lighting_transforms, p=0.3),

            # 触觉专用（中等概率）
            A.OneOf(tactile_transforms, p=0.4),
        ]

        return A.Compose(final_transforms)

    def _convert_dataset_tensors_to_augmentation_format(self, sample):
        """
        将数据集返回的张量格式转换为数据增强所需的格式
        
        Args:
            sample (dict): 数据集返回的样本，可能包含张量
            
        Returns:
            dict: 转换为PIL图像、numpy数组格式的样本
        """
        image = sample['image']
        mask = sample['mask']

        # 处理图像数据
        if torch.is_tensor(image):
            # 如果是归一化的张量，需要反归一化
            if image.dim() == 3 and image.shape[0] == 3:  # (C, H, W)
                # 反归一化
                mean = torch.tensor(settings.NORMALIZE_MEAN).view(3, 1, 1)
                std = torch.tensor(settings.NORMALIZE_STD).view(3, 1, 1)
                image = image * std + mean
                image = torch.clamp(image, 0, 1)
                # 转换为PIL格式 (H, W, C) 并转为uint8
                image_np = (image.permute(1, 2, 0).numpy() * 255).astype(
                    np.uint8)
                image = Image.fromarray(image_np)

        # 处理掩码数据
        if torch.is_tensor(mask):
            if mask.dim() == 3 and mask.shape[0] == 1:  # (1, H, W)
                mask_np = mask.squeeze(0).numpy()  # (H, W)
            else:
                mask_np = mask.numpy()
        else:
            mask_np = mask

        return {'image': image, 'mask': mask_np.astype(np.float32)}

    def __call__(self, sample):
        """
        对样本应用数据增强 - 只处理图像和掩码
        
        Args:
            sample (dict): 包含 'image', 'mask' 的样本
            
        Returns:
            dict: 增强后的样本
        """
        # 如果不使用增强，直接返回原始样本
        if not self.use_augmentation or self.mode != 'train':
            return sample

        # 检查数据是否来自数据集（包含张量）
        is_tensor_data = (torch.is_tensor(sample['image'])
                          or torch.is_tensor(sample['mask']))

        if is_tensor_data:
            # 转换张量数据为数据增强格式
            sample = self._convert_dataset_tensors_to_augmentation_format(
                sample)

        # 提取数据
        image = sample['image']
        mask = sample['mask']

        # 转换PIL图像为numpy数组
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # 确保mask是numpy数组
        if not isinstance(mask, np.ndarray):
            mask_np = np.array(mask, dtype=np.float32)
        else:
            mask_np = mask.astype(np.float32)

        # 应用增强
        try:
            transformed = self.transform(image=image_np, mask=mask_np)

            # 提取变换结果
            aug_image = transformed['image']
            aug_mask = transformed['mask']

            # 转换回PIL格式（如果原来是PIL）
            if isinstance(image, Image.Image):
                aug_image = Image.fromarray(aug_image)

            return {'image': aug_image, 'mask': aug_mask}

        except Exception as e:
            print(f"Data augmentation failed: {e}")
            print(
                f"原始数据 - image shape: {image_np.shape}, mask: {mask_np.shape}")
            # 返回原始数据
            return sample

    def visualize_augmentation(self, sample, num_examples=6, save_path=None):
        """
        可视化数据增强效果 - 只显示图像和掩码
        
        Args:
            sample (dict): 原始样本
            num_examples (int): 生成的增强示例数量
            save_path (str): 保存路径
        """
        if not self.use_augmentation:
            print("Data augmentation not enabled, skipping visualization")
            return

        fig, axes = plt.subplots(2,
                                 num_examples + 1,
                                 figsize=(4 * (num_examples + 1), 8))

        # 显示原始图像
        original_image = sample['image']
        original_mask = sample['mask']

        if isinstance(original_image, Image.Image):
            original_image_np = np.array(original_image)
        else:
            original_image_np = original_image

        # 原始图像
        axes[0, 0].imshow(original_image_np)
        axes[0, 0].set_title('Original image')
        axes[0, 0].axis('off')

        # 原始掩码
        axes[1, 0].imshow(original_mask, cmap='gray')
        axes[1, 0].set_title('Original mask')
        axes[1, 0].axis('off')

        # 生成增强示例
        for i in range(num_examples):
            try:
                aug_sample = self(
                    sample.copy() if hasattr(sample, 'copy') else {
                        'image':
                        original_image,
                        'mask':
                        original_mask.copy() if hasattr(original_mask, 'copy'
                                                        ) else original_mask
                    })

                aug_image = aug_sample['image']
                aug_mask = aug_sample['mask']

                if isinstance(aug_image, Image.Image):
                    aug_image_np = np.array(aug_image)
                else:
                    aug_image_np = aug_image

                # 增强图像
                axes[0, i + 1].imshow(aug_image_np)
                axes[0, i + 1].set_title(f'Augmented example {i + 1}')
                axes[0, i + 1].axis('off')

                # 增强掩码
                axes[1, i + 1].imshow(aug_mask, cmap='gray')
                axes[1, i + 1].set_title(f'Augmented mask {i + 1}')
                axes[1, i + 1].axis('off')

            except Exception as e:
                print(f"Error generating augmented example {i + 1}: {e}")
                # 显示错误信息
                axes[0, i + 1].text(0.5,
                                    0.5,
                                    f'Error: {str(e)[:50]}...',
                                    ha='center',
                                    va='center',
                                    transform=axes[0, i + 1].transAxes)
                axes[0, i + 1].set_title(f'Error example {i + 1}')
                axes[1, i + 1].text(0.5,
                                    0.5,
                                    'Error',
                                    ha='center',
                                    va='center',
                                    transform=axes[1, i + 1].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            default_path = "augmentation_visualization.png"
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {default_path}")

        plt.close()

    def get_augmentation_info(self):
        """
        获取增强配置信息
        """
        info = {
            'mode': self.mode,
            'use_augmentation': self.use_augmentation,
            'enabled': settings.ENABLE_DATA_AUGMENTATION,
            'transforms': []
        }

        if self.use_augmentation:
            info['transforms'] = [
                'geometric_transforms', 'color_transforms',
                'noise_blur_transforms', 'occlusion_transforms',
                'lighting_transforms', 'tactile_transforms'
            ]

        return info


def create_detailed_visualization(dataset, save_dir="./"):
    """
    创建详细的数据增强可视化，展示每个增强步骤的效果
    
    Args:
        dataset: 数据集对象
        save_dir: 保存目录
    """
    if len(dataset) == 0:
        print("Dataset is empty, cannot create visualization")
        return

    # 创建增强器
    augmentation = TactileAugmentation(mode='train', use_augmentation=True)

    # 获取第一个样本
    sample = dataset[0]

    # 创建步骤可视化
    print("Creating detailed data augmentation visualization...")

    # 可视化增强效果
    save_path = os.path.join(save_dir,
                             "detailed_augmentation_visualization.png")
    augmentation.visualize_augmentation(sample,
                                        num_examples=6,
                                        save_path=save_path)

    print("Detailed visualization created!")


def create_detailed_visualization_from_sample(sample, save_dir="./"):
    """
    从单个样本创建详细的数据增强可视化
    
    Args:
        sample: 单个样本数据
        save_dir: 保存目录
    """
    # 创建增强器
    augmentation = TactileAugmentation(mode='train', use_augmentation=True)

    # 创建步骤可视化
    print("Creating detailed data augmentation visualization...")

    # 可视化增强效果
    save_path = os.path.join(save_dir,
                             "detailed_augmentation_visualization.png")
    augmentation.visualize_augmentation(sample,
                                        num_examples=6,
                                        save_path=save_path)

    print("Detailed visualization created!")


if __name__ == '__main__':
    """
    测试数据增强功能 - 使用真实数据集
    """
    print("开始测试数据增强功能...")

    # 导入数据集类
    try:
        from .dataset import CornerPointDataset
    except ImportError:
        # 如果相对导入失败，尝试直接导入
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from data.dataset import CornerPointDataset

    # 创建测试数据集
    try:
        print("正在加载真实数据集...")
        test_dataset = CornerPointDataset(mode='train',
                                          transform=None,
                                          use_augmentation=False)

        if len(test_dataset) == 0:
            print("错误: 数据集为空，请检查数据集路径和文件")
            exit(1)

        print(f"数据集加载成功，共有 {len(test_dataset)} 个样本")

        # 获取第一个原始样本
        test_sample = test_dataset.get_raw_sample(0)
        print(f"获取样本 0:")
        print(f"  图像类型: {type(test_sample['image'])}")
        print(f"  图像尺寸: {test_sample['image'].size}")
        print(f"  掩码形状: {test_sample['mask'].shape}")
        print(f"  掩码数据类型: {test_sample['mask'].dtype}")
        print(
            f"  掩码值范围: [{test_sample['mask'].min():.3f}, {test_sample['mask'].max():.3f}]"
        )
        print(f"  角点形状: {test_sample['corner'].shape}")
        print(f"  角点坐标: {test_sample['corner']}")

    except Exception as e:
        print(f"数据集加载失败: {e}")
        print("创建备用测试数据...")
        # 使用原来的测试数据作为备用
        test_image = Image.new('RGB', (640, 480), color=(100, 150, 200))
        test_mask = np.zeros((480, 640), dtype=np.float32)
        test_mask[50:350, 80:420] = 1.0
        test_sample = {'image': test_image, 'mask': test_mask}

    # 测试训练模式（启用增强）
    print(f"\n测试训练模式数据增强...")
    train_augmentation = TactileAugmentation(mode='train',
                                             use_augmentation=True)
    print(f"增强配置信息: {train_augmentation.get_augmentation_info()}")

    # 应用增强
    try:
        augmented_sample = train_augmentation(test_sample.copy())
        print("数据增强成功!")

        if isinstance(test_sample['image'], Image.Image):
            print(f"原始图像尺寸: {test_sample['image'].size}")
        else:
            print(f"原始图像形状: {test_sample['image'].shape}")

        print(f"增强后掩码形状: {augmented_sample['mask'].shape}")
        print(
            f"增强后掩码值范围: [{augmented_sample['mask'].min():.3f}, {augmented_sample['mask'].max():.3f}]"
        )

        # 创建可视化
        print("创建数据增强可视化...")
        save_path = "augmentation_visualization_real_data.png"
        train_augmentation.visualize_augmentation(test_sample,
                                                  num_examples=6,
                                                  save_path=save_path)

        # 如果成功加载了数据集，测试多个样本
        if 'test_dataset' in locals() and len(test_dataset) > 5:
            print("\n测试多个样本的数据增强...")
            for i in range(min(3, len(test_dataset))):
                try:
                    sample = test_dataset.get_raw_sample(i)
                    aug_sample = train_augmentation(sample.copy())
                    print(f"样本 {i}: 增强成功")
                except Exception as e:
                    print(f"样本 {i}: 增强失败 - {e}")

    except Exception as e:
        print(f"数据增强测试出错: {e}")
        import traceback
        traceback.print_exc()

    # 测试测试模式（禁用增强）
    print("\n测试测试模式（不使用增强）...")
    test_augmentation = TactileAugmentation(mode='test',
                                            use_augmentation=False)
    test_result = test_augmentation(test_sample.copy())
    print("测试模式结果（应该与原始数据相同）")

    # 比较图像是否相同
    if isinstance(test_sample['image'], Image.Image) and isinstance(
            test_result['image'], Image.Image):
        original_array = np.array(test_sample['image'])
        result_array = np.array(test_result['image'])
        images_same = np.array_equal(original_array, result_array)
        print(f"图像是否相同: {images_same}")
    else:
        print("图像类型不同，无法直接比较")

    # 比较掩码是否相同
    masks_same = np.array_equal(test_sample['mask'], test_result['mask'])
    print(f"掩码是否相同: {masks_same}")

    print(f"\n数据增强测试完成! 使用真实数据集进行测试")
