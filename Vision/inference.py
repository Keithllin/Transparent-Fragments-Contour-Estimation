import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import albumentations as A
from typing import List, Tuple, Any
import sys

# 将当前目录添加到 sys.path 以允许直接导入项目模块
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if current_file_dir not in sys.path:
    sys.path.insert(0, current_file_dir)

from model.network import CornerMaskModel
from config import settings  # 直接从 settings 导入配置
from utils.postprocessing import mask_to_longest_edge_corners, smooth_mask_morphology
from utils.visualize import denormalize_image  # 用于可视化时反归一化
from utils.metrics import intersection_over_union  # 导入IoU计算函数


def get_inference_transforms(img_height, img_width, mean, std):
    """定义推理时使用的图像变换。"""
    return transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_tta_transforms(img_height, img_width, mean, std):
    """定义测试时数据增强（TTA）的变换组合。"""
    # 获取亮度增强配置
    use_brightness_enhancement = getattr(settings,
                                         'INFERENCE_BRIGHTNESS_ENHANCEMENT',
                                         False)
    brightness_factor = getattr(settings, 'INFERENCE_BRIGHTNESS_FACTOR', 1.3)
    contrast_factor = getattr(settings, 'INFERENCE_CONTRAST_FACTOR', 1.1)

    # TTA变换组合
    tta_transforms = []

    # 1. 原始图像（包含亮度增强）
    original_transforms: List[A.BasicTransform] = [
        A.Resize(img_height, img_width, interpolation=cv2.INTER_LINEAR)
    ]
    if use_brightness_enhancement:
        original_transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=(brightness_factor - 1.0,
                                  brightness_factor - 1.0),
                contrast_limit=(contrast_factor - 1.0, contrast_factor - 1.0),
                p=1.0))
    tta_transforms.append(("original", A.Compose(original_transforms)))

    # 2. 水平翻转
    if settings.TTA_FLIPS and True in settings.TTA_FLIPS:
        flip_transforms: List[A.BasicTransform] = [
            A.Resize(img_height, img_width, interpolation=cv2.INTER_LINEAR)
        ]
        if use_brightness_enhancement:
            flip_transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=(brightness_factor - 1.0,
                                      brightness_factor - 1.0),
                    contrast_limit=(contrast_factor - 1.0,
                                    contrast_factor - 1.0),
                    p=1.0))
        flip_transforms.append(A.HorizontalFlip(p=1.0))
        tta_transforms.append(("horizontal_flip", A.Compose(flip_transforms)))

    # 3. 不同尺度
    if hasattr(settings, 'TTA_SCALES') and settings.TTA_SCALES:
        for scale in settings.TTA_SCALES:
            if scale != 1.0:  # 避免重复原始尺度
                scale_transforms: List[A.BasicTransform] = [
                    A.Resize(int(img_height * scale),
                             int(img_width * scale),
                             interpolation=cv2.INTER_LINEAR),
                    A.Resize(img_height,
                             img_width,
                             interpolation=cv2.INTER_LINEAR)
                ]
                if use_brightness_enhancement:
                    scale_transforms.append(
                        A.RandomBrightnessContrast(
                            brightness_limit=(brightness_factor - 1.0,
                                              brightness_factor - 1.0),
                            contrast_limit=(contrast_factor - 1.0,
                                            contrast_factor - 1.0),
                            p=1.0))
                tta_transforms.append(
                    (f"scale_{scale}", A.Compose(scale_transforms)))

    # 4. 轻微旋转
    rotate_transforms_5: List[A.BasicTransform] = [
        A.Resize(img_height, img_width, interpolation=cv2.INTER_LINEAR)
    ]
    if use_brightness_enhancement:
        rotate_transforms_5.append(
            A.RandomBrightnessContrast(
                brightness_limit=(brightness_factor - 1.0,
                                  brightness_factor - 1.0),
                contrast_limit=(contrast_factor - 1.0, contrast_factor - 1.0),
                p=1.0))
    rotate_transforms_5.append(
        A.Rotate(limit=5, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0))
    tta_transforms.append(("rotate_5", A.Compose(rotate_transforms_5)))

    rotate_transforms_neg5: List[A.BasicTransform] = [
        A.Resize(img_height, img_width, interpolation=cv2.INTER_LINEAR)
    ]
    if use_brightness_enhancement:
        rotate_transforms_neg5.append(
            A.RandomBrightnessContrast(
                brightness_limit=(brightness_factor - 1.0,
                                  brightness_factor - 1.0),
                contrast_limit=(contrast_factor - 1.0, contrast_factor - 1.0),
                p=1.0))
    rotate_transforms_neg5.append(
        A.Rotate(limit=(-5, -5),
                 p=1.0,
                 border_mode=cv2.BORDER_CONSTANT,
                 value=0))
    tta_transforms.append(("rotate_-5", A.Compose(rotate_transforms_neg5)))

    # 5. 额外的亮度调整变换
    if use_brightness_enhancement:
        # 更强的亮度增强
        bright_up_transforms: List[A.BasicTransform] = [
            A.Resize(img_height, img_width, interpolation=cv2.INTER_LINEAR),
            A.RandomBrightnessContrast(
                brightness_limit=(brightness_factor + 0.1 - 1.0,
                                  brightness_factor + 0.1 - 1.0),
                contrast_limit=(contrast_factor - 1.0, contrast_factor - 1.0),
                p=1.0)
        ]
        tta_transforms.append(
            ("brightness_enhanced", A.Compose(bright_up_transforms)))
    else:
        # 如果没有启用基础亮度增强，仍然添加轻微的亮度调整
        tta_transforms.append(
            ("brightness_up",
             A.Compose([
                 A.Resize(img_height,
                          img_width,
                          interpolation=cv2.INTER_LINEAR),
                 A.RandomBrightnessContrast(brightness_limit=0.1,
                                            contrast_limit=0,
                                            p=1.0),
             ])))

        tta_transforms.append(
            ("brightness_down",
             A.Compose([
                 A.Resize(img_height,
                          img_width,
                          interpolation=cv2.INTER_LINEAR),
                 A.RandomBrightnessContrast(brightness_limit=-0.1,
                                            contrast_limit=0,
                                            p=1.0),
             ])))

    # 将PyTorch变换用于最终的归一化
    pytorch_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)])

    return tta_transforms, pytorch_transform


def apply_tta_prediction(model, image, tta_transforms, pytorch_transform,
                         device):
    """
    使用测试时数据增强进行预测，并融合多个结果。
    
    Args:
        model: 训练好的模型
        image: PIL图像
        tta_transforms: TTA变换列表
        pytorch_transform: PyTorch变换（用于归一化）
        device: 设备
    
    Returns:
        融合后的预测结果
    """
    predictions = []
    transform_names = []

    # 将PIL图像转换为numpy数组
    image_np = np.array(image)

    print(f"应用 {len(tta_transforms)} 种TTA变换进行预测...")

    for transform_name, transform in tta_transforms:
        try:
            # 应用Albumentations变换
            transformed = transform(image=image_np)
            transformed_image = transformed['image']

            # 转换为PIL图像并应用PyTorch变换
            transformed_pil = Image.fromarray(transformed_image)
            tensor = pytorch_transform(transformed_pil).unsqueeze(0).to(device)

            # 预测
            with torch.no_grad():
                pred_logits = model(tensor)
                pred_probs = torch.sigmoid(pred_logits)

                # 如果需要，对某些变换进行逆变换（例如水平翻转）
                if transform_name == "horizontal_flip":
                    pred_probs = torch.flip(pred_probs, dims=[3])  # 水平翻转预测结果

                predictions.append(pred_probs.cpu())
                transform_names.append(transform_name)

        except Exception as e:
            print(f"警告: TTA变换 '{transform_name}' 失败: {e}")
            continue

    if not predictions:
        raise RuntimeError("所有TTA变换都失败了")

    # 融合预测结果 - 使用平均值
    print(f"融合 {len(predictions)} 个TTA预测结果...")
    fused_prediction = torch.stack(predictions, dim=0).mean(dim=0)

    print(f"TTA变换列表: {', '.join(transform_names)}")

    return fused_prediction


def load_model(model_path, device):
    """加载预训练的模型。"""
    print(f"正在初始化模型 ({settings.MODEL_NAME})...")
    model = CornerMaskModel()
    model.to(device)

    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    try:
        print(f"正在加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # 打印检查点信息
        if isinstance(checkpoint, dict):
            print(f"检查点包含的键: {list(checkpoint.keys())}")
            if 'epoch' in checkpoint:
                print(f"模型训练轮数: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                if 'val_mask_iou' in metrics:
                    print(f"验证集 IoU: {metrics['val_mask_iou']:.4f}")
                print(f"检查点包含的指标: {list(metrics.keys())}")

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:  # 兼容一些其他保存方式
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"模型权重从 {model_path} 加载成功。")

    except FileNotFoundError:
        print(f"错误: 模型文件 {model_path} 未找到。")
        print(f"请检查 config/settings.py 中的 INFERENCE_MODEL_PATH 设置是否正确。")
        sys.exit(1)
    except Exception as e:
        print(f"加载模型权重时发生错误: {e}")
        print("请确保模型路径正确，且模型已按照 CornerMaskModel 结构保存。")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    model.eval()
    print("模型已设置为评估模式。")
    return model


def predict_and_save(model,
                     image_path,
                     output_dir,
                     device,
                     transform,
                     use_tta=False,
                     tta_transforms=None,
                     pytorch_transform=None):
    """对单张图片进行预测、可视化并保存结果 (支持多物体和TTA)。"""
    try:
        pil_image = Image.open(image_path).convert("RGB")
        original_width, original_height = pil_image.size
    except FileNotFoundError:
        print(f"错误：图片文件 {image_path} 未找到。")
        return
    except Exception as e:
        print(f"加载图片 {image_path} 时出错: {e}")
        return

    # 检查是否存在对应的.npy mask文件
    # 处理命名规则：1_Color.png -> 1.npy
    image_basename = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_basename)[0]

    # 提取数字部分（假设格式为 数字_Color）
    if '_Color' in image_name_without_ext:
        number_part = image_name_without_ext.split('_Color')[0]
        npy_filename = f"{number_part}.npy"
    else:
        # 如果不包含_Color，则使用原始文件名
        npy_filename = f"{image_name_without_ext}.npy"

    npy_mask_path = os.path.join(os.path.dirname(image_path), npy_filename)
    true_mask_np = None
    iou_value = None

    if os.path.exists(npy_mask_path):
        try:
            true_mask_np = np.load(npy_mask_path)
            print(f"找到对应的真实mask文件: {npy_mask_path}")
        except Exception as e:
            print(f"加载真实mask文件 {npy_mask_path} 时出错: {e}")
            true_mask_np = None
    else:
        print(f"未找到对应的真实mask文件: {npy_mask_path}")  # 添加调试信息

    # 保存原始图像用于可视化
    original_pil_image = pil_image.copy()

    # 检查是否需要进行亮度增强（仅对标准推理）
    use_brightness_enhancement = getattr(settings,
                                         'INFERENCE_BRIGHTNESS_ENHANCEMENT',
                                         False)
    if use_brightness_enhancement and not use_tta:
        brightness_factor = getattr(settings, 'INFERENCE_BRIGHTNESS_FACTOR',
                                    1.3)
        contrast_factor = getattr(settings, 'INFERENCE_CONTRAST_FACTOR', 1.1)
        gamma = getattr(settings, 'INFERENCE_GAMMA_CORRECTION', 0.8)

        print(
            f"正在对图像 {os.path.basename(image_path)} 进行亮度增强 (亮度: {brightness_factor}, 对比度: {contrast_factor}, 伽马: {gamma})..."
        )
        pil_image = enhance_image_brightness(pil_image, brightness_factor,
                                             contrast_factor, gamma)

    # 选择预测方式
    if use_tta and tta_transforms is not None:
        inference_method = "TTA (包含亮度增强)" if use_brightness_enhancement else "TTA"
        print(
            f"正在使用{inference_method}对图像 {os.path.basename(image_path)} 进行预测..."
        )
        # TTA中亮度增强已在变换列表中处理，使用原始图像
        pred_mask_probs = apply_tta_prediction(model, original_pil_image,
                                               tta_transforms,
                                               pytorch_transform, device)
    else:
        inference_method = "标准推理 (包含亮度增强)" if use_brightness_enhancement else "标准推理"
        print(
            f"正在使用{inference_method}对图像 {os.path.basename(image_path)} 进行预测..."
        )
        img_tensor_transformed = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask_logits = model(img_tensor_transformed)
            pred_mask_probs = torch.sigmoid(pred_mask_logits)

    # 获取mask平滑参数
    smooth_params = get_mask_smoothing_params()

    # 应用mask平滑处理（减少锯齿状边缘）
    if smooth_params['enabled']:
        print(
            f"正在应用mask平滑处理 (类型: {getattr(settings, 'MASK_SMOOTH_TYPE', 'gentle')})..."
        )
        pred_mask_probs = smooth_mask_morphology(
            pred_mask_probs,
            gaussian_sigma=smooth_params['gaussian_sigma'],
            morph_kernel_size=smooth_params['morph_kernel_size'],
            median_kernel_size=smooth_params['median_kernel_size'],
            enable_gaussian=smooth_params['enable_gaussian'],
            enable_morphology=smooth_params['enable_morphology'],
            enable_median=smooth_params['enable_median'])
        print("Mask平滑处理完成。")

    # 使用 mask_to_longest_edge_corners 进行多物体检测
    # 返回: corners_tensor [B, max_obj, 4], num_objects_tensor [B]
    derived_corners_model_space_batch, num_objects_batch = mask_to_longest_edge_corners(
        pred_mask_probs,
        original_image_width=settings.
        IMAGE_WIDTH,  # mask_to_longest_edge_corners 期望模型空间的尺寸
        original_image_height=settings.IMAGE_HEIGHT,
        threshold=0.5,  # 可以考虑加入 settings.py
        max_objects=settings.INFERENCE_MAX_OBJECTS,
        min_contour_area=settings.INFERENCE_MIN_CONTOUR_AREA)

    # 我们处理的是单张图片 (batch size = 1 for inference script)
    derived_corners_model_space_np = derived_corners_model_space_batch.squeeze(
        0).cpu().numpy()  # [max_obj, 4]
    num_objects_detected = num_objects_batch.squeeze(0).item()  # scalar

    inference_method = "TTA" if use_tta else "标准"
    print(
        f"使用{inference_method}推理在图像 {os.path.basename(image_path)} 中检测到 {num_objects_detected} 个物体。"
    )

    # 确保转换为numpy数组并处理数据类型
    if isinstance(pred_mask_probs, torch.Tensor):
        pred_mask_model_space_np = pred_mask_probs.squeeze().cpu().numpy()
    else:
        pred_mask_model_space_np = pred_mask_probs.squeeze()

    # 确保数据类型正确
    pred_mask_model_space_np = np.asarray(pred_mask_model_space_np,
                                          dtype=np.float32)

    binary_mask_original_scale = (cv2.resize(
        pred_mask_model_space_np, (original_width, original_height),
        interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8) * 255

    # --- 计算IoU（如果存在真实mask） ---
    if true_mask_np is not None:
        try:
            # 将真实mask调整为与预测mask相同的尺寸
            if true_mask_np.shape != (original_height, original_width):
                true_mask_resized = cv2.resize(true_mask_np.astype(
                    np.float32), (original_width, original_height),
                                               interpolation=cv2.INTER_NEAREST)
            else:
                true_mask_resized = true_mask_np.astype(np.float32)

            # 确保真实mask是二值的 (0或1)
            true_mask_binary = (true_mask_resized > 0.5).astype(np.float32)

            # 将预测mask转换为tensor格式以使用IoU函数
            pred_mask_tensor = torch.from_numpy(
                (binary_mask_original_scale / 255.0).astype(np.float32))
            pred_mask_tensor = pred_mask_tensor.unsqueeze(0).unsqueeze(
                0)  # [1, 1, H, W]

            true_mask_tensor = torch.from_numpy(true_mask_binary)
            true_mask_tensor = true_mask_tensor.unsqueeze(0).unsqueeze(
                0)  # [1, 1, H, W]

            # 计算IoU
            iou_tensor = intersection_over_union(pred_mask_tensor,
                                                 true_mask_tensor,
                                                 threshold=0.5)
            iou_value = iou_tensor.item()

            print(f"预测mask与真实mask的IoU: {iou_value:.4f}")

        except Exception as e:
            print(f"计算IoU时出错: {e}")
            iou_value = None

    # --- 可视化 ---
    fig, axes = plt.subplots(1,
                             1,
                             figsize=(10,
                                      10 * original_height / original_width))
    # 如果只有一个子图，axes 是单个对象而不是数组
    ax = axes if hasattr(axes, 'imshow') else axes

    # 使用原始图像进行可视化，而不是亮度增强后的图像
    ax.imshow(original_pil_image)

    mask_overlay_rgba = np.zeros((original_height, original_width, 4),
                                 dtype=np.float32)
    mask_overlay_rgba[binary_mask_original_scale == 255, 0] = 1.0
    mask_overlay_rgba[binary_mask_original_scale == 255, 3] = 0.4
    ax.imshow(mask_overlay_rgba)

    all_scaled_corners_for_file = []
    colors = ['yellow', 'cyan', 'magenta', 'lime', 'orange']  # 不同物体的颜色

    if num_objects_detected > 0:
        x_scale = original_width / settings.IMAGE_WIDTH
        y_scale = original_height / settings.IMAGE_HEIGHT

        for i in range(int(num_objects_detected)):
            obj_corners_model = derived_corners_model_space_np[i]  # [4]
            if not np.isnan(obj_corners_model).any():
                scaled_corners = [
                    obj_corners_model[0] * x_scale,
                    obj_corners_model[1] * y_scale,
                    obj_corners_model[2] * x_scale,
                    obj_corners_model[3] * y_scale
                ]
                all_scaled_corners_for_file.append(scaled_corners)

                color = colors[i % len(colors)]
                ax.plot(
                    [scaled_corners[0], scaled_corners[2]],
                    [scaled_corners[1], scaled_corners[3]],
                    'o-',  # Separated marker and linestyle
                    color=color,  # Explicitly set color
                    linewidth=2,
                    markersize=8,
                    markerfacecolor=
                    color,  # Use the object's color for marker face
                    markeredgecolor='black')

    # 移除所有UI元素，只保留纯净的图像
    ax.axis('off')

    # 文件名添加TTA标识
    filename_prefix = "tta_vis" if use_tta else "vis"
    vis_filename = os.path.join(
        output_dir, f"{filename_prefix}_{os.path.basename(image_path)}")
    plt.savefig(vis_filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"可视化结果已保存到: {vis_filename}")

    # --- 保存 anno_mask 和 corner ---
    mask_prefix = "tta_mask" if use_tta else "mask"
    mask_save_path = os.path.join(
        output_dir,
        f"{mask_prefix}_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    )
    cv2.imwrite(mask_save_path, binary_mask_original_scale)
    print(f"预测掩码 (anno_mask) 已保存到: {mask_save_path}")

    corner_prefix = "tta_corners" if use_tta else "corners"
    corner_save_path = os.path.join(
        output_dir,
        f"{corner_prefix}_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
    )
    with open(corner_save_path, 'w') as f:
        if all_scaled_corners_for_file:
            for corners in all_scaled_corners_for_file:
                f.write(
                    f"{corners[0]},{corners[1]},{corners[2]},{corners[3]}\n")
            print(
                f"预测的 {len(all_scaled_corners_for_file)} 个物体的角点已保存到: {corner_save_path}"
            )
        else:
            f.write("NaN,NaN,NaN,NaN\n")  # 表示未检测到物体或有效角点
            print(f"未检测到有效角点，已记录到: {corner_save_path}")

    # 如果计算了IoU，保存IoU结果到文件
    if iou_value is not None:
        iou_prefix = "tta_iou" if use_tta else "iou"
        iou_save_path = os.path.join(
            output_dir,
            f"{iou_prefix}_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
        )
        with open(iou_save_path, 'w') as f:
            f.write(f"{iou_value:.6f}\n")
        print(f"IoU结果已保存到: {iou_save_path}")

    return iou_value  # 返回IoU值供主函数使用


def print_model_config():
    """打印当前模型配置信息"""
    print("=" * 60)
    print("模型推理配置")
    print("=" * 60)
    print(f"模型名称: {settings.MODEL_NAME}")
    print(f"图像尺寸: {settings.IMAGE_HEIGHT} x {settings.IMAGE_WIDTH}")
    print(f"预训练权重: {settings.PRETRAINED_BACKBONE}")
    print(f"归一化均值: {settings.NORMALIZE_MEAN}")
    print(f"归一化标准差: {settings.NORMALIZE_STD}")
    print(f"批量大小: {settings.BATCH_SIZE}")
    print(f"最大检测物体数: {settings.INFERENCE_MAX_OBJECTS}")
    print(f"最小轮廓面积: {settings.INFERENCE_MIN_CONTOUR_AREA}")
    print(f"损失函数: SOTA Tversky Loss (α=0.7, β=0.3)")

    # 亮度增强配置
    use_brightness_enhancement = getattr(settings,
                                         'INFERENCE_BRIGHTNESS_ENHANCEMENT',
                                         False)
    print(f"亮度增强: {'启用' if use_brightness_enhancement else '禁用'}")
    if use_brightness_enhancement:
        brightness_factor = getattr(settings, 'INFERENCE_BRIGHTNESS_FACTOR',
                                    1.3)
        contrast_factor = getattr(settings, 'INFERENCE_CONTRAST_FACTOR', 1.1)
        gamma_correction = getattr(settings, 'INFERENCE_GAMMA_CORRECTION', 0.8)
        print(f"  - 亮度因子: {brightness_factor} (>1.0增亮)")
        print(f"  - 对比度因子: {contrast_factor} (>1.0增强对比度)")
        print(f"  - 伽马校正: {gamma_correction} (<1.0增亮阴影)")

    # TTA配置
    use_tta = getattr(settings, 'USE_TEST_TIME_AUGMENTATION', False)
    print(f"测试时数据增强 (TTA): {'启用' if use_tta else '禁用'}")
    if use_tta:
        tta_scales = getattr(settings, 'TTA_SCALES', [1.0])
        tta_flips = getattr(settings, 'TTA_FLIPS', [False])
        print(f"  - TTA 尺度: {tta_scales}")
        print(f"  - TTA 翻转: {tta_flips}")
        if use_brightness_enhancement:
            print(f"  - TTA 自动应用亮度增强")

    # Mask平滑配置
    smooth_params = get_mask_smoothing_params()
    print(f"Mask边缘平滑: {'启用' if smooth_params['enabled'] else '禁用'}")
    if smooth_params['enabled']:
        smooth_type = getattr(settings, 'MASK_SMOOTH_TYPE', 'gentle')
        print(f"  - 平滑类型: {smooth_type}")
        print(
            f"  - 高斯模糊: {'启用' if smooth_params['enable_gaussian'] else '禁用'}")
        if smooth_params['enable_gaussian']:
            print(f"    * 标准差: {smooth_params['gaussian_sigma']}")
        print(
            f"  - 形态学操作: {'启用' if smooth_params['enable_morphology'] else '禁用'}"
        )
        if smooth_params['enable_morphology']:
            print(f"    * 核大小: {smooth_params['morph_kernel_size']}")
        print(f"  - 中值滤波: {'启用' if smooth_params['enable_median'] else '禁用'}")
        if smooth_params['enable_median']:
            print(f"    * 核大小: {smooth_params['median_kernel_size']}")

    print("=" * 60)


def get_mask_smoothing_params():
    """
    根据配置文件设置获取mask平滑参数。
    
    Returns:
        dict: 包含平滑参数的字典
    """
    if not getattr(settings, 'MASK_SMOOTH_EDGES', False):
        return {
            'enabled': False,
            'gaussian_sigma': 0,
            'morph_kernel_size': 0,
            'median_kernel_size': 0,
            'enable_gaussian': False,
            'enable_morphology': False,
            'enable_median': False
        }

    # 获取平滑类型
    smooth_type = getattr(settings, 'MASK_SMOOTH_TYPE', 'gentle')

    # 如果有预设配置，使用预设
    if hasattr(settings, 'MASK_SMOOTH_PRESETS'
               ) and smooth_type in settings.MASK_SMOOTH_PRESETS:
        preset = settings.MASK_SMOOTH_PRESETS[smooth_type]
        return {
            'enabled': True,
            'gaussian_sigma': preset['gaussian_sigma'],
            'morph_kernel_size': preset['morph_kernel_size'],
            'median_kernel_size': preset['median_kernel_size'],
            'enable_gaussian': preset['enable_gaussian'],
            'enable_morphology': preset['enable_morphology'],
            'enable_median': preset['enable_median']
        }

    # 否则使用独立配置
    return {
        'enabled': True,
        'gaussian_sigma': getattr(settings, 'MASK_GAUSSIAN_SIGMA', 1.0),
        'morph_kernel_size': getattr(settings, 'MASK_MORPH_KERNEL_SIZE', 3),
        'median_kernel_size': getattr(settings, 'MASK_MEDIAN_KERNEL_SIZE', 3),
        'enable_gaussian': getattr(settings, 'MASK_ENABLE_GAUSSIAN_BLUR',
                                   True),
        'enable_morphology': getattr(settings, 'MASK_ENABLE_MORPHOLOGY', True),
        'enable_median': getattr(settings, 'MASK_ENABLE_MEDIAN_FILTER', True)
    }


def enhance_image_brightness(image,
                             brightness_factor=1.3,
                             contrast_factor=1.1,
                             gamma=0.8):
    """
    增强图像亮度以提升阴暗环境下的推理性能
    
    Args:
        image: PIL图像或numpy数组
        brightness_factor: 亮度增强因子 (>1.0增亮, <1.0变暗)
        contrast_factor: 对比度增强因子 (>1.0增强对比度)
        gamma: 伽马校正值 (<1.0增亮阴影, >1.0增暗高光)
    
    Returns:
        增强后的图像 (保持输入格式)
    """
    import cv2

    # 判断输入类型
    is_pil = hasattr(image, 'convert')

    if is_pil:
        # PIL图像转换为numpy数组
        img_array = np.array(image)
    else:
        img_array = image.copy()

    # 确保是uint8格式
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)

    # 1. 亮度调整 (线性变换)
    if brightness_factor != 1.0:
        img_array = cv2.convertScaleAbs(img_array,
                                        alpha=brightness_factor,
                                        beta=0)

    # 2. 对比度调整
    if contrast_factor != 1.0:
        img_array = cv2.convertScaleAbs(img_array,
                                        alpha=contrast_factor,
                                        beta=0)

    # 3. 伽马校正 (非线性变换，对阴暗区域效果更好)
    if gamma != 1.0:
        # 构建伽马校正查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0)**inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        img_array = cv2.LUT(img_array, table)

    # 返回相同格式
    if is_pil:
        return Image.fromarray(img_array)
    else:
        return img_array


def main():
    print_model_config()

    device = settings.DEVICE
    print(f"使用设备: {device}")

    # 从 settings.py 获取配置
    model_path = settings.INFERENCE_MODEL_PATH
    input_dir = settings.INFERENCE_INPUT_DIR
    output_dir = settings.INFERENCE_OUTPUT_DIR
    img_extensions_list = settings.INFERENCE_IMG_EXTENSIONS

    # 检查是否启用TTA和亮度增强
    use_tta = getattr(settings, 'USE_TEST_TIME_AUGMENTATION', False)
    use_brightness_enhancement = getattr(settings,
                                         'INFERENCE_BRIGHTNESS_ENHANCEMENT',
                                         False)

    # 检查路径是否为占位符
    if model_path == "/path/to/your/trained_model.pth" or \
       input_dir == "/path/to/your/input_images":
        print(
            "错误：请在 config/settings.py 文件中配置 INFERENCE_MODEL_PATH 和 INFERENCE_INPUT_DIR 为您的实际路径。"
        )
        print(f"当前模型路径: {model_path}")
        print(f"当前输入路径: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path, device)

    # 准备变换
    inference_transform = get_inference_transforms(settings.IMAGE_HEIGHT,
                                                   settings.IMAGE_WIDTH,
                                                   settings.NORMALIZE_MEAN,
                                                   settings.NORMALIZE_STD)

    tta_transforms = None
    pytorch_transform = None
    if use_tta:
        print("正在初始化TTA变换...")
        tta_transforms, pytorch_transform = get_tta_transforms(
            settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,
            settings.NORMALIZE_MEAN, settings.NORMALIZE_STD)
        print(f"TTA变换数量: {len(tta_transforms)}")

    image_extensions = tuple(f".{ext.strip().lower()}"
                             for ext in img_extensions_list)

    try:
        image_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(image_extensions)
        ]
    except FileNotFoundError:
        print(f"错误: 输入文件夹 {input_dir} 未找到。")
        print(f"请检查 config/settings.py 中的 INFERENCE_INPUT_DIR 设置是否正确。")
        sys.exit(1)
    except Exception as e:
        print(f"读取输入文件夹 {input_dir} 时发生错误: {e}")
        sys.exit(1)

    if not image_files:
        print(
            f"在 {input_dir} 中未找到指定的图像文件 (扩展名: {', '.join(img_extensions_list)})。"
        )
        return

    # 构建推理方法描述
    brightness_status = " + 亮度增强" if use_brightness_enhancement else ""
    inference_method = f"TTA{brightness_status}" if use_tta else f"标准推理{brightness_status}"
    print(f"找到 {len(image_files)} 张图像进行{inference_method}处理...")

    # 用于统计IoU结果
    iou_values = []
    images_with_ground_truth = 0

    for i, image_name in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_name)
        print(f"\n[{i}/{len(image_files)}] 正在处理图像: {image_path}")
        iou_value = predict_and_save(model,
                                     image_path,
                                     output_dir,
                                     device,
                                     inference_transform,
                                     use_tta=use_tta,
                                     tta_transforms=tta_transforms,
                                     pytorch_transform=pytorch_transform)

        # 收集IoU统计信息
        if iou_value is not None:
            iou_values.append(iou_value)
            images_with_ground_truth += 1

    print("\n" + "=" * 60)
    print(f"模型推理完成")
    print("=" * 60)
    print(f"处理图像数量: {len(image_files)}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {settings.MODEL_NAME}")
    print(f"推理方式: {inference_method}")
    print(f"损失函数: SOTA Tversky Loss")

    # 显示IoU统计信息
    if images_with_ground_truth > 0:
        mean_iou = sum(iou_values) / len(iou_values)
        min_iou = min(iou_values)
        max_iou = max(iou_values)
        print(f"IoU评估结果:")
        print(
            f"  - 包含真实标注的图像数量: {images_with_ground_truth}/{len(image_files)}")
        print(f"  - 平均IoU: {mean_iou:.4f}")
        print(f"  - 最小IoU: {min_iou:.4f}")
        print(f"  - 最大IoU: {max_iou:.4f}")
    else:
        print(f"IoU评估: 未找到对应的.npy真实标注文件")

    # 显示详细的处理设置
    if use_brightness_enhancement:
        brightness_factor = getattr(settings, 'INFERENCE_BRIGHTNESS_FACTOR',
                                    1.3)
        contrast_factor = getattr(settings, 'INFERENCE_CONTRAST_FACTOR', 1.1)
        gamma_correction = getattr(settings, 'INFERENCE_GAMMA_CORRECTION', 0.8)
        print(
            f"亮度增强设置: 亮度={brightness_factor}, 对比度={contrast_factor}, 伽马={gamma_correction}"
        )

    # 显示mask平滑设置
    smooth_params = get_mask_smoothing_params()
    if smooth_params['enabled']:
        smooth_type = getattr(settings, 'MASK_SMOOTH_TYPE', 'gentle')
        print(f"Mask平滑设置: 类型={smooth_type}")
        enabled_ops = []
        if smooth_params['enable_gaussian']:
            enabled_ops.append(f"高斯模糊(σ={smooth_params['gaussian_sigma']})")
        if smooth_params['enable_morphology']:
            enabled_ops.append(f"形态学(核={smooth_params['morph_kernel_size']})")
        if smooth_params['enable_median']:
            enabled_ops.append(
                f"中值滤波(核={smooth_params['median_kernel_size']})")
        if enabled_ops:
            print(f"  应用操作: {', '.join(enabled_ops)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
