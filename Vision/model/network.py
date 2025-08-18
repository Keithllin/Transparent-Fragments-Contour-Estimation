import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys
import os
# from dsntnn import flat_softmax # Removed: dsntnn and flat_softmax are for heatmaps

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 如果项目根目录不在sys.path中，则添加它
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import settings


class CornerMaskModel(nn.Module):
    """
    改进的掩码分割模型。
    使用timm库中的backbone，采用UNet风格的解码器，增强特征表达能力。
    """

    def __init__(self):
        super(CornerMaskModel, self).__init__()

        # 创建backbone
        self.backbone = timm.create_model(
            settings.MODEL_NAME,
            pretrained=settings.PRETRAINED_BACKBONE,
            features_only=True,
            pretrained_cfg_overlay=dict(file=settings.PRETRAINED_MODEL_PATH)
            if settings.PRETRAINED_MODEL_PATH else None)

        self.backbone_channels = self.backbone.feature_info.channels()

        # 检查是否为Swin Transformer模型
        self.is_swin_model = 'swin' in settings.MODEL_NAME.lower()

        # 改进的解码器 - 使用多层特征
        self.decoder = self._build_improved_decoder()

    def _build_improved_decoder(self):
        """构建改进的解码器，使用多层特征"""
        # 使用最后3个特征层进行融合
        channels = self.backbone_channels[-3:]  # 例如: [256, 512, 1024]

        # 特征融合层 - 统一输出通道数为256
        self.lateral_conv3 = nn.Conv2d(channels[-1], 256,
                                       kernel_size=1)  # 1024->256
        self.lateral_conv2 = nn.Conv2d(channels[-2], 256,
                                       kernel_size=1)  # 512->256
        self.lateral_conv1 = nn.Conv2d(channels[-3], 256,
                                       kernel_size=1)  # 256->256

        # 上采样和特征融合 - 保持通道数一致
        self.upsample3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        self.upsample2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 保持256通道
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        self.upsample1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 最后再降维
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1))

        return nn.ModuleDict({
            'lateral_conv3': self.lateral_conv3,
            'lateral_conv2': self.lateral_conv2,
            'lateral_conv1': self.lateral_conv1,
            'upsample3': self.upsample3,
            'upsample2': self.upsample2,
            'upsample1': self.upsample1,
            'final_conv': self.final_conv
        })

    def _convert_swin_features(self, features):
        """将Swin Transformer的NHWC格式特征转换为NCHW格式"""
        converted_features = []
        for feat in features:
            if len(feat.shape) == 4:
                # 检查是否为NHWC格式
                if feat.shape[-1] > feat.shape[1] and feat.shape[
                        -1] > feat.shape[2]:
                    # 从NHWC转换为NCHW
                    feat = feat.permute(0, 3, 1, 2)
            converted_features.append(feat)
        return converted_features

    def forward(self, x):
        """模型前向传播"""
        input_h, input_w = x.shape[2:]

        # 提取特征
        features = self.backbone(x)

        # 如果是Swin模型，转换特征格式
        if self.is_swin_model:
            features = self._convert_swin_features(features)

        # 使用最后3个特征层
        f1, f2, f3 = features[-3], features[-2], features[-1]

        # 特征融合（FPN风格）
        # 最深层特征
        p3 = self.decoder['lateral_conv3'](f3)
        p3_up = self.decoder['upsample3'](p3)

        # 中层特征
        p2 = self.decoder['lateral_conv2'](f2)
        # 将上采样后的深层特征与中层特征相加
        if p3_up.shape[2:] != p2.shape[2:]:
            p3_up = F.interpolate(p3_up,
                                  size=p2.shape[2:],
                                  mode='bilinear',
                                  align_corners=False)
        p2 = p2 + p3_up
        p2_up = self.decoder['upsample2'](p2)

        # 浅层特征
        p1 = self.decoder['lateral_conv1'](f1)
        # 将上采样后的中层特征与浅层特征相加
        if p2_up.shape[2:] != p1.shape[2:]:
            p2_up = F.interpolate(p2_up,
                                  size=p1.shape[2:],
                                  mode='bilinear',
                                  align_corners=False)
        p1 = p1 + p2_up

        # 最终上采样和输出
        final_features = self.decoder['upsample1'](p1)
        pred_mask_logits = self.decoder['final_conv'](final_features)

        # 确保输出尺寸匹配输入
        if pred_mask_logits.shape[2:] != (input_h, input_w):
            pred_mask_logits = F.interpolate(pred_mask_logits,
                                             size=(input_h, input_w),
                                             mode='bilinear',
                                             align_corners=False)

        return pred_mask_logits


def test_model():
    """测试改进模型"""
    original_model_name = settings.MODEL_NAME
    original_pretrained = settings.PRETRAINED_BACKBONE
    original_pretrained_path = settings.PRETRAINED_MODEL_PATH

    # 为测试使用轻量模型
    if 'swin' in settings.MODEL_NAME.lower():
        settings.MODEL_NAME = 'swin_tiny_patch4_window7_224'
        settings.PRETRAINED_BACKBONE = False
    else:
        settings.MODEL_NAME = 'swin_tiny_patch4_window7_224'
        settings.PRETRAINED_BACKBONE = False

    settings.PRETRAINED_MODEL_PATH = None

    print(f"--- 改进模型测试 ---")
    print(f"测试模型: {settings.MODEL_NAME}")
    print(f"目标尺寸: {settings.IMAGE_HEIGHT}x{settings.IMAGE_WIDTH}")

    model = CornerMaskModel()
    model.eval()

    batch_size = 2
    test_img_height = settings.IMAGE_HEIGHT or 224
    test_img_width = settings.IMAGE_WIDTH or 224

    dummy_input = torch.rand(batch_size, 3, test_img_height, test_img_width)

    try:
        pred_mask_logits = model(dummy_input)

        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)

    except Exception as e:
        print(f"模型前向传播错误: {e}")
        import traceback
        traceback.print_exc()
        # 恢复设置
        settings.MODEL_NAME = original_model_name
        settings.PRETRAINED_BACKBONE = original_pretrained
        settings.PRETRAINED_MODEL_PATH = original_pretrained_path
        return False

    print(f"--- 改进模型测试结果 ---")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {pred_mask_logits.shape}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    expected_shape = (batch_size, 1, test_img_height, test_img_width)
    print(f"期望形状: {expected_shape}")

    success = pred_mask_logits.shape == expected_shape
    print(f"形状测试: {'通过' if success else '失败'}")

    # 恢复设置
    settings.MODEL_NAME = original_model_name
    settings.PRETRAINED_BACKBONE = original_pretrained
    settings.PRETRAINED_MODEL_PATH = original_pretrained_path

    return success


if __name__ == "__main__":
    print("运行改进模型测试...")
    success = test_model()
    print(f"\n改进模型测试结果: {'成功' if success else '失败'}")
