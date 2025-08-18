import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
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
# 移除了 dsntnn 相关导入，因为 HeatmapLoss 被移除
# from dsntnn import js_reg_losses, flat_softmax, average_loss

# Removed: HeatmapLoss class
# class HeatmapLoss(nn.Module):
#     ...


class MaskLoss(nn.Module):
    """
    掩码分割损失函数。使用二元交叉熵损失(BCEWithLogitsLoss)，适合二分类分割任务。
    
    BCEWithLogitsLoss结合了Sigmoid激活和BCE损失，数值上更稳定，是分割任务的常用选择。
    """

    def __init__(self):
        super(MaskLoss, self).__init__()
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred_mask_logits, target_mask):
        """
        计算掩码分割的二元交叉熵损失。
        
        Args:
            pred_mask_logits: 预测的掩码logits，形状为 [batch_size, 1, H, W]
            target_mask: 目标掩码，形状为 [batch_size, 1, H, W]，值为0或1
            
        Returns:
            torch.Tensor: 标量损失值
        """
        return self.bce_logits(pred_mask_logits, target_mask)


class CombinedLoss(nn.Module):
    """
    仅包含掩码损失的总损失函数。
    使用settings.py中定义的权重进行加权 (如果需要，当前 MASK_LOSS_WEIGHT 通常为 1.0)。
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        # self.heatmap_loss_fn = HeatmapLoss() # Removed
        self.mask_loss_fn = MaskLoss()
        # self.heatmap_weight = settings.HEATMAP_LOSS_WEIGHT # Removed
        self.mask_weight = settings.MASK_LOSS_WEIGHT
        # self.use_regularization = settings.USE_HEATMAP_REGULARIZATION # Removed
        # if self.use_regularization: # Removed
        # self.reg_weight = settings.REG_LOSS_WEIGHT # Removed

    def forward(self, pred_mask_logits, target_mask):
        """
        计算掩码损失。
        
        Args:
            pred_mask_logits (torch.Tensor): 预测掩码logits [B, 1, H, W]
            target_mask (torch.Tensor): 目标掩码 [B, 1, H, W]
                            
        Returns:
            dict: 包含总损失和掩码损失的字典
        """
        # heatmap_loss_val, reg_loss_val = self.heatmap_loss_fn(...) # Removed
        mask_loss_val = self.mask_loss_fn(pred_mask_logits, target_mask)

        # total_loss = self.heatmap_weight * heatmap_loss_val + self.mask_weight * mask_loss_val # Old calculation
        total_loss = self.mask_weight * mask_loss_val

        loss_dict = {
            'total': total_loss,
            # 'heatmap_loss': heatmap_loss_val, # Removed
            'mask_loss': mask_loss_val
        }

        # Removed regularization logic
        # if self.use_regularization: ...
        # else: ...
        # loss_dict['reg_loss'] = torch.tensor(0.0, device=total_loss.device) # Removed

        return loss_dict


# 可选的替代损失函数，可根据需要使用
class AlternativeMaskLoss(nn.Module):
    """
    掩码分割的替代损失函数。使用Focal Loss，适合处理类别不平衡的分割任务。
    
    Focal Loss对易分类样本的损失进行降权，关注更具挑战性的样本。
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(AlternativeMaskLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_mask_logits, target_mask):
        """
        计算掩码分割的Focal Loss。
        
        Args:
            pred_mask_logits: 预测的掩码logits，形状为 [batch_size, 1, H, W]
            target_mask: 目标掩码，形状为 [batch_size, 1, H, W]，值为0或1
            
        Returns:
            torch.Tensor: 标量损失值
        """
        return sigmoid_focal_loss(pred_mask_logits,
                                  target_mask,
                                  alpha=self.alpha,
                                  gamma=self.gamma,
                                  reduction='mean')


class AlternativeCornerLoss(nn.Module):
    """
    角点坐标回归的替代损失函数。使用Mean Squared Error (MSE)损失。
    MSE损失对大误差的惩罚更强，适用于对精确预测要求较高的场景。
    """

    def __init__(self):
        super(AlternativeCornerLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred_corners, target_corners):
        """
        计算角点坐标的MSE损失。
        
        Args:
            pred_corners: 预测的角点坐标，形状为 [batch_size, 4]
            target_corners: 目标角点坐标，形状为 [batch_size, 4]
            
        Returns:
            torch.Tensor: 标量损失值
        """
        return self.mse(pred_corners, target_corners)


if __name__ == '__main__':

    class MockSettings:

        def __init__(self):
            # Settings relevant to remaining losses
            self.MASK_LOSS_WEIGHT = 1.0  # Typically 1.0 for mask-only or if other weights are removed
            # Removed heatmap related mock settings
            # self.NUM_KEYPOINTS = 2
            # self.HEATMAP_SIGMA = 5.0
            # self.USE_HEATMAP_REGULARIZATION = True
            # self.HEATMAP_LOSS_WEIGHT = 0.5
            # self.REG_LOSS_WEIGHT = 0.1

    settings = MockSettings()

    B, H, W = 2, 64, 64

    # NUM_KEYPOINTS = settings.NUM_KEYPOINTS # Removed

    # --- 测试 HeatmapLoss --- (Removed)
    # def test_heatmap_loss():
    #     ...

    # --- 测试 MaskLoss ---
    def test_mask_loss():
        print("\n--- Testing MaskLoss ---")
        loss_fn = MaskLoss()
        pred_mask_logits = torch.randn(B, 1, H, W)
        target_mask = torch.randint(0, 2, (B, 1, H, W)).float()
        loss = loss_fn(pred_mask_logits, target_mask)
        print(f"MaskLoss: {loss.item()}")

    # --- 测试 CombinedLoss ---
    def test_combined_loss():
        print("\n--- Testing CombinedLoss (Mask Only) ---")
        loss_fn = CombinedLoss()
        # pred_heatmaps = torch.rand(B, NUM_KEYPOINTS, H, W) # Removed
        # target_heatmaps = torch.rand(B, NUM_KEYPOINTS, H, W) # Removed
        pred_mask_logits = torch.randn(B, 1, H, W)
        target_mask = torch.randint(0, 2, (B, 1, H, W)).float()
        # target_corners_flat = (torch.rand(B, NUM_KEYPOINTS * 2) * 2) - 1 # Removed

        # print(f"Testing CombinedLoss with USE_HEATMAP_REGULARIZATION = {settings.USE_HEATMAP_REGULARIZATION}") # Removed

        losses = loss_fn(pred_mask_logits, target_mask)
        print("CombinedLoss (Mask Only):")
        for k, v in losses.items():
            print(f"  {k}: {v.item()}")

        # Ensure 'total' and 'mask_loss' are present, and 'heatmap_loss' / 'reg_loss' are not
        assert 'total' in losses
        assert 'mask_loss' in losses
        assert 'heatmap_loss' not in losses
        assert 'reg_loss' not in losses
        assert losses['total'].item(
        ) == settings.MASK_LOSS_WEIGHT * losses['mask_loss'].item()

    # --- 测试 AlternativeMaskLoss ---
    def test_alternative_mask_loss():
        print("\n--- Testing AlternativeMaskLoss (Focal Loss) ---")
        loss_fn = AlternativeMaskLoss(alpha=0.25, gamma=2.0)
        pred_mask_logits = torch.randn(B, 1, H, W)
        target_mask = torch.randint(0, 2, (B, 1, H, W)).float()
        loss = loss_fn(pred_mask_logits, target_mask)
        print(f"AlternativeMaskLoss (Focal): {loss.item()}")

    # --- 测试 AlternativeCornerLoss ---
    def test_alternative_corner_loss():
        print("\n--- Testing AlternativeCornerLoss (MSE for corners) ---")
        loss_fn = AlternativeCornerLoss()
        num_coords = 4
        pred_corners = torch.rand(B, num_coords)
        target_corners = torch.rand(B, num_coords)
        loss = loss_fn(pred_corners, target_corners)
        print(f"AlternativeCornerLoss (MSE): {loss.item()}")

    print("Running tests with MockSettings for Losses (Mask Only):")
    print(f"  MASK_LOSS_WEIGHT = {settings.MASK_LOSS_WEIGHT}")

    # test_heatmap_loss() # Removed
    test_mask_loss()
    test_combined_loss()
    test_alternative_mask_loss()
    test_alternative_corner_loss()

    print("\nAll loss tests completed.")
