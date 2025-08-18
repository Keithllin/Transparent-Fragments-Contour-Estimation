import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from .transforms import Preprocessing  # 从同级目录的 transforms.py 导入
import sys

sys.path.append("..")
from config import settings


class CornerPointDataset(Dataset):
    """
    用于加载掩码数据集的自定义 Dataset 类。
    训练模式下只加载图像和掩码，不加载角点数据。

    Args:
        mode (str): 数据集模式，'train' 或 'test'。
                      决定是从训练集目录还是测试集目录加载数据。
        transform (callable, optional): 一个可选的转换函数/类，
                                       应用于样本数据。
        use_augmentation (bool, optional): 是否启用数据增强。
        load_corners (bool, optional): 是否加载角点数据，默认False（只在测试时需要）。
    """

    def __init__(self,
                 mode='train',
                 transform=None,
                 use_augmentation=None,
                 load_corners=False):
        super().__init__()
        if mode not in ['train', 'test']:
            raise ValueError("模式必须是 'train' 或 'test'")

        self.mode = mode
        self.load_corners = load_corners  # 控制是否加载角点数据

        # 智能设置数据增强
        if use_augmentation is None:
            # 训练模式默认启用数据增强，测试模式默认禁用
            use_augmentation = (mode == 'train')

        # 如果没有提供transform，则使用默认的Preprocessing
        if transform is None:
            from .transforms import Preprocessing
            self.transform = Preprocessing(mode=mode,
                                           use_augmentation=use_augmentation,
                                           load_corners=self.load_corners)
        else:
            self.transform = transform

        # self.dataset_root_dir = settings.DATASET_ROOT_DIR  # 数据集根目录，例如 /root/autodl-tmp/dataset_cornerpoint3
        # 解析 DATASET_ROOT_DIR，使其相对于项目结构固定，而不是当前工作目录
        # settings.DATASET_ROOT_DIR (例如 "../dataset_cornerpoint3") 预期是相对于 newmodel 目录的
        # __file__ 指向当前文件 (dataset.py) 的路径
        # e.g., /home/visllm/program/KGC/tactile/newmodel/data/dataset.py
        current_file_path = os.path.abspath(__file__)
        # e.g., /home/visllm/program/KGC/tactile/newmodel/data/
        current_dir = os.path.dirname(current_file_path)
        # e.g., /home/visllm/program/KGC/tactile/newmodel/
        newmodel_dir = os.path.dirname(current_dir)
        # 将 settings.DATASET_ROOT_DIR (相对路径) 与 newmodel_dir 拼接，并获取绝对路径
        # e.g., os.path.join("/path/to/newmodel", "../dataset_cornerpoint3") -> /path/to/dataset_cornerpoint3
        resolved_dataset_root_dir = os.path.abspath(
            os.path.join(newmodel_dir, settings.DATASET_ROOT_DIR))
        self.dataset_root_dir = resolved_dataset_root_dir

        # 根据模式（train/test）构建特定数据子目录的路径
        # 例如 /root/autodl-tmp/dataset_cornerpoint3/train/
        self.base_data_path = os.path.join(
            self.dataset_root_dir,
            settings.TRAIN_DIR if mode == 'train' else settings.TEST_DIR)

        # 定义 rgb 图像和掩码 .npy 文件的具体子目录
        self.rgb_dir = os.path.join(self.base_data_path, 'rgb')
        self.anno_mask_dir = os.path.join(self.base_data_path,
                                          'anno_mask')  # 掩码文件在 anno_mask/ 子目录

        # 只在需要时定义角点目录
        if self.load_corners:
            self.corner_dir = os.path.join(self.base_data_path, 'corner')

        # 检查路径是否存在
        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(f"RGB图像目录未找到: {self.rgb_dir}")
        if not os.path.isdir(self.anno_mask_dir):
            raise FileNotFoundError(f"掩码数据目录未找到: {self.anno_mask_dir}")
        if self.load_corners and not os.path.isdir(self.corner_dir):
            raise FileNotFoundError(f"角点数据目录未找到: {self.corner_dir}")

        # 获取所有图像文件名 (例如 '0.png', '1.png', ...)
        # 过滤掉非png文件、隐藏文件，以及包含'aug'的数据增强文件，并排序以保证一致性
        all_image_files = sorted([
            f for f in os.listdir(self.rgb_dir)
            if os.path.isfile(os.path.join(self.rgb_dir, f))
            and f.endswith('.png') and not f.startswith('.')
        ])

        # 只保留有对应掩码文件的图像
        self.image_filenames = []
        skipped_count = 0
        for img_file in all_image_files:
            base_filename = os.path.splitext(img_file)[0]

            # 适应新的掩码文件名规则: 移除 '_Color'
            mask_base_filename = base_filename.replace('_Color', '')

            mask_path = os.path.join(self.anno_mask_dir,
                                     f"{mask_base_filename}.npy")

            if os.path.exists(mask_path):
                self.image_filenames.append(img_file)
            else:
                skipped_count += 1
                print(f"跳过图像 {img_file}：未找到对应的掩码文件 {mask_base_filename}.npy")

        if skipped_count > 0:
            print(f"总共跳过了 {skipped_count} 个没有对应掩码文件的图像")

        if not self.image_filenames:
            print(f"警告: 在目录 {self.rgb_dir} 中没有找到有效的图像-掩码配对。")

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个数据样本。

        Args:
            idx (int): 样本的索引。

        Returns:
            dict: 包含 'image', 'mask' 的字典，如果load_corners=True则还包含'corner'。
                  如果定义了 transform，则返回转换后的样本。
        """
        if idx < 0 or idx >= len(self.image_filenames):
            raise IndexError(
                f"索引 {idx} 超出范围 [0, {len(self.image_filenames)-1}]")

        # 获取基础文件名 (例如 '0')
        base_filename = os.path.splitext(self.image_filenames[idx])[0]

        # 构建各文件的完整路径
        img_path = os.path.join(self.rgb_dir, f"{base_filename}.png")

        # 适应新的掩码文件名规则: 移除 '_Color'
        mask_base_filename = base_filename.replace('_Color', '')
        mask_path = os.path.join(self.anno_mask_dir,
                                 f"{mask_base_filename}.npy")

        # 掩码文件的存在性已在初始化时检查过，这里不再重复检查

        # 加载数据
        try:
            image = Image.open(img_path).convert('RGB')  # 加载RGB图像
            anno_mask = np.load(mask_path)  # 加载标注掩码
        except Exception as e:
            print(f"加载索引 {idx} (图像: {img_path}) 的数据时出错: {e}")
            raise

        # 确保掩码是numpy数组
        if not isinstance(anno_mask, np.ndarray):
            anno_mask = np.array(anno_mask)

        sample = {'image': image, 'mask': anno_mask}

        # 只在需要时加载角点数据
        if self.load_corners:
            corner_path = os.path.join(self.corner_dir, f"{base_filename}.npy")
            if not os.path.exists(corner_path):
                raise FileNotFoundError(
                    f"角点文件未找到: {corner_path} (对应图像: {img_path})")

            try:
                corner_coords = np.load(corner_path)  # 加载角点数据
                if not isinstance(corner_coords, np.ndarray):
                    corner_coords = np.array(corner_coords)
                sample['corner'] = corner_coords
            except Exception as e:
                print(f"加载角点数据时出错: {e}")
                raise

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_raw_sample(self, idx):
        """
        获取原始的、未经预处理的样本数据
        
        Args:
            idx (int): 样本的索引
            
        Returns:
            dict: 包含原始PIL图像、numpy掩码的字典，如果load_corners=True则还包含角点
        """
        if idx < 0 or idx >= len(self.image_filenames):
            raise IndexError(
                f"索引 {idx} 超出范围 [0, {len(self.image_filenames)-1}]")

        # 获取基础文件名 (例如 '0')
        base_filename = os.path.splitext(self.image_filenames[idx])[0]

        # 构建各文件的完整路径
        img_path = os.path.join(self.rgb_dir, f"{base_filename}.png")

        # 适应新的掩码文件名规则: 移除 '_Color'
        mask_base_filename = base_filename.replace('_Color', '')
        mask_path = os.path.join(self.anno_mask_dir,
                                 f"{mask_base_filename}.npy")

        # 掩码文件的存在性已在初始化时检查过，这里不再重复检查

        # 加载数据
        try:
            image = Image.open(img_path).convert('RGB')  # 加载RGB图像
            anno_mask = np.load(mask_path)  # 加载标注掩码
        except Exception as e:
            print(f"加载索引 {idx} (图像: {img_path}) 的数据时出错: {e}")
            raise

        # 确保掩码是numpy数组
        if not isinstance(anno_mask, np.ndarray):
            anno_mask = np.array(anno_mask)

        result = {'image': image, 'mask': anno_mask.astype(np.float32)}

        # 只在需要时加载角点数据
        if self.load_corners:
            corner_path = os.path.join(self.corner_dir, f"{base_filename}.npy")
            if not os.path.exists(corner_path):
                raise FileNotFoundError(
                    f"角点文件未找到: {corner_path} (对应图像: {img_path})")

            try:
                corner_coords = np.load(corner_path)  # 加载角点数据
                if not isinstance(corner_coords, np.ndarray):
                    corner_coords = np.array(corner_coords)
                result['corner'] = corner_coords.astype(np.float32)
            except Exception as e:
                print(f"加载角点数据时出错: {e}")
                raise

        return result


if __name__ == '__main__':
    # 用于测试 CornerPointDataset 类的示例代码
    # 确保 settings.py 文件存在且路径配置正确
    # 确保 transforms.py 文件存在且 Preprocessing 类已定义

    print(f"使用的 settings.DATASET_ROOT_DIR: {settings.DATASET_ROOT_DIR}")

    # 初始化预处理器 (从 transforms.py)
    # 注意：直接运行此文件时，需要确保 Preprocessing 和 settings 能够正确导入
    # 如果在包外运行，可能需要手动添加到 sys.path 或调整导入语句
    try:
        preprocessor = Preprocessing()
    except NameError:  # 如果 Preprocessing 未定义 (通常是导入问题)
        print("错误: Preprocessing 类未定义。请确保 transforms.py 和其依赖项可被正确导入。")
        print("如果直接运行此脚本，请检查PYTHONPATH或项目结构。")
        exit(1)
    except Exception as e:
        print(f"初始化 Preprocessing 时出错: {e}")
        exit(1)

    print("\n尝试加载训练数据集...")
    try:
        train_dataset = CornerPointDataset(mode='train',
                                           transform=preprocessor)
        if len(train_dataset) > 0:
            print(f"成功加载训练数据集，样本数: {len(train_dataset)}")
            # 获取并打印第一个样本的信息
            first_sample = train_dataset[0]
            print("第一个训练样本 (转换后):")
            print(f"  图像张量形状: {first_sample['image'].shape}")
            print(f"  掩码张量形状: {first_sample['mask'].shape}")
            print(f"  掩码张量唯一值: {torch.unique(first_sample['mask'])}")
        else:
            print("训练数据集中没有找到样本。请检查 'rgb' 目录和文件。")
    except FileNotFoundError as e:
        print(f"加载训练数据集时出错 (文件未找到): {e}")
    except Exception as e:
        print(f"加载训练数据集时发生未知错误: {e}")

    print("\n尝试加载测试数据集...")
    try:
        test_dataset = CornerPointDataset(mode='test', transform=preprocessor)
        if len(test_dataset) > 0:
            print(f"成功加载测试数据集，样本数: {len(test_dataset)}")
            # 获取并打印第一个样本的信息
            first_test_sample = test_dataset[0]
            print("第一个测试样本 (转换后):")
            print(f"  图像张量形状: {first_test_sample['image'].shape}")
        else:
            print("测试数据集中没有找到样本。请检查 'rgb' 目录和文件。")
    except FileNotFoundError as e:
        print(f"加载测试数据集时出错 (文件未找到): {e}")
    except Exception as e:
        print(f"加载测试数据集时发生未知错误: {e}")

    print("\nDataset 测试完成。")
