import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 定义 CIFAR-10 的数据预处理流程
cifar10_transform = transforms.Compose([
    # 缩放到 32x32
    transforms.Resize((32, 32)),
    # 转换为 Tensor，并缩放到 [0, 1] 之间
    transforms.ToTensor(),
])


class CIFAR10TrainDataset(Dataset):
    """适配 CIFAR 10 训练集的自定义数据集
    """
    def __init__(self, images_dir, labels_csv, transform=cifar10_transform, class_to_idx=None):
        """
        Args:
            images_dir (str): 存放所有训练图片的文件夹路径。
            labels_csv (str): 存放所有训练图片标签的 CSV 文件路径。
            transform (callable, optional): 数据预处理流程，可选。
            class_to_idx (dict, optional): 类别到索引的映射，可选。
        """
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Assuming the CSV has columns: 'id', 'label'
        img_name = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 1]
        img_path = os.path.join(self.images_dir, f"{img_name}.png")
        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
        if self.transform:
            image = self.transform(image)
        if self.class_to_idx:
            label = self.class_to_idx[label]
        return image, label


# Custom Dataset for Testing
class CIFAR10TestDataset(Dataset):
    def __init__(self, images_dir, transform=cifar10_transform):
        """
        Args:
            images_dir (str): Directory with all the test images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_dir = images_dir
        self.image_names = sorted(os.listdir(images_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
        if self.transform:
            image = self.transform(image)
            
        # 把 img_name 从 xx.png 转为数字 xx
        img_name = int(img_name.split('.')[0])
        return image, img_name  # Returning img_name for identification if needed
