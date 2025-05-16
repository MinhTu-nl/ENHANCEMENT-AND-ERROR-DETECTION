import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class LoLDataset(Dataset):
    def __init__(self, low_light_dir, high_light_dir, transform=None, target_size=(256, 256)):
        self.low_light_dir = low_light_dir
        self.high_light_dir = high_light_dir
        self.transform = transform
        self.target_size = target_size
        self.low_light_images = sorted(os.listdir(low_light_dir))
        self.high_light_images = sorted(os.listdir(high_light_dir))

    def __len__(self):
        return len(self.low_light_images)

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_light_dir, self.low_light_images[idx])
        high_img_path = os.path.join(self.high_light_dir, self.high_light_images[idx])

        low_img = cv2.imread(low_img_path)
        high_img = cv2.imread(high_img_path)

        if low_img is None or high_img is None:
            raise ValueError(f"Failed to load image at {low_img_path} or {high_img_path}")

        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

        # Resize images to target size
        low_img = cv2.resize(low_img, self.target_size, interpolation=cv2.INTER_AREA)
        high_img = cv2.resize(high_img, self.target_size, interpolation=cv2.INTER_AREA)

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img

def get_data_loaders(low_light_dir, high_light_dir, batch_size=16, target_size=(256, 256)):
    transform = lambda x: torch.from_numpy(x.transpose(2, 0, 1)).float() / 255.0
    dataset = LoLDataset(low_light_dir, high_light_dir, transform, target_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def save_image(tensor, path):
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255.0  # Thêm .detach()
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)