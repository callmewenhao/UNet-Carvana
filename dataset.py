import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import load_image, load_mask


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, new_size=(240, 160)):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.new_size = new_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            )
        ])
        # 得到图片名字列表
        self.image_names = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        mask_name = image_name.replace('.jpg', '_mask.gif')
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        # 拿相应的图片
        img = load_image(image_path, self.new_size)
        img = self.transform(img)
        label = load_mask(mask_path, self.new_size)
        label = torch.tensor(np.array(label))
        return img, label


def main():
    image_dir = "data/train_images"
    mask_dir = "data/train_masks"
    dataset = CarvanaDataset(image_dir, mask_dir, (240, 160))
    image = dataset[0][0]
    mask = dataset[0][1]
    print(image.shape, mask.shape)


if __name__ == "__main__":
    main()


