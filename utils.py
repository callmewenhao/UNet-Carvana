import os
import numpy as np
import torch
from PIL import Image

# image data shape: RGB, 1918, 1280


def load_image(image_path, new_size=(160, 240)):
    img = Image.open(image_path)
    img = img.resize(new_size)
    return img


def load_mask(mask_path, new_size=(160, 240)):
    img = Image.open(mask_path)
    img = img.resize(new_size)
    return img


def main():
    image_path = "data/train_images/0cdf5b5d0ce1_01.jpg"
    mask_path = "data/train_masks/0cdf5b5d0ce1_01_mask.gif"
    mask = load_mask(mask_path)


if __name__ == '__main__':
    main()



















