import torch
import numpy as np
import torchvision.transforms as transforms
from model import UNET
from utils import load_image, load_mask
import matplotlib.pyplot as plt


# path & device
weight_path = "save_weights/unet_carvana.pth"
test_image_path = "data/val_images/fff9b3a5373f_16.jpg"
test_seg_path = "data/val_masks/fff9b3a5373f_16_mask.gif"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
    )
])

# data
image = load_image(test_image_path, (240, 160))
data = transform(image).unsqueeze(dim=0).to(device)
seg = load_mask(test_seg_path, (240, 160)).convert('L')
seg = np.array(seg)

# model
model = UNET(3, 2).to(device)
model.load_state_dict(torch.load(weight_path))  # 加载参数

# 拿一张测试图片 预测
model.eval()
pred = model(data)[0]
print(pred.shape)

pred_seg = pred.cpu().detach().numpy().argmax(0) * 255
print(seg.max(), pred_seg.max())

# plot
plt.figure()
plt.subplot(1, 3, 1)
plt.title('test image')
plt.axis('off')
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.title('test seg')
plt.axis('off')
plt.imshow(seg)
plt.subplot(1, 3, 3)
plt.title('output mask')
plt.axis('off')
plt.imshow(pred_seg)
plt.show()



