import torch
from torch import nn, optim
from model import UNET
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

# path & device
image_dir = "data/train_images"
mask_dir = "data/train_masks"
weight_path = "save_weights/unet_carvana.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
batch_size = 8
learning_rate = 1e-5
epochs = 6

# 模型，数据集
model = UNET(3, 2).to(device)
dataset = CarvanaDataset(image_dir, mask_dir, (240, 160))  # 宽240 高160
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# 损失函数，优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(epochs):
    for idx, (image, label) in enumerate(dataloader):
        image = image.to(device,  dtype=torch.float32)
        label = label.to(device, dtype=torch.long)
        # forward calculate
        out = model(image)
        # loss
        loss = criterion(out, label)
        # backward boardcast
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(f"epoch: {epoch}/{epochs}, iter: {idx}th, loss: {loss.item()}")
    torch.save(model.state_dict(), weight_path)  # 每个epoch结束后都存一个参数













