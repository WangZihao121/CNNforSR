import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import SRCNN
from datasets import SRDataset

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    
    # 初始化权重 
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
            nn.init.constant_(m.bias, 0.0)

    # 差异化学习率：最后一层更小以利于收敛 [cite: 239, 240]
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': 1e-5}
    ], lr=1e-4)

    criterion = nn.MSELoss()
    train_loader = DataLoader(SRDataset('train.h5'), batch_size=128, shuffle=True)

    model.train()
    # 论文中训练了约 8e8 次 Backprop [cite: 225, 243]
    # 这里你可以根据 epoch 转换，大约 1e5 到 1e6 个 iter 即可看到良好效果
    for epoch in range(200):
        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            preds = model(lr)
            loss = criterion(preds, hr)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
