import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import SRCNN
from datasets import SRDataset

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SRCNN().to(device)

    # 1. 忠实论文的初始化 
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
            nn.init.constant_(m.bias, 0.0)

    # 2. 忠实论文的差异化学习率与 SGD 
    optimizer = optim.SGD([
        {'params': model.conv1.parameters(), 'lr': 1e-4},
        {'params': model.conv2.parameters(), 'lr': 1e-4},
        {'params': model.conv3.parameters(), 'lr': 1e-5}
    ], momentum=0.9)

    criterion = nn.MSELoss()
    
    # 3. 加载训练集
    train_loader = DataLoader(SRDataset('train.h5'), batch_size=128, shuffle=True)

    model.train()
    num_epochs = 200 # 你可以根据需要调整
    
    # 4. 完整的训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for i, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device)
            hr = hr.to(device)

            # 前向传播
            preds = model(lr)
            
            # 【关键修改】：处理无 padding 带来的尺寸缩小
            # lr 是 32x32 -> preds 变成 20x20，但 hr 依然是 32x32
            # 需要对 hr 进行中心裁剪，使其变成 20x20 才能计算 MSELoss
            if preds.shape != hr.shape:
                diff = (hr.size(2) - preds.size(2)) // 2
                hr_cropped = hr[:, :, diff:-diff, diff:-diff]
            else:
                hr_cropped = hr
                
            loss = criterion(preds, hr_cropped)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.6f}")
        
        # 每隔 10 个 epoch 保存一次模型权重
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'srcnn_epoch_{epoch+1}.pth')

    # 保存最终模型
    torch.save(model.state_dict(), 'srcnn_final.pth')
    print("Training finished! Model saved as 'srcnn_final.pth'")

if __name__ == '__main__':
    train()
