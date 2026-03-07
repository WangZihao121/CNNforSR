import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SRCNN
from datasets import SRDataset
import math

def calc_psnr(mse):
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)

    # 1. 忠实论文的初始化 
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
            nn.init.constant_(m.bias, 0.0)

    # 2. 忠实论文的差异化学习率与 SGD (Momentum=0.9)
    optimizer = optim.SGD([
        {'params': model.conv1.parameters(), 'lr': 1e-4},
        {'params': model.conv2.parameters(), 'lr': 1e-4},
        {'params': model.conv3.parameters(), 'lr': 1e-5}
    ], momentum=0.9)

    criterion = nn.MSELoss()
    
    # 3. 加载训练集和验证集
    train_loader = DataLoader(SRDataset('train.h5'), batch_size=128, shuffle=True)
    val_loader = DataLoader(SRDataset('test.h5'), batch_size=1, shuffle=False)

    num_epochs = 400 # 论文训练步数极长，可以设置较大的 epoch 数量
    best_psnr = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # --- 训练阶段 ---
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            
            # [关键] 尺寸对齐计算 Loss
            # 输入 32x32 -> 输出 20x20。因此需要对标签进行中心裁剪
            # 如果你的 h5 文件里的 hr 已经是 20x20，可以直接 loss = criterion(preds, labels)
            if preds.shape != labels.shape:
                diff = (labels.size(2) - preds.size(2)) // 2
                labels = labels[:, :, diff:-diff, diff:-diff]
                
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss/len(train_loader):.6f}")

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                inputs, labels = val_data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                
                # 同理，尺寸对齐
                if preds.shape != labels.shape:
                    diff = (labels.size(2) - preds.size(2)) // 2
                    labels = labels[:, :, diff:-diff, diff:-diff]
                    
                mse = criterion(preds, labels)
                val_loss += mse.item()
                
        # 计算验证集平均 PSNR
        avg_val_mse = val_loss / len(val_loader)
        val_psnr = calc_psnr(avg_val_mse)
        print(f"Epoch [{epoch+1}/{num_epochs}] Val PSNR: {val_psnr:.2f} dB")

        # 保存最佳模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), 'best_srcnn.pth')
            print(f"-> 发现最佳模型，已保存! (PSNR: {best_psnr:.2f} dB)")

if __name__ == '__main__':
    train()
