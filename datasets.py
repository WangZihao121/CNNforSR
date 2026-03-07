import h5py
import torch
from torch.utils.data import Dataset

class SRDataset(Dataset):
    def __init__(self, h5_file):
        super(SRDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # 读取数据并转为 float32 类型的 PyTorch 张量
            lr = torch.from_numpy(f['lr'][idx]).float()
            hr = torch.from_numpy(f['hr'][idx]).float()
            
            # 【关键修改】：如果数据只有二维 (H, W)，增加通道维度变成 (1, H, W)
            if lr.dim() == 2:
                lr = lr.unsqueeze(0)
            if hr.dim() == 2:
                hr = hr.unsqueeze(0)
                
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
