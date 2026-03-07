import h5py
import torch
from torch.utils.data import Dataset

class SRDataset(Dataset):
    def __init__(self, h5_file):
        super(SRDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # 修改处：将 'lr' 改为 'data'，将 'hr' 改为 'label'
            # 原始论文通常在 Y 通道上进行 Patch 训练
            lr = torch.from_numpy(f['data'][idx]).float()
            hr = torch.from_numpy(f['label'][idx]).float()
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            # 修改处：将 'lr' 改为 'data'
            return len(f['data'])
