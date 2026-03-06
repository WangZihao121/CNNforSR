import h5py
import torch
from torch.utils.data import Dataset

class SRDataset(Dataset):
    def __init__(self, h5_file):
        super(SRDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # 原论文通常处理 Y 通道 [cite: 230]
            lr = torch.from_numpy(f['lr'][idx]).float()
            hr = torch.from_numpy(f['hr'][idx]).float()
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
