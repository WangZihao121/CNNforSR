import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class SRDataset(Dataset):
    def __init__(self, h5_file):
        super(SRDataset, self).__init__()
        self.h5_file = h5_file

        # 读取数据长度
        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f['data'])

    def __getitem__(self, index):
        with h5py.File(self.h5_file, 'r') as f:
            lr = np.array(f['data'][index])
            hr = np.array(f['label'][index])

        # 转为 tensor
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()

        # 增加 channel 维度
        lr = lr.unsqueeze(0)
        hr = hr.unsqueeze(0)

        return lr, hr

    def __len__(self):
        return self.length
