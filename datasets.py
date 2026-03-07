import h5py
import torch
from torch.utils.data import Dataset

class SRDataset(Dataset):
    def __init__(self, h5_file):
        super(SRDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # 修正：原论文生成的 H5 文件通常使用 'data' 和 'label'
            lr = torch.from_numpy(f['data'][idx]).float()
            hr = torch.from_numpy(f['label'][idx]).float()
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            # 修正：确保此处键名与上面一致
            return len(f['data'])
