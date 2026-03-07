import h5py
with h5py.File('train.h5', 'r') as f:
    print("文件中的 Keys:", list(f.keys()))
