import torch
import numpy as np
from PIL import Image

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# 推理流程：
# 1. 读取图像并转为 YCbCr
# 2. 对 Y 通道进行 Bicubic 插值放大 [cite: 94]
# 3. 输入 SRCNN 得到结果
# 4. 计算 PSNR
