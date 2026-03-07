import numpy as np

def calculate_psnr(img1, img2, border=0):
    # border 取决于卷积损失的像素 (f1=9, f2=1, f3=5 总损失约 6+0+2=8 像素半径)
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))
