import torch
import numpy as np
from PIL import Image

def test(model_path, image_path, scale):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 读取并转为 YCbCr
    img = Image.open(image_path).convert('RGB')
    img_ycbcr = np.array(img.convert('YCbCr'))
    y = img_ycbcr[:, :, 0].astype(np.float32) / 255.0

    # Bicubic 预放大 
    input_y = Image.fromarray((y * 255).astype(np.uint8)).resize(
        (y.shape[1] * scale, y.shape[0] * scale), resample=Image.BICUBIC)
    
    input_tensor = torch.from_numpy(np.array(input_y).astype(np.float32) / 255.0).view(1, 1, *input_y.size[::-1]).to(device)

    with torch.no_grad():
        # 测试时建议给卷积层加 padding=True (手动或修改模型) 
        # 或者在此处处理 output 以对齐尺寸
        output = model(input_tensor)
    
    # ... 后处理转回 RGB 并保存 ...
