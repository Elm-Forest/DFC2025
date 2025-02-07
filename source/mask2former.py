import kornia
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoConfig, \
    Mask2FormerForUniversalSegmentation


class Timm_Mask2Former(nn.Module):
    def __init__(self, in_channels=1, pretrained='facebook/mask2former-swin-large-cityscapes-semantic'):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained)
        config.backbone_config.num_channels = in_channels
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained, config=config, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 测试代码
if __name__ == "__main__":
    model = Timm_Mask2Former(in_channels=1).cuda()
    from PIL import Image
    image = Image.open('K:\\dataset\\dfc25\\train\\sar_images\\TrainArea_057.tif')
    img_array = np.array(image)
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        logits = model(img_tensor)
    print(logits)  # 预期输出: torch.Size([2, 10, 256, 256])
