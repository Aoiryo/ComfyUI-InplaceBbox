from random import random
from easy_nodes import (
    NumberInput,
    ComfyNode,
    MaskTensor,
    StringInput,
    ImageTensor,
    Choice,
)
import easy_nodes
import torch
import torchvision.transforms.functional as F
import cv2
import numpy as np

@ComfyNode(color="#0066cc", bg_color="#ffcc00", return_names=["Below", "Above"])
def draw_box_inplace_with_mask(bg_image: ImageTensor, mask: MaskTensor,
                    width: float = NumberInput(0.5, 1, 2, 3, display="slider")) -> ImageTensor:
    """Draw a bounding box associated with the binary mask on the image(inplace)."""

    # augment the mask
    mask = torch.any(mask > 0, dim=-1)

    # get the bounding box
    print(bg_image.shape)
    print(mask.shape)

    return 

# def draw_mask_bboxes_tensor(image_tensor, mask_tensor, color=(0, 255, 0), thickness=3):
#     """
#     在图像张量上绘制mask的边界框
    
#     参数:
#     image_tensor: 原始图像张量 (C, H, W)
#     mask_tensor: mask图像张量 (1, H, W) 或 (H, W)
#     color: 边界框颜色，默认为绿色 (R, G, B)
#     thickness: 边界框线条粗细，默认为3像素
    
#     返回:
#     带有边界框的图像张量
#     """
#     # 将张量转换为numpy数组
#     image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
#     mask_np = mask_tensor.squeeze().cpu().numpy()
    
#     # 确保mask是二值图像
#     mask_np = (mask_np > 0).astype(np.uint8) * 255
    
#     # 找到mask中的轮廓
#     contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # 为每个轮廓绘制边界框
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image_np, (x, y), (x + w, y + h), color[::-1], thickness)
    
#     # 将numpy数组转换回张量
#     result_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    
#     return result_tensor

# 使用示例
# image_tensor = torch.rand(3, 224, 224)  # 随机生成的图像张量
# mask_tensor = torch.zeros(1, 224, 224)  # 创建一个mask张量
# mask_tensor[0, 50:150, 50:150] = 1  # 在mask中创建一个矩形区域
# mask_tensor[0, 200:300, 200:300] = 1  # 再创建一个矩形区域
# 
# result_tensor = draw_mask_bboxes_tensor(image_tensor, mask_tensor)
# 
# # 如果你想要显示结果，可以使用以下代码：
# import matplotlib.pyplot as plt
# plt.imshow(result_tensor.permute(1, 2, 0).numpy())
# plt.axis('off')
# plt.show()