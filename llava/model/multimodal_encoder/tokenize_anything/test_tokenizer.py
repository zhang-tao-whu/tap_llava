import torch
import numpy as np
from PIL import Image
from .utils import im_rescale
from .utils import im_vstack

from .build_model import model_registry_


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def preprocess_images(imgs):
    """Preprocess the inference images."""
    im_batch, im_shapes, im_scales = [], [], []
    for img in imgs:
        scaled_imgs, scales = im_rescale(img, scales=[1024])
        im_batch.__iadd__(scaled_imgs), im_scales.__iadd__(scales)
        im_shapes += [x.shape[:2] for x in scaled_imgs]
    im_batch = im_vstack(im_batch, tokenizer.pixel_mean_value, size=(1024, 1024))
    im_shapes = np.array(im_shapes)
    im_scales = np.array(im_scales).reshape((len(im_batch), -1))
    im_info = np.hstack([im_shapes, im_scales]).astype("float32")
    return im_batch, im_info

tokenizer = model_registry_['tap_vit_b'](checkpoint='/home/zhangtao19/lmms/LLaVA/work_dirs/tokenize-anything/models/tap_vit_b_b45cbf.pkl')

image = '/home/zhangtao19/lmms/LLaVA/work_dirs/test.jpg'
image = Image.open(image).convert('RGB')

images = [np.array(image)]
images, _ = preprocess_images(images)
images = torch.Tensor(images)
outputs = tokenizer.foward_for_image_tokenize(images, grid_size=8, image_size=1024)
for key in outputs.keys():
    print(outputs[key].shape)
    if key == 'boxes':
        print(outputs[key])

masks = outputs['mask_pred']  # (N, H, W)

import cv2
import numpy as np
import random

def display_mask_on_image(image, mask):
    # 获取图像和 mask 的尺寸
    image_h, image_w, _ = image.shape
    mask_n, mask_h, mask_w = mask.shape

    # 创建一个与图像相同大小的空白画布
    canvas = np.zeros((image_h, image_w, 3), dtype=np.uint8)

    # 将 mask 逐个叠加到画布上
    for i in range(mask_n):
        # 将当前 mask 缩放到与图像相同大小
        resized_mask = cv2.resize(mask[i], (image_w, image_h))

        # 将 mask 的值映射到 0-255 范围
        resized_mask = (resized_mask * 255).astype(np.uint8)

        # 将 mask 的值设为蓝色
        resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        resized_mask[:, :, 0] = random.randint(0, 255)
        resized_mask[:, :, 1] = random.randint(0, 255)
        resized_mask[:, :, 2] = random.randint(0, 255)

        # 将 mask 叠加到画布上
        canvas = cv2.addWeighted(canvas, 1, resized_mask, 0.5, 0)

    # 将图像和 mask 叠加到画布上
    result = cv2.addWeighted(image, 0.7, canvas, 0.3, 0)

    # 显示结果
    cv2.imwrite('/home/zhangtao19/lmms/LLaVA/work_dirs/test_result.jpg', result)

# 示例用法
image = '/home/zhangtao19/lmms/LLaVA/work_dirs/test.jpg'
image = Image.open(image).convert('RGB')
image = np.array(image, dtype=np.uint8)
mask = masks.cpu().numpy()

display_mask_on_image(image, mask)


