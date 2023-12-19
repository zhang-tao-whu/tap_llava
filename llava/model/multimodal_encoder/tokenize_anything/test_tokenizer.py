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
images = torch.Tensor(images).to(tokenizer.pixel_mean.device)
outputs = tokenizer.foward_for_image_tokenize(images, grid_size=8, image_size=1024)
for key in outputs.keys():
    print(outputs[key].shape)



