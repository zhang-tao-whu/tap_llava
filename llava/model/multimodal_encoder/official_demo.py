import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

np.random.seed(1234)
inference_mode = torch.inference_mode()
inference_mode.__enter__()

def show_mask(mask, ax):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    ax.imshow(mask.reshape(mask.shape[-2:] + (1,)) * color.reshape(1, 1, -1))

import sys
sys.path.append("..")
from tokenize_anything import model_registry

model_type = "tap_vit_b"
checkpoint = "/home/zhangtao19/lmms/LLaVA/work_dirs/tokenize-anything/models/tap_vit_b_b45cbf.pkl"
model = model_registry[model_type](checkpoint=checkpoint)

concept_weights = "/home/zhangtao19/lmms/LLaVA/work_dirs/tokenize-anything/concepts/merged_2560.pkl"
model.concept_projector.reset_weights(concept_weights)
model.text_decoder.reset_cache(max_batch_size=8)

img = cv2.imread("./fairytale.jpg")
vis_img = img.copy()[:, :, ::-1]
plt.figure(figsize=(10, 10))
plt.imshow(vis_img)
# plt.show()


from tokenize_anything.utils.image import im_rescale

img_list, img_scales = im_rescale(img, scales=[1024], max_size=1024)
input_size, original_size = img_list[0].shape, img.shape[:2]
print(input_size, "<-", original_size, "*", img_scales[0])

from tokenize_anything.utils.image import im_vstack

img_batch = im_vstack(img_list, fill_value=model.pixel_mean_value, size=(1024, 1024))
inputs = model.get_inputs({"img": img_batch})
inputs.update(model.get_features(inputs))

inputs["points"] = np.array([[[1050, 600, 1], [0, 0, 4]]], "float32")
inputs["points"][:, :, :2] *= np.array(img_scales, "float32")

# Decode outputs for the point prompt.
outputs = model.get_outputs(inputs)

# Select final mask.
iou_score, mask_pred = outputs["iou_pred"], outputs["mask_pred"]
iou_score[:, 0] -= 1000.0  # Penalize the score of boundary boxes.
mask_index = torch.arange(iou_score.shape[0]), iou_score.argmax(1)

# Upscale masks to the original image resolution.
iou_scores, masks = iou_score[mask_index], mask_pred[mask_index]
masks = model.upscale_masks(masks[:, None], img_batch.shape[1:-1])
masks = masks[..., : input_size[0], : input_size[1]]
masks = model.upscale_masks(masks, original_size).gt(0).cpu().numpy()

# Predict concepts and generate captions.
sem_tokens, sem_embeds = outputs["sem_tokens"], outputs["sem_embeds"]
concepts, scores = model.predict_concept(sem_embeds[mask_index])
#captions = model.generate_text(sem_tokens[mask_index][:, None, :])

# Display comprehensive visual understanding.
text_contents = [v.flatten()[0] for v in (concepts, iou_scores, scores)]
vis_text = "{} ({:.2f}, {:.2f}):".format(*text_contents)
plt.figure(figsize=(10,10))
plt.imshow(vis_img)
plt.figtext(0.5, 0.1, vis_text, fontsize=16, ha="center")
show_mask(masks, plt.gca())
plt.axis('off')
plt.savefig('/home/zhangtao19/lmms/LLaVA/work_dirs/test_fig.png')

