import os
from .clip_encoder import CLIPVisionTower
from .tokenize_anything.build_model import model_registry_


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_tap_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = 'tap_vit_l'
    checkpoint = '/home/zhangtao19/lmms/LLaVA/work_dirs/tokenize-anything/models/tap_vit_l_03f8ec.pkl'

    model = model_registry_[vision_tower](checkpoint=checkpoint)
    model.requires_grad_(False)
    model.eval()
    model.semantic_hidden_channel = 1024
    return model
