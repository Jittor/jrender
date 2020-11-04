import numpy as np
import jittor as jt
from jittor import nn

def directional_lighting(light, normals, light_intensity=0.5, light_color=(1,1,1), light_direction=(0,1,0)):
    light_color = jt.array(light_color, "float32")
    light_direction = jt.array(light_direction, "float32")

    if len(light_color.shape) == 1:
        light_color = light_color.unsqueeze(0)
    if len(light_direction.shape) == 1:
        light_direction = light_direction.unsqueeze(0)

    cosine = nn.relu(jt.sum(normals * light_direction, dim=2))
    light += light_intensity * (light_color.unsqueeze(1) * cosine.unsqueeze(2))
    return light