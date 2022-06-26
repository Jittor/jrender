import numpy as np
import jittor as jt

def ambient_lighting(light, light_intensity=0.5, light_color=(1,1,1)):
    light_color = jt.array(light_color)
    if len(light_color.shape) == 1:
        light_color = light_color.unsqueeze(0)
    
    light += light_intensity * light_color.unsqueeze(1)
    return light