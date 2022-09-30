import numpy as np
import jittor as jt
from ..textures.texture import Texture

class Light():
    DepthMap:Texture
    def __init__(self, position=[0, 0, 0], direction=[0, 0, 1], color=[1, 1, 1], up=[0, 1, 0], intensity=0.5, area = 0, type="directional", shadow = True):
        self.position = position
        self.direction = direction
        self.up = up
        self.color = jt.normalize(jt.array(color, "float32"), dim=0)
        self.intensity = intensity
        self.type = type
        self.area = area

        self.viewing_angle = 45
        self.viewing_scale = 0.9
        self.near = 0.1
        self.far = 100
        self.fillback = False
        self.shadow = shadow
        self.DepthMap = None