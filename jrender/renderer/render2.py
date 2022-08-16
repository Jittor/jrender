import jittor as jt

from .transform.transform import Transform

from .lighting import *
from .transform import *
from .dr import *
from ..structures import *

class Render():
    def __init__(self, image_size=256, background_color=[0, 0, 0], near=1, far=100,
                 camera_mode='look_at',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0,
                 eye=None, camera_direction=[0, 0, 1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1, 1, 1],
                 light_intensity_directionals=0.5, light_color_directionals=[1, 1, 1],
                 light_directions=[0, 1, 0]):

        self.transform = Transform(camera_mode,
                                   K, R, t, dist_coeffs, orig_size,
                                   perspective, viewing_angle, viewing_scale,
                                   eye, camera_direction)

        self.rasterize = SoftRasterizeFunction(image_size,
                                               background_color, near, far,texture_type="vertex")

    def VP_transform(self,vertices):
        vertices=vertices.unsqueeze(0)
        return self.transform.tranpos(vertices).squeeze(0)

    def view_transform(self,vertices):
        vertices=vertices.unsqueeze(0)
        return self.transform.view_transform(vertices).squeeze(0)


