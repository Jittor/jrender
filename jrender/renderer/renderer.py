import math
import jittor as jt
from jittor import nn
import numpy

from .lighting import *
from .transform import *
from .dr import *
from ..structures import *

class Renderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=False, fill_back=True, eps=1e-3,
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='surface',
                 camera_mode='look_at',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0], dr_type='softras', Gbuffer='None'):
        super(Renderer, self).__init__()
        # camera
        self.transform = Transform(camera_mode, 
                                      K, R, t, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction)

        # light
        self.lighting = Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions, Gbuffer, self.transform)
        # rasterization
        self.dr_type = dr_type
        if dr_type == 'softras':
            self.rasterizer = SoftRasterizer(image_size, background_color, near, far, 
                                            anti_aliasing, fill_back, eps,
                                            sigma_val, dist_func, dist_eps,
                                            gamma_val, aggr_func_rgb, aggr_func_alpha,
                                            texture_type)
        elif dr_type == 'n3mr':
            self.rasterizer = N3mrRasterizer(image_size, anti_aliasing, background_color, fill_back)
        else:
            raise ValueError("dr_type should be one of None, 'softras' or 'n3mr'")

    def set_sigma(self, sigma):
        self.rasterizer.sigma_val = sigma

    def set_gamma(self, gamma):
        self.rasterizer.gamma_val = gamma

    def set_texture_mode(self, mode):
        assert mode in ['vertex', 'surface'], 'Mode only support surface and vertex'

        self.lighting.light_mode = mode
        self.rasterizer.texture_type = mode

    def render_mesh(self, mesh, mode='rgb'):
        self.set_texture_mode(mesh.texture_type)
        mesh = self.lighting(mesh, self.transform.eyes)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)
        

    def execute(self, vertices, faces, textures=None,mode='rgb', texture_type='surface', metallic_textures=None, roughness_textures=None):
        mesh = Mesh(vertices, faces, textures=textures, texture_type=texture_type, metallic_textures=metallic_textures, roughness_textures=roughness_textures)
        return self.render_mesh(mesh, mode)
