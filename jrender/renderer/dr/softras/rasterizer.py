
import numpy as np
import jittor as jt
from jittor import nn

from .soft_rasterize import *

class SoftRasterizer(nn.Module):
    def __init__(self, image_size=256, background_color=[0, 0, 0], near=1, far=100, 
                 anti_aliasing=False, fill_back=False, eps=1e-3, 
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='surface',
                 bin_size = 0, max_elems_per_bin = 0, max_faces_per_pixel_for_grad=16):
        super(SoftRasterizer, self).__init__()

        if dist_func not in ['hard', 'euclidean', 'barycentric']:
            raise ValueError('Distance function only support hard, euclidean and barycentric')
        if aggr_func_rgb not in ['hard', 'softmax']:
            raise ValueError('Aggregate function(rgb) only support hard and softmax')
        if aggr_func_alpha not in ['hard', 'prod', 'sum']:
            raise ValueError('Aggregate function(a) only support hard, prod and sum')
        if texture_type not in ['surface', 'vertex']:
            raise ValueError('Texture type only support surface and vertex')

        self.image_size = image_size
        self.background_color = background_color
        self.near = near
        self.far = far
        self.anti_aliasing = anti_aliasing
        self.eps = eps
        self.fill_back = fill_back
        self.sigma_val = sigma_val
        self.dist_func = dist_func
        self.dist_eps = dist_eps
        self.gamma_val = gamma_val
        self.aggr_func_rgb = aggr_func_rgb
        self.aggr_func_alpha = aggr_func_alpha
        self.texture_type = texture_type
        self.bin_size = bin_size
        self.max_elems_per_bin = max_elems_per_bin
        self.max_faces_per_pixel_for_grad = max_faces_per_pixel_for_grad

    def execute(self, mesh, mode=None):
        image_size = self.image_size * (2 if self.anti_aliasing else 1)
        images = soft_rasterize(mesh.face_vertices, mesh.face_textures, image_size, 
                                    self.background_color, self.near, self.far, 
                                    self.fill_back, self.eps,
                                    self.sigma_val, self.dist_func, self.dist_eps,
                                    self.gamma_val, self.aggr_func_rgb, self.aggr_func_alpha,
                                    self.texture_type, self.bin_size, self.max_elems_per_bin,
                                    self.max_faces_per_pixel_for_grad)

        if self.anti_aliasing:
            images = nn.pool(images, 2, "mean", stride=2)
        if mode == 'silhouettes':
            return images[:,3,:,:]
        elif mode == 'rgb':
            return images[:,:3,:,:]
        elif mode is None:
            return images[:,3,:,:], images[:,:3,:,:]