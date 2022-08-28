import jittor as jt
from jittor import nn
from jittor import Function
import numpy as np
from .cuda import soft_rasterize as soft_rasterize_cuda

class SoftRasterizeFunction(Function):
    def __init__(self, image_size=256, 
                background_color=[0, 0, 0], near=1, far=100, 
                fill_back=True, eps=1e-3,
                sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                texture_type='surface'):

        self.image_size = image_size
        self.background_color = background_color
        self.near = near
        self.far = far
        self.eps = eps
        self.sigma_val = sigma_val
        self.gamma_val = gamma_val
        self.dist_func = dist_func
        self.dist_eps = np.log(1. / dist_eps - 1.)
        self.aggr_func_rgb = aggr_func_rgb
        self.aggr_func_alpha = aggr_func_alpha
        self.fill_back = fill_back
        self.texture_type = texture_type

    def execute(self, face_vertices, textures):

        # face_vertices: [nb, nf, 9]
        # textures: [nb, nf, 9]

        func_dist_map = {'hard': 0, 'barycentric': 1, 'euclidean': 2}
        func_rgb_map = {'hard': 0, 'softmax': 1,'none':2}
        func_alpha_map = {'hard': 0, 'sum': 1, 'prod': 2}
        func_map_sample = {'surface': 0, 'vertex': 1}

        image_size = self.image_size
        background_color = self.background_color
        near = self.near
        far = self.far
        eps = self.eps
        sigma_val = self.sigma_val
        gamma_val = self.gamma_val
        dist_eps = self.dist_eps
        fill_back = self.fill_back

        func_dist_type = func_dist_map[self.dist_func]
        func_rgb_type = func_rgb_map[self.aggr_func_rgb]
        func_alpha_type = func_alpha_map[self.aggr_func_alpha]
        texture_type = func_map_sample[self.texture_type]

        face_vertices = face_vertices.clone()
        textures = textures.clone()

        self.batch_size, self.num_faces = face_vertices.shape[:2]

        faces_info = jt.zeros((self.batch_size, self.num_faces, 9*3)) # [inv*9, sym*9, obt*3, 0*6]
        aggrs_info = jt.zeros((self.batch_size, 2, self.image_size, self.image_size))

        soft_colors = jt.ones((self.batch_size, 4, self.image_size, self.image_size))
        soft_colors[:, 0, :, :] *= background_color[0]
        soft_colors[:, 1, :, :] *= background_color[1]
        soft_colors[:, 2, :, :] *= background_color[2]

        # TODO: soft_colors do not init in cuda
        faces_info, aggrs_info, soft_colors = \
            soft_rasterize_cuda.forward_soft_rasterize(face_vertices, textures,
                                                       faces_info, aggrs_info,
                                                       soft_colors,
                                                       image_size, near, far, eps,
                                                       sigma_val, func_dist_type, dist_eps,
                                                       gamma_val, func_rgb_type, func_alpha_type,
                                                       texture_type, int(fill_back))

        self.save_vars = face_vertices, textures, soft_colors, faces_info, aggrs_info
        return soft_colors

    def grad(self, grad_soft_colors):

        face_vertices, textures, soft_colors, faces_info, aggrs_info = self.save_vars
        image_size = self.image_size
        background_color = self.background_color
        near = self.near
        far = self.far
        eps = self.eps
        sigma_val = self.sigma_val
        dist_eps = self.dist_eps
        gamma_val = self.gamma_val
        func_dist_type = self.func_dist_type
        func_rgb_type = self.func_rgb_type
        func_alpha_type = self.func_alpha_type
        texture_type = self.texture_type
        fill_back = self.fill_back

        grad_faces = jt.zeros((face_vertices.shape))
        grad_textures = jt.zeros((textures.shape))

        grad_faces, grad_textures = \
            soft_rasterize_cuda.backward_soft_rasterize(face_vertices, textures, soft_colors, 
                                                        faces_info, aggrs_info,
                                                        grad_faces, grad_textures, grad_soft_colors, 
                                                        image_size, near, far, eps,
                                                        sigma_val, func_dist_type, dist_eps,
                                                        gamma_val, func_rgb_type, func_alpha_type,
                                                        texture_type, int(fill_back))
        return grad_faces, grad_textures


def soft_rasterize(face_vertices, textures, image_size=256, 
                   background_color=[0, 0, 0], near=1, far=100, 
                   fill_back=True, eps=1e-3,
                   sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                   gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                   texture_type='surface'):
    return SoftRasterizeFunction(image_size, 
                                background_color, near, far,
                                fill_back, eps,
                                sigma_val, dist_func, dist_eps,
                                gamma_val, aggr_func_rgb, aggr_func_alpha, 
                                texture_type)(face_vertices, textures)