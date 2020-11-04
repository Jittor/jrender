import math
import jittor as jt
from jittor import nn
import numpy
import numpy as np

from .n3mr import *

def vertices_to_faces(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 3]
    """
    assert (len(vertices.shape) == 3)
    assert (len(faces.shape) == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces + (jt.array(np.arange(bs)) * nv).unsqueeze(-1).unsqueeze(-1)
    vertices = vertices.reshape((bs * nv, 3))
    return vertices[faces]

class N3mrRasterizer(nn.Module):
    def __init__(self, image_size=256, anti_aliasing=True, background_color=[0,0,0], fill_back=True,  near=0.1, far=100):
        super(N3mrRasterizer, self).__init__()
        # rendering
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back

        self.near = near
        self.far = far

        # rasterization
        self.rasterizer_eps = 1e-3

    def execute(self, mesh, mode=None):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''
        vertices = mesh.vertices
        faces = mesh.faces
        textures = mesh.textures
        
        if mode is None:
            return self.render(vertices, faces, textures)
        elif mode is 'rgb':
            return self.render_rgb(vertices, faces, textures)
        elif mode == 'silhouettes':
            return self.render_silhouettes(vertices, faces)
        elif mode == 'depth':
            return self.render_depth(vertices, faces)
        else:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")

    def render_silhouettes(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = jt.contrib.concat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)

        # rasterization
        faces = vertices_to_faces(vertices, faces)
        images = rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = jt.contrib.concat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).stop_grad()

        # rasterization
        faces = vertices_to_faces(vertices, faces)
        images = rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render_rgb(self, vertices, faces, textures):
        # fill back
        if self.fill_back:
            faces = jt.contrib.concat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).stop_grad()
            textures = jt.contrib.concat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # rasterization
        faces = vertices_to_faces(vertices, faces)
        images = rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images

    def render(self, vertices, faces, textures):
        # fill back
        if self.fill_back:
            faces = jt.contrib.concat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).stop_grad()
            textures = jt.contrib.concat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)
        
        # rasterization
        faces = vertices_to_faces(vertices, faces)
        out = rasterize_rgbad(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return out['rgb'], out['depth'], out['alpha']
