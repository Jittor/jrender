import numpy as np
import jittor as jt

from ..io import *
from .utils import *

class Mesh(object):
    '''
    A class for creating and manipulating trimesh objects.
    
    Attributes:
        * vertices: [batch_size, num_vertices, 3]
        * faces: [batch_size, num_faces, 3]
        * textures: 
            - if dr_type is 'softras': 
                - if texture_type is 'surface', textures' shape is [batch_size, num_faces, texture_res ** 2, 3] 
                - if texture_type is 'vertex', textures' shape is [batch_size, num_vertices, 3] 
            - if dr_type is 'n3mr': 
                - textures' shape is [batch_size, num_faces, texture_res, texture_res, texture_res, 3] 
        * face_vertices: [batch_size, num_faces, 3, 3]
        * face_textures: 
            - if dr_type is 'softras': 
                - if texture_type is 'surface', face_textures' shape is [batch_size, num_faces, texture_res ** 2, 3] 
                - if texture_type is 'vertex', face_textures' shape is [batch_size, num_faces, 3] 
            - if dr_type is 'n3mr': 
                - face_textures' shape is [batch_size, num_faces, texture_res, texture_res, texture_res, 3] 
        * surface_normals: the normals of all surfaces. Shape is [batch_size, num_faces, 3]
        * vertex_normals: the normals of all vertices. Shape is [batch_size, num_vertices, 3]

    Functions:
        faces: set faces
        vertices: set vertices
        textures: set textures
        fill_back_: fill the back of all triangles
        reset_: reset vertices, faces, textures to their origin created values
        from_obj: create a mesh from one .obj file
        save_obj: save a mesh to one .obj file
        voxelize: voxelize the vertices to voxel space
    '''
    def __init__(self, vertices, faces, textures=None, texture_res=1, texture_type='surface', dr_type='softras'):
        '''
        vertices, faces and textures (if not None) are expected to be Tensor objects
        '''
        self._vertices = vertices
        self._faces = faces

        if isinstance(self._vertices, np.ndarray):
            self._vertices = jt.array(self._vertices).float()
        if isinstance(self._faces, np.ndarray):
            self._faces = jt.array(self._faces).int()
        if len(self._vertices.shape) == 2:
            self._vertices = self._vertices.unsqueeze(0)
        if len(self._faces.shape) == 2:
            self._faces = self._faces.unsqueeze(0)

        self.texture_type = texture_type

        self.batch_size = self._vertices.shape[0]
        self.num_vertices = self._vertices.shape[1]
        self.num_faces = self._faces.shape[1]

        self._face_vertices = None
        self._face_vertices_update = True
        self._surface_normals = None
        self._surface_normals_update = True
        self._vertex_normals = None
        self._vertex_normals_update = True

        self._fill_back = False
        self.dr_type = dr_type

        # create textures
        if textures is None:
            if texture_type == 'surface':
                if self.dr_type == 'softras':
                    self._textures = jt.ones((self.batch_size, self.num_faces, texture_res**2, 3))
                elif self.dr_type == 'n3mr':
                    self._textures = jt.ones((self.batch_size, self.num_faces, texture_res, texture_res, texture_res, 3))
                self.texture_res = texture_res
            elif texture_type == 'vertex':
                self._textures = jt.ones((self.batch_size, self.num_vertices, 3))
                self.texture_res = 1
        else:
            if isinstance(textures, np.ndarray):
                textures = jt.array(textures).float()
            if len(textures.shape) == 3 and texture_type == 'surface':
                textures = textures.unsqueeze(0)
            if len(textures.shape) == 2 and texture_type == 'vertex':
                textures = textures.unsqueeze(0)
            if len(textures.shape) == 5:
                textures = textures.unsqueeze(0)
            self._textures = textures
            self.texture_res = int(np.sqrt(self._textures.shape[2]))

        self._origin_vertices = self._vertices
        self._origin_faces = self._faces
        self._origin_textures = self._textures

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        # need check tensor
        self._faces = faces
        self.num_faces = self._faces.shape[1]
        self._face_vertices_update = True
        self._surface_normals_update = True
        self._vertex_normals_update = True

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        # need check tensor
        self._vertices = vertices
        self.num_vertices = self._vertices.shape[1]
        self._face_vertices_update = True
        self._surface_normals_update = True
        self._vertex_normals_update = True

    @property
    def textures(self):
        return self._textures

    @textures.setter
    def textures(self, textures):
        # need check tensor
        self._textures = textures

    @property
    def face_vertices(self):
        if self._face_vertices_update:
            self._face_vertices = face_vertices(self.vertices, self.faces)
            self._face_vertices_update = False
        return self._face_vertices

    @property
    def surface_normals(self):
        if self._surface_normals_update:
            v10 = self.face_vertices[:, :, 0] - self.face_vertices[:, :, 1]
            v12 = self.face_vertices[:, :, 2] - self.face_vertices[:, :, 1]
            self._surface_normals = jt.normalize(jt.cross(v12, v10), p=2, dim=2, eps=1e-6)
            self._surface_normals_update = False
        return self._surface_normals

    @property
    def vertex_normals(self):
        if self._vertex_normals_update:
            self._vertex_normals = vertex_normals(self.vertices, self.faces)
            self._vertex_normals_update = False
        return self._vertex_normals

    @property
    def face_textures(self):
        if self.texture_type in ['surface']:
            return self.textures
        elif self.texture_type in ['vertex']:
            return face_vertices(self.textures, self.faces)
        else:
            raise ValueError('texture type not applicable')

    def fill_back_(self):
        if not self._fill_back:
            self.faces = jt.contrib.concat((self.faces, self.faces[:, :, [2, 1, 0]]), dim=1)
            self.textures = jt.contrib.cat((self.textures, self.textures), dim=1)
            self._fill_back = True

    def reset_(self):
        self.vertices = self._origin_vertices
        self.faces = self._origin_faces
        self.textures = self._origin_textures
        self._fill_back = False
    
    @classmethod
    def from_obj(cls, filename_obj, normalization=False, load_texture=False, dr_type='softras', texture_res=1, texture_type='surface', texture_wrapping='REPEAT', use_bilinear=True):
        '''
        Create a Mesh object from a .obj file
        '''
        if load_texture:
            vertices, faces, textures = load_obj(filename_obj,
                                                normalization=normalization,
                                                texture_res=texture_res,
                                                load_texture=True,
                                                dr_type=dr_type,
                                                texture_type=texture_type,
                                                texture_wrapping=texture_wrapping, 
                                                use_bilinear=use_bilinear)
        else:
            vertices, faces = load_obj(filename_obj,
                                    normalization=normalization,
                                    texture_res=texture_res,
                                    load_texture=False, dr_type=dr_type)
            textures = None
        return cls(vertices, faces, textures, texture_res, texture_type, dr_type=dr_type)

    def save_obj(self, filename_obj, save_texture=False, texture_res_out=16):
        if self.batch_size != 1:
            raise ValueError('Could not save when batch size >= 1')
        if save_texture:
            save_obj(filename_obj, self.vertices[0], self.faces[0], 
                         textures=self.textures[0],
                         texture_res=texture_res_out, texture_type=self.texture_type)
        else:
            save_obj(filename_obj, self.vertices[0], self.faces[0], textures=None)

    def voxelize(self, voxel_size=32):
        face_vertices_norm = self.face_vertices * voxel_size / (voxel_size - 1) + 0.5
        return voxelization(face_vertices_norm, voxel_size, False)