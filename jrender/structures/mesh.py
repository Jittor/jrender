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
    def __init__(self, vertices, faces, textures=None, texture_res=1, texture_type='surface', dr_type='softras', metallic_textures=None, roughness_textures=None,normal_textures=None,TBN=None,with_SSS=False,face_texcoords=None):
        '''
        vertices, faces and textures (if not None) are expected to be Tensor objects
        '''
        self._vertices = vertices
        self._faces = faces

        if isinstance(self._vertices, np.ndarray):
            self._vertices = jt.array(self._vertices).float()
        if isinstance(self._faces, np.ndarray):
            self._faces = jt.array(self._faces).int()

        self.texture_type = texture_type

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
        self._surface_ResNormals=None
        self._surface_ResNormals_update= True
        self._vertex_normals = None
        self._vertex_normals_update = True
        self._with_specular = True

        self._face_texcoords= face_texcoords
        if self._face_texcoords is not None:
            self._face_texcoords=self._face_texcoords.unsqueeze(0)
        self._with_SSS = with_SSS

        self._fill_back = False
        self.dr_type = dr_type
        
        if texture_type == 'surface':
            if self.dr_type == 'softras':
                self._metallic_textures = jt.zeros((self.batch_size, self.num_faces, texture_res**2, 1))
                self._roughness_textures = jt.ones((self.batch_size, self.num_faces, texture_res**2, 1))
            elif self.dr_type == 'n3mr':
                self._metallic_textures = jt.zeros((self.batch_size, self.num_faces, texture_res, texture_res, texture_res, 1))
                self._roughness_textures = jt.ones((self.batch_size, self.num_faces, texture_res, texture_res, texture_res, 1))
        elif texture_type == 'vertex':
            self._metallic_textures = jt.zeros((self.batch_size, self.num_vertices, 1))
            self._roughness_textures = jt.ones((self.batch_size, self.num_vertices, 1))
       
        if metallic_textures is not None:
            self._metallic_textures = metallic_textures
        if roughness_textures is not None:
            self._roughness_textures = roughness_textures

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

        #create normal_textures
        if normal_textures is not None:
            normal_textures=normal_textures.unsqueeze(0)
            TBN=TBN.unsqueeze(0)
        self._TBN=TBN             
        self._normal_textures=normal_textures
        self._origin_vertices = self._vertices
        self._origin_faces = self._faces
        self._origin_textures = self._textures

    @property
    def with_specular(self):
        return self._with_specular
    
    @property
    def with_SSS(self):
        return self._with_SSS

    @with_specular.setter
    def with_specular(self, with_specular):
        self._with_specular = with_specular

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
    def metallic_textures(self):
        return self._metallic_textures

    @metallic_textures.setter
    def metallic_textures(self, metallic_textures):
        self._metallic_textures = metallic_textures

    @property
    def roughness_textures(self):
        return self._roughness_textures

    @roughness_textures.setter
    def roughness_textures(self, roughness_textures):
        self._roughness_textures = roughness_textures

    @property
    def face_vertices(self):
        if self._face_vertices_update:
            self._face_vertices = face_vertices(self.vertices, self.faces)
            self._face_vertices_update = False
        return self._face_vertices

    @property
    def surface_normals(self):
        if self._surface_normals_update:
            if self.normal_textures is None:
                v10 = self.face_vertices[:, :, 0] - self.face_vertices[:, :, 1]
                v12 = self.face_vertices[:, :, 2] - self.face_vertices[:, :, 1]
                self._surface_normals = jt.normalize(jt.cross(v12, v10), p=2, dim=2, eps=1e-6)
            else:
                surface_normals=jt.sum(self._normal_textures,dim=2)/self.texture_res**2
                surface_normals=surface_normals.unsqueeze(2)
                self._surface_normals=jt.normalize((jt.matmul(surface_normals,self.TBN).squeeze(2)),dim=2)
                
            self._surface_normals_update = False
        return self._surface_normals
        
    @property
    def vertex_normals(self):                       #没有直接从.obj读取vertex_normals的功能
        if self._vertex_normals_update:
            bs, nv = self.vertices.shape[:2]
            bs, nf = self.faces.shape[:2]

            faces = self.faces + (jt.arange(bs) * nv)[:, None, None]
            vertices_faces = self.vertices.reshape((bs * nv, 3))[faces.long()]

            faces = faces.view(-1, 3)
            vertices_faces = vertices_faces.view(-1, 3, 3)

            normals = (jt.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1])).reindex_reduce(op="sum", shape=[bs * nv, 3], indexes=["@e0(i0)", "i1"], extras=[faces[:, 1].long()]) + (jt.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2])).reindex_reduce(op="sum", shape=[bs * nv, 3], indexes=["@e0(i0)", "i1"], extras=[faces[:, 2].long()]) + (jt.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0])).reindex_reduce(op="sum", shape=[bs * nv, 3], indexes=["@e0(i0)", "i1"], extras=[faces[:, 0].long()])
            
            normals = jt.normalize(normals, p=2, eps=1e-6, dim=1)
            self._vertex_normals = normals.reshape((bs, nv, 3))
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
    
    @property
    def normal_textures(self):
        return self._normal_textures

    @property
    def surface_ResNormals(self):
        if self._surface_ResNormals_update:
            TBN=self.TBN.unsqueeze(2)
            TBN=TBN.repeat(1,1,self.texture_res**2,1,1)
            normal_textures=self.normal_textures.unsqueeze(3)
            surface_ResNormals=jt.matmul(normal_textures,TBN).squeeze(3)
            self._surface_ResNormals=jt.normalize(surface_ResNormals,dim=3)
            self._surface_ResNormals_update = False
        return self._surface_ResNormals

    @property
    def face_texcoords(self):
        return self._face_texcoords

    @property
    def TBN(self):
        return self._TBN           

    @classmethod
    def from_obj(cls, filename_obj, normalization=False, load_texture=False, dr_type='softras', texture_res=1, texture_type='surface', texture_wrapping='REPEAT', use_bilinear=True, with_SSS = False):
        '''
        Create a Mesh object from a .obj file
        '''
        if load_texture:
            vertices, faces, textures ,normal_textures,TBN,face_texcoords= load_obj(filename_obj,
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
            normal_textures= None
            TBN = None
        return cls(vertices, faces, textures, texture_res, texture_type, dr_type=dr_type,normal_textures=normal_textures,TBN=TBN,with_SSS=with_SSS,face_texcoords=face_texcoords)

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
