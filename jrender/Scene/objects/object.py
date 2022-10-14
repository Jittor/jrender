
from sre_constants import RANGE
import numpy as np
import jittor as jt
from ..textures.texture import Texture
from .utils import *
from ...io.utils.load_textures import _load_textures_for_softras

class obj():
    def __init__(self, Ka, Kd, Ke, Ns, Ni,
                 face_vertices, face_normals_from_obj, kd_texture_uv,
                 material_name,
                 reflection_type="diffuse",
                 map_Kd_path=None, map_normal_path=None, obj_path=None, mtl_path=None,
                 map_metallic_path=None, map_roughness_path=None, map_albedo_path=None, kd_res = 0):
        self.material_name = material_name
        self._Ka = Ka
        self._Kd = Kd
        self._Ke = Ke
        self._Ns = Ns
        self._Ni = Ni
        self._roughness = 1
        self.reflection_type = reflection_type
        self.with_specular = True
        self.kd_res = kd_res

        #texture
        self._albedo_textures = None
        self.albedo_textures_update = True
        self._metallic_textures = None
        self.metallic_textures_update = True
        self._roughness_textures = None
        self.roughness_textures_update = True
        self._kd_textures = None
        self.kd_textures_update = True
        self._normal_textures = None
        self.normal_textures_update = True

        #path
        self.map_Kd_path = map_Kd_path
        self.map_normal_path = map_normal_path
        self.map_albedo_path = map_albedo_path
        self.map_metallic_path = map_metallic_path
        self.map_roughness_path = map_roughness_path
        self.obj_path = obj_path
        self.mtl_path = mtl_path

        #geometry
        if isinstance(face_vertices, np.ndarray):
            face_vertices = jt.array(face_vertices)
        if isinstance(kd_texture_uv, np.ndarray):
            kd_texture_uv = jt.array(kd_texture_uv)
        if isinstance(face_normals_from_obj, np.ndarray):
            face_normals_from_obj = jt.array(face_normals_from_obj)
        self._face_vertices = face_vertices
        self.face_normals_from_obj = face_normals_from_obj
        self._face_normals = None
        self.face_normals_update = True
        self._kd_texture_uv = kd_texture_uv
        self._face_kd = None
        self.face_kd_update = True

        self._surface_normals = None
        self.surface_normals_update = True
        self.Generate_Normals = "surface"
        
    @property
    def face_vertices(self):  # [nf,3,3]
        return self._face_vertices

    @property
    def face_normals(self):  # [nf,3,3]
        if self.face_normals_update:
            if self.Generate_Normals == "surface":
                self._face_normals = jt.ones_like(self.face_vertices) * self.surface_normals.unsqueeze(1)
            elif self.Generate_Normals == "vertex":  # to do
                self._face_normals = None
            elif self.Generate_Normals == "normal_texture":
                normals = self.normal_textures.query_uv
                TBN = create_TBN(self._kd_texture_uv, self.face_vertices)
                self._face_normals = jt.matmul(normals.unsqueeze(2), TBN.unsqueeze()).squeeze(2)
            elif self.Generate_Normals == "from_obj":
                self._face_normals = jt.normalize(self.face_normals_from_obj,dim=2)
            self.face_normals_update = False
        return self._face_normals

    @property
    def surface_normals(self):
        if self.surface_normals_update:
            if self.normal_textures is None:
                v10 = self.face_vertices[:, 0] - self.face_vertices[:, 1]
                v12 = self.face_vertices[:, 2] - self.face_vertices[:, 1]
                self._surface_normals = jt.normalize(jt.cross(v12, v10), p=2, dim=1, eps=1e-6)
            self.surface_normals_update = False
        return self._surface_normals

    @property
    def metallic_textures(self):
        if self.metallic_textures_update:
            self._metallic_textures = Texture.from_path(self.map_metallic_path)
            self.metallic_textures_update = False
        return self._metallic_textures

    @property
    def roughness_textures(self):
        if self.roughness_textures_update:
            self._roughness_textures = Texture.from_path(self.map_roughness_path)
            self.roughness_textures_update = False
        return self._roughness_textures

    @property
    def albedo_textures(self):
        if self.albedo_textures_update:
            self._albedo_textures = Texture.from_path(self.map_albedo_path)
            self.albedo_textures_update = False
        return self._albedo_textures

    @property
    def normal_textures(self):
        if self.normal_textures_update:
            self._normal_textures = Texture.from_path(self.map_normal_path)
            self.normal_textures_update = False
        return self._normal_textures

    @property
    def kd_textures(self):
        if self.kd_textures_update:
            self._kd_textures = Texture.from_path(self.map_Kd_path)
            if self._kd_textures is not None:
                self._kd_textures.image = self._kd_textures.image[::-1,:,:]
            self.kd_textures_update = False
        return self._kd_textures

    @property
    def face_albedo(self):
        if self.albedo_textures is not None:
            return self.albedo_textures.query_uv
        else:
            return jt.ones_like(self.face_vertices)

    @property
    def face_metallic(self):  # [nf,3,1]
        if self.metallic_textures is not None:
            return self.metallic_textures.query_uv
        else:
            return jt.zeros([self.face_vertices.shape[0], self.face_vertices.shape[1], 1], "float32")

    @property
    def face_roughness(self):  # [nf,3,1]
        if self.roughness_textures is not None:
            return self.roughness_textures.query_uv
        else:
            return jt.ones((self.face_vertices.shape[0], self.face_vertices.shape[1], 1), "float32") * self._roughness

    @property
    def specular(self):
        if self.with_specular is True:
            return jt.ones((self.face_vertices.shape[0], self.face_vertices.shape[1], 1), "float32")
        else:
            return jt.zeros((self.face_vertices.shape[0], self.face_vertices.shape[1], 1), "float32") 

    @property
    def face_kd(self):
        if self.face_kd_update:
            if self.kd_textures is not None:
                if (self.kd_res == 0):
                    self.kd_textures.uv = self._kd_texture_uv
                    self._face_kd = self.kd_textures.query_uv
                else:
                    image = self.kd_textures.image
                    faces = self._kd_texture_uv
                    textures = jt.ones((self.face_vertices.shape[0],self.kd_res,3),"float32")
                    is_update = jt.ones((self.face_vertices.shape[0]),"int32")
                    self._face_kd = _load_textures_for_softras(image, faces, textures, is_update)


            else:
                if (self.kd_res == 0):
                    self._face_kd = jt.ones_like(self.face_vertices) * jt.array(self._Kd).float32()
                else:
                    self._face_kd = jt.ones((self.face_vertices.shape[0],self.kd_res,3),"float32") * jt.array(self._Kd).float32()
            self.face_kd_update = False
        return self._face_kd

    def set_vertices(self, transform):  # 未考虑非等比缩放   model_transform
        self._face_vertices = transform(self._face_vertices)
        self._face_normals = transform(self._face_normals)

    def rescaling(self,scale):
        max = jt.max(self.face_vertices,dims=(0,1),keepdims=True)
        min = jt.min(self.face_vertices,dims=(0,1),keepdims=True)
        center = (max + min) / 2
        scale = jt.max(max - min) / scale / 2
        self._face_vertices = (self.face_vertices - center) / scale




