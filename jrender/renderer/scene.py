import numpy as np
import jittor as jt
import os
from .render2 import *


class obj():
    def __init__(self, Ka, Kd, Ke, Ns, Ni,
                 face_vertices, face_normals, face_texcoords,
                 material_name,
                 refection_type="diffuse",
                 map_Kd_path=None, map_normal_path=None, obj_path=None, mtl_path=None):
        self.material_name = material_name
        self.Ka = Ka
        self.Kd = Kd
        self.Ke = Ke
        self.Ns = Ns
        self.Ni = Ni
        self.reflection_type = refection_type

        self.map_Kd_path = map_Kd_path
        self.map_normal_path = map_normal_path
        self.obj_path = obj_path
        self.mtl_path = mtl_path
        if isinstance(face_vertices, np.ndarray):
            face_vertices = jt.array(face_vertices)
        if isinstance(face_texcoords, np.ndarray):
            face_texcoords = jt.array(face_texcoords)
        if isinstance(face_normals, np.ndarray):
            face_normals = jt.array(face_normals)
        self._face_vertices = face_vertices
        self._face_normals = face_normals
        self.face_normals_update = True
        self._face_texcoords = face_texcoords

        self._surface_normals = None
        self.surface_normals_update = True
        self._normal_texture = None
        self.normal_texture_update = True
        self._texture = None
        self.texture_update = True
        self.Generate_Normals = "surface"

    @property
    def face_vertices(self):
        return self._face_vertices

    @property
    def face_normals(self):
        if self.face_normals_update:
            if self._face_normals.numel() == 0:
                if self.Generate_Normals == "surface":
                    self._face_normals = jt.ones_like(self.face_vertices) * self.surface_normals.unsqueeze(1)
                elif self.Generate_Normals == "vertex":
                    self._face_normals = None
            else:
                self._face_normals = self.normals[self.normals_ind]
        self.face_normals_update = False
        return self._face_normals

    @property
    def face_texcoords(self):
        return self._face_texcoords

    @property
    def normal_textures(self):
        if self.normal_texture_update:
            self._normal_textures = None
        self.normal_texture_update = False
        return self._normal_textures

    @property
    def textures(self):
        if self.texture_update:
            if self.map_Kd_path is None:
                self._textures = jt.array(self.Kd, "float32").unsqueeze(0).unsqueeze(0)
        self.texture_update = False
        return self._texture

    @property
    def surface_normals(self):
        if self.surface_normals_update:
            if self.normal_textures is None:
                v10 = self.face_vertices[:, 0] - self.face_vertices[:, 1]
                v12 = self.face_vertices[:, 2] - self.face_vertices[:, 1]
                self._surface_normals = jt.normalize(jt.cross(v12, v10), p=2, dim=1, eps=1e-6)
        self.surface_normals_update = False
        return self._surface_normals

    def set_vertices(self, transform):  # 未考虑非等比缩放   model_transform
        self._face_vertices = transform(self._face_vertices)
        self._face_normals = transform(self._face_normals)


class Light():
    def __init__(self, position=[0, 0, 0], direction=[0, 0, 1], color=[1, 1, 1], intensity=0.5, type="directional"):
        self.position = position
        self.direction = direction
        self.color = jt.normalize(jt.array(color, "float32"), dim=0)
        self.intensity = intensity
        self.type = type


class Scene():
    def __init__(self, objects, lights=[], render=Render()):
        self.objects = objects
        self.lights = lights
        self.MRT_update = True
        self._MRT = None
        self.render = render
        self._name_dic = {}
        self.name_dic_update = True

    def set_obj(self):
        self.MRT_update = False
        return

    def set_render(self, render):
        self.render = render

    @property
    def name_dic(self):
        if self.name_dic_update:
            self._name_dic = {}
            for i, obj in enumerate(self.objects):
                self._name_dic[obj.material_name] = i
        self.name_dic_update = False
        return self._name_dic

    @property
    def MRT(self):
        if self.MRT_update:
            worldcoords = jt.array([])
            normals = jt.array([])
            texcoords = jt.array([])
            obj_mark = jt.array([])
            KD = jt.array([])
            name_dic = self.name_dic
            for i, obj in enumerate(self.objects):
                face_vertices = obj.face_vertices
                obj_mark = jt.concat([obj_mark, jt.ones_like(face_vertices)*i], dim=0)
                worldcoords = jt.concat([worldcoords, face_vertices], dim=0)
                normals = jt.concat([normals, obj.face_normals], dim=0)
                if obj.face_texcoords.numel() == 0:
                    kd = jt.ones_like(face_vertices)*obj.Kd
                else:
                    kd = jt.concat([texcoords, sample2D(obj.textures, obj.face_texcoords)], dim=0)
                KD = jt.concat([KD, kd], dim=0)
            self._MRT = {"worldcoords": worldcoords, "normals": normals,
                         "KD": KD, "obj_mark": obj_mark, "name_dic": name_dic, "render_update": [True, True, True, True, True]}
        self.MRT_update = False

        return self._MRT

    def append_light(self,light):
        self.lights.append(light)
        return

    def deferred_render(self):
        image = self.render.fragment_shader(self.MRT, self.objects, self.lights)

        return image

    @classmethod
    def load_scene_from_obj(cls, filenames):
        objects = []
        if (isinstance(filenames, list)):
            for filename in filenames:
                new_obj = load_obj(filename)
                objects += new_obj
        else:
            objects = load_obj(filenames)
        return cls(objects)


def load_obj(filename):
    objects = []
    obj_group = {}
    vertices = []
    texcoords = []
    normals = []
    world_ind = []
    tex_ind = []
    normal_ind = []
    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('mtllib'):
            filename_mtl = os.path.join(os.path.dirname(filename), line.split()[1])

    material_name = ""
    length = len(lines)
    for i, line in enumerate(lines):
        if len(line.split()) == 0 :
            if i == length - 1 :
                line = "usemtl end" 
            else :
                continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
        if line.split()[0] == 'vn':
            normals.append([float(vn) for vn in line.split()[1:4]])
        if line.split()[0] == 'vt':
            texcoords.append([float(vt) for vt in line.split()[1:4]])

        if line.split()[0] == 'f':
            index = line.split()[1:]
            for ind in index:
                if len(ind.split('/')) >= 2:
                    world_ind.append(int(ind.split('/')[0]))
                else:
                    world_ind.append(int(ind))
                if len(ind.split('/')) >= 2:
                    tex_ind.append(int(ind.split('/')[1]))
                if len(ind.split('/')) == 3:
                    normal_ind.append(int(ind.split('/')[2]))

        if line.split()[0] == 'usemtl' or i == length - 1:
            next_name = line.split()[1]
            if material_name == "":
                material_name = next_name
                continue

            faces_world_ind = np.reshape(world_ind, (int(len(world_ind)/3), 3))
            faces_world_ind = jt.array(faces_world_ind)-1
            face_vertices = jt.array(vertices)[faces_world_ind]

            faces_tex_ind = np.reshape(tex_ind, (int(len(tex_ind)/3), 3))
            faces_tex_ind = jt.array(faces_tex_ind)-1
            face_texcoords = jt.array(texcoords)[faces_tex_ind]

            faces_normal_ind = np.reshape(normal_ind, (int(len(normal_ind)/3), 3))
            faces_normal_ind = jt.array(faces_normal_ind)-1
            face_normals = jt.array(normals)[faces_normal_ind]

            obj_group[material_name] = ({"face_vertices": face_vertices,
                                        "face_texcoords": face_texcoords, "face_normals": face_normals})
            material_name = next_name
            world_ind = []
            tex_ind = []
            normal_ind = []

    f.close()

    next = 0
    material_name = ''
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
                    if material_name not in obj_group.keys():
                        next = 1
                        continue
                    else :
                        next = 0
                if next :
                    continue
                if line.split()[0] == 'map_Kd':
                    obj_group[material_name].update({"map_Kd": line.split()[1]})
                if line.split()[0] == 'map_normal':
                    obj_group[material_name].update({"map_normal": line.split()[2]})
                if line.split()[0] == 'Kd':
                    obj_group[material_name].update({"Kd": list(map(float, line.split()[1:4]))})
                if line.split()[0] == 'Ka':
                    obj_group[material_name].update({"Ka": list(map(float, line.split()[1:4]))})
                if line.split()[0] == 'Ke':
                    obj_group[material_name].update({"Ke": list(map(float, line.split()[1:4]))})
                if line.split()[0] == 'Ns':
                    obj_group[material_name].update({"Ns": line.split()[1]})
                if line.split()[0] == 'Ni':
                    obj_group[material_name].update({"Ni": line.split()[1]})

    f.close()

    for obj_name in obj_group.keys():
        face_vertices = np.array(obj_group[obj_name].get('face_vertices'))
        face_texcoords = np.array(obj_group[obj_name].get('face_texcoords'))
        face_normals = np.array(obj_group[obj_name].get('face_normals'))
        map_Kd = obj_group[obj_name].get('map_Kd')
        map_normal = obj_group[obj_name].get('map_normal')
        Kd = obj_group[obj_name].get('Kd')
        Ka = obj_group[obj_name].get('Ka')
        Ke = obj_group[obj_name].get('Ke')
        Ns = obj_group[obj_name].get('Ns')
        Ni = obj_group[obj_name].get('Ni')

        vertices = jt.array(vertices)
        texcoords = jt.array(texcoords)
        normals = jt.array(normals)

        new_obj = obj(Ka=Ka, Kd=Kd, Ke=Ke, Ns=Ns, Ni=Ni, face_vertices=face_vertices, material_name=obj_name, face_texcoords=face_texcoords,
                      face_normals=face_normals, map_Kd_path=map_Kd, map_normal_path=map_normal, obj_path=filename, mtl_path=filename_mtl)
        objects.append(new_obj)

    return objects
