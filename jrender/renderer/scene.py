from tkinter.tix import NoteBook
import numpy as np
import jittor as jt
import os
from render2 import Render


class obj():
    def __init__(self, Ka, Kd, Ke, Ns, Ni,
                 vertices, normals, texcoords,
                 vertices_ind, normals_ind, texcoords_ind,
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

        self.vertices = vertices
        self.vertices_ind = vertices_ind
        self.normals = normals
        self.normals_ind = normals_ind
        self.texcoords = texcoords
        self.texcoords_ind = texcoords_ind

        self.map_Kd_path = map_Kd_path
        self.map_normal_path = map_normal_path
        self.obj_path = obj_path
        self.mtl_path = mtl_path

        self._face_vertices = None
        self.face_vertices_update = True
        self._face_normals = None
        self.face_normals_update = True
        self._face_texcoords = None
        self.face_texcoords_update = True

        self._surface_normals = None
        self.surface_normals_update = True
        self._normal_texture = None
        self.normal_texture_update = True
        self._texture = None
        self.texture_update = True

    @property
    def face_vertices(self):
        if self.face_vertices_update:
            self._face_vertices = self.vertices[self.vertices_ind]
        self.face_vertices_update = False
        return self._face_vertices

    @property
    def face_normals(self):
        if self.face_normals_update:
            if self.normals_ind == None:
                self._face_normals = jt.array()
            else:
                self._face_normals = self.normals[self.normals_ind]
        self.face_normals_update = False
        return self._face_normals

    @property
    def face_texcoords(self):
        if self.face_texcoords_update:
            if self.texcoords_ind == None:
                self._face_texcoords = jt.array()
            else:
                self._face_texcoords = self.texcoords[self.texcoords_ind]
        self.face_texcoords_update = False
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
            self._texture = None
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

    def set_vertices(self, transform):  # 未考虑非等比缩放
        self.vertices = transform(self.vertices)
        self.normals = transform(self.normals)
        self.face_vertices_update = True
        self.face_normals_update = True


class scene():
    def __init__(self, objects, lights, render=Render()):
        self.objects = objects
        self.lights = lights
        self.MRT_update = True
        self._MRT = None
        self.render = render

    @property
    def MRT(self):
        if self.MRT_update:
            worldcoords = jt.array()
            normals = jt.array()
            texcoords = jt.array()
            name_dic = {}
            for i, obj in enumerate(self.objects):
                name_dic[obj.material_name] = i
                face_vertices = obj.face_vertices
                obj_mark = jt.ones_like(face_vertices)*i
                worldcoords = jt.concat(worldcoords, face_vertices, dim=0)
                normals = jt.concat(normals, obj.face_normals)
                texcoords = jt.concat(texcoords, obj.face_texcoords)
            self._MRT = {"worldcoords": worldcoords, "normals":normals,"texcoords":texcoords,"obj_mark": obj_mark, "name_dic": name_dic}
        self.MRT_update = False
        return self._MRT

    def deferred_render(self):
        for obj in self.objects:
            
            obj.set_vertices(self.render.view_transform)
        return

    @property
    def G_buffer(self):
        face_vertices = self.MRT["worldcoords"]
        texture = MRT["obj_mark"].unsqueeze(0)

        specular_Map = jt.transpose(specular_Map.squeeze(0)[:3, :, :], (1, 2, 0))
        return self.render.rasterize(face_vertices, texture)

    @classmethod
    def load_scene(cls, filenames):
        objects = []
        lights = []
        if (isinstance(filenames, list)):
            for filename in filenames:
                new_obj, new_light = load_obj(filename)
                objects += new_obj
                lights += new_light
        else:
            objects = load_obj(filename)
        return cls(objects, lights)


def load_obj(filename):
    objects = []
    lights = []

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

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
        if line.split()[0] == 'vn':
            normals.append([float(vn) for vn in line.split()[1:4]])
        if line.split()[0] == 'vt':
            texcoords.append([float(vt) for vt in line.split()[1:4]])
        if line.split()[0] == 'usemtl':
            material_name = line.split()[1]
            obj_group[material_name].update({"vertices": vertices, "normals": normals, "texcoords": texcoords})
            vertices = []
            normals = []
            texcoords = []
        if line.split()[0] == 'f':
            index = line.split()[1:]
            for ind in index:
                world_ind += index.split('/')[0]
                if len(ind) >= 2:
                    tex_ind += ind.split('/')[1]
                if len(ind) == 3:
                    normal_ind += ind.split('/')[2]
            obj_group[material_name].update({"world_ind": world_ind, "tex_ind": tex_ind, "normal_ind": normal_ind})
            world_ind = []
            tex_ind = []
            normal_ind = []

    f.close()

    material_name = ''
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
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
        faces_world = np.array(obj_group[obj_name].get('world_ind'))
        faces_tex = np.array(obj_group[obj_name].get('tex_ind'))
        faces_normal_ind = np.array(obj_group[obj_name].get('normal_ind'))
        vertices = np.array(obj_group[obj_name].get('vertices'))
        texcoords = np.array(obj_group[obj_name].get('texcoords'))
        normals = np.array(obj_group[obj_name].get('normals'))
        map_Kd = obj_group[obj_name].get('map_Kd')
        map_normal = obj_group[obj_name].get('map_normal')
        Kd = obj_group[obj_name].get('Kd')
        Ka = obj_group[obj_name].get('Ka')
        Ke = obj_group[obj_name].get('Ke')
        Ns = obj_group[obj_name].get('Ns')
        Ni = obj_group[obj_name].get('Ni')

        faces_world = np.reshape(faces_world, len(faces_world)/3, 3)
        faces_world = jt.array(faces_world)
        vertices = np.reshape(vertices, len(vertices)/3, 3)
        vertices = jt.array(vertices)

        if len(faces_tex) != 0:
            faces_tex = np.reshape(faces_tex, len(faces_tex)/3, 3)
            faces_tex = jt.array(faces_tex)
            texcoords = np.reshape(texcoords, len(texcoords)/3, 3)
            texcoords = jt.array(texcoords)

        if len(faces_normal_ind) != 0:
            faces_normal_ind = np.reshape(faces_normal_ind, len(faces_normal_ind)/3, 3)
            faces_normal_ind = jt.array(faces_normal_ind)
            normals = np.reshape(normals, len(normals)/3, 3)
            normals = jt.array(normals)

        new_obj = obj(Ka=Ka, Kd=Kd, Ke=Ke, Ns=Ns, Ni=Ni, vertices=vertices, vertices_ind=faces_world, normals_ind=faces_normal_ind, texcoords_ind=faces_tex,
                      material_name=obj_name, texcoords=texcoords, normals=normals, map_Kd_path=map_Kd, map_normal_path=map_normal, obj_path=filename, mtl_path=filename_mtl)
        if Ke == None:
            objects.append(new_obj)
        else:
            lights.append(new_obj)

    return objects, lights
