
from typing import List
import jittor as jt
from .objects import *
from ..render2 import *
import os

from jrender.Scene import objects

class Scene():
    objects: List[obj]
    lights: List[Light]

    def __init__(self, objects=[], lights=[], render=Render()):
        self.objects = objects
        self.lights = lights
        self.MRT_update = True
        self._MRT = None
        self.render = render
        self._name_dic = {}
        self.name_dic_update = True
        self.render_target = [i for i in range(len(objects))]
        self.print_scene()

    def set_obj(self):
        self.MRT_update = False
        return

    def set_render(self, render):
        self.render = render

    def set_kd_res(self,res):
        for obj in self.objects:
            obj.kd_res = res

    def set_roughness(self,ind,roughness):
        if isinstance(ind,list):
            for _ind in ind:
                    self.objects[_ind]._roughness = roughness
        else:
            self.objects[ind]._roughness = roughness

    def set_render_target(self, index):
        if isinstance(index, list):
            self.render_target = index
        else:
            self.render_target = [index]
        return

    def set_reflection(self, ind, type):
        self.objects[ind].reflection_type = type
        return

    def set_specular(self,ind,with_specular):
        if isinstance(ind,list):
            for _ind in ind:
                self.objects[_ind].with_specular = with_specular
        else:
            self.objects[ind].with_specular = with_specular

    def set_GenerateNormal(self,ind,mode):
        if isinstance(ind,list):
            for _ind in ind:
                self.objects[_ind].Generate_Normals = mode
        else:
            self.objects[ind].Generate_Normals = mode

    def set_rescaling(self,ind,scale):
        if isinstance(ind,list):
            for _ind in ind:
                self.objects[_ind].rescaling(scale)
        else:
            self.objects[ind].rescaling(scale)

    def print_scene(self):
        print("Scene:")
        for name in self.name_dic.keys():
            print(f"name:{name} ind:{self.name_dic[name]}")

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
            KD = jt.array([])
            Albedo = jt.array([])
            Metallic = jt.array([])
            Roughness = jt.array([])
            with_specular = jt.array([])
            for i in self.render_target:
                obj = self.objects[i]
                face_vertices = obj.face_vertices
                worldcoords = jt.concat([worldcoords, face_vertices], dim=0)
                normals = jt.concat([normals, obj.face_normals], dim=0)
                Albedo = jt.concat([Albedo, obj.face_albedo], dim=0)
                Metallic = jt.concat([Metallic, obj.face_metallic], dim=0)
                Roughness = jt.concat([Roughness, obj.face_roughness], dim=0)
                with_specular = jt.concat([with_specular, obj.specular], dim=0)
                KD = jt.concat([KD, obj.face_kd], dim=0)
            metallic_roughness = jt.concat([Metallic, Roughness, with_specular], dim=2)
            self._MRT = MultipleRenderTargets(worldcoords=worldcoords, normals=normals,
                                              KD=KD, albedo=Albedo, metallic_roughness=metallic_roughness)
            self.MRT_update = False

        return self._MRT

    def GeometryDesc(self):
        obj_faces = {}
        name_dic = self.name_dic
        nf = 0
        for i in self.render_target:
            obj = self.objects[i]
            face_vertices = obj.face_vertices
            obj_faces.update({f"{i}": [nf, nf + face_vertices.shape[0]]})
            nf += face_vertices.shape[0]
        return GeometryDescption(obj_faces, name_dic)

    def MaterialDesc(self,PBR):
        objects = []
        for i in self.render_target:
            obj = self.objects[i]
            objects.append(obj)
        return MaterialDescption(objects,PBR=PBR)

    def IlluminationDesc(self,shading):
        return IlluminationDescption(self.lights,shading=shading)

    def append_light(self, lights):
        if isinstance(lights, list):
            for light in lights:
                self.lights.append(light)
        else:
            self.lights.append(light)
        return

    def deferred_render(self):
        self.render.MRT = self.MRT
        GeometryDesc=self.GeometryDesc()
        IlluminationDesc=self.IlluminationDesc("Cook_Torrance")
        MaterialDesc=self.MaterialDesc(PBR=True)
        self.render.GeometryDesc = GeometryDesc
        self.render.MaterialDesc = MaterialDesc
        self.render.IlluminationDesc = IlluminationDesc
        image = self.render.fragment_shader()
        return image

    @classmethod
    def load_scene_from_obj(cls, filenames):
        objects = []
        if (isinstance(filenames, list)):
            for filename in filenames:
                if filename.split('.')[-1] == 'obj':
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
        if len(line.split()) == 0:
            if i == length - 1:
                line = "usemtl end"
            else:
                continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
        if line.split()[0] == 'vn':
            normals.append([float(vn) for vn in line.split()[1:4]])
        if line.split()[0] == 'vt':
            texcoords.append([float(vt) for vt in line.split()[1:4]])

        if line.split()[0] == 'f':
            index = line.split()[1:]
            if (len(index) == 4):
                index = index[:3] + index[2:] + [index[0]]
            for ind in index:
                v = ind.split('/')
                if len(v) >= 2:
                    world_ind.append(int(v[0]))
                else:
                    world_ind.append(int(ind))
                if len(v) >= 2 and v[1] != '':
                    tex_ind.append(int(v[1]))
                if len(v) == 3 and v[2] != '':
                    normal_ind.append(int(v[2]))

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
                    else:
                        next = 0
                if next:
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
        if map_Kd is not None:
            map_Kd =  os.path.join(os.path.dirname(filename), map_Kd)
        map_normal = obj_group[obj_name].get('map_normal')
        if map_normal is not None:
            map_normal =  os.path.join(os.path.dirname(filename), map_normal)
        Kd = obj_group[obj_name].get('Kd')
        Ka = obj_group[obj_name].get('Ka')
        Ke = obj_group[obj_name].get('Ke')
        Ns = obj_group[obj_name].get('Ns')
        Ni = obj_group[obj_name].get('Ni')

        vertices = jt.array(vertices)
        texcoords = jt.array(texcoords)
        normals = jt.array(normals)

        new_obj = obj(Ka=Ka, Kd=Kd, Ke=Ke, Ns=Ns, Ni=Ni, face_vertices=face_vertices, material_name=obj_name, kd_texture_uv=face_texcoords,
                      face_normals_from_obj=face_normals, map_Kd_path=map_Kd, map_normal_path=map_normal, obj_path=filename, mtl_path=filename_mtl)
        objects.append(new_obj)

    return objects

